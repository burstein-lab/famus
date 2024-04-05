import multiprocessing as mp
import os
import pickle
import random
import subprocess
from typing import Generator

import numpy as np
import torch

from app import logger
from app.sdf import SparseDataFrame


class SDFloader:
    """
    Class for loading triplets from SparseDataFrame. Triplet indices are created by randomly sampling from the
    SparseDataFrame. The triplets are created in a way that the anchor and positive samples are from the same class
    and the negative sample is from a different class. The number of triplets created for each class is limited by
    examples_per_class_limit. Sets of indices (load stacks) are used to load the next batch of triplets iteratively
    as they become necessary. The loader can supply triplets for training the neural network in batches, but skips
    the last batch if it is smaller than the batch size.
    The purpose of this class is to reduce the memory usage when training the model.
    """

    def __init__(
        self,
        sdf: SparseDataFrame,
        leftovers_sequence_ids: dict,  # @TODO: add preferential sampling for leftovers
        triplets_per_class=3000,
        triplets_per_leftover=10,
        max_leftover_triplets=300,
        load_stack_size=100000,
        nthreads=2,
    ) -> None:
        logger.info("Creating triplets")
        self.load_stack_min_len = load_stack_size
        sdf_sample_indices = set(range(len(sdf)))
        label_to_indices = sdf.get_label_to_indices()
        index_ids_to_indices = {idx: i for i, idx in enumerate(sdf.index_ids)}
        if leftovers_sequence_ids and triplets_per_leftover > 0:
            label_to_leftover_indices = {
                label: [index_ids_to_indices[idx_id] for idx_id in idx_ids]
                for label, idx_ids in leftovers_sequence_ids.items()
            }
        else:
            label_to_leftover_indices = {}

        manager = mp.Manager()
        label_to_indices = manager.dict(label_to_indices)
        pool = mp.Pool(nthreads)
        labels = sdf.unique_labels()

        triplets = []
        self.idx_load_stacks = []  # list of sets of indices to load iteratively to limit memory usage
        args = []
        for label in labels:
            if label == "unknown":
                continue
            leftover_indices = (
                label_to_leftover_indices[label]
                if label in label_to_leftover_indices
                else []
            )
            args.append(
                (
                    label_to_indices,
                    label,
                    sdf_sample_indices,
                    leftover_indices,
                    triplets_per_class,
                    triplets_per_leftover,
                    max_leftover_triplets,
                )
            )
        triplets = pool.starmap(
            SDFloader.sample_triplets,
            args,
        )
        pool.close()
        pool.join()
        del label_to_indices
        triplets = [item for sublist in triplets for item in sublist]

        logger.info("Created " + str(len(triplets)) + " triplets")
        self.sdf = sdf
        self.anchors = []
        self.positives = []
        self.negatives = []
        self.triplets_per_stack = []
        logger.info("Shuffling samples")
        random.shuffle(triplets)
        curr_load_stack = set()
        logger.info("Creating load stacks")
        triplet_index = 0
        self.stack_start_indices = [0]

        for a, p, n in triplets:
            curr_load_stack.add(a)
            self.anchors.append(a)
            curr_load_stack.add(p)
            self.positives.append(p)
            curr_load_stack.add(n)
            self.negatives.append(n)
            triplet_index += 1
            if len(curr_load_stack) >= load_stack_size:
                logger.info("Adding load stack")
                self.idx_load_stacks.append(curr_load_stack)
                self.triplets_per_stack.append(int(len(curr_load_stack) / 3))
                self.stack_start_indices.append(triplet_index)
                curr_load_stack = set()

        if len(curr_load_stack) > 0:
            self.idx_load_stacks.append(curr_load_stack)
            self.triplets_per_stack.append(int(len(curr_load_stack) / 3))

        del triplets
        logger.info("Created " + str(len(self.idx_load_stacks)) + " load stacks")

    def triplet_batch_generator(
        self, batch_size=32
    ) -> Generator[torch.Tensor, None, None]:
        assert batch_size <= self.load_stack_min_len, (
            "batch_size is too small, increase it to at least "
            + str(self.load_stack_min_len)
            + " or decrease load_stack_size"
        )
        for stack_index, stack in enumerate(self.idx_load_stacks):
            stack = list(stack)
            triplets_start_index = self.stack_start_indices[stack_index]
            total_curr_stack_iterations = int(
                np.floor(self.triplets_per_stack[stack_index] / batch_size)
            )
            curr_tensor = torch.tensor(self.sdf.matrix[stack].todense())
            matrix_idx_to_tensor_idx = {idx: i for i, idx in enumerate(stack)}
            for curr_stack_iteration in range(total_curr_stack_iterations):
                s = triplets_start_index + curr_stack_iteration * batch_size
                e = triplets_start_index + (curr_stack_iteration + 1) * batch_size
                anchor_ids = self.anchors[s:e]
                positive_ids = self.positives[s:e]
                negative_ids = self.negatives[s:e]
                anchor_ids = [matrix_idx_to_tensor_idx[idx] for idx in anchor_ids]
                positive_ids = [matrix_idx_to_tensor_idx[idx] for idx in positive_ids]
                negative_ids = [matrix_idx_to_tensor_idx[idx] for idx in negative_ids]
                yield (
                    curr_tensor[anchor_ids],
                    curr_tensor[positive_ids],
                    curr_tensor[negative_ids],
                )

    def get_num_batches(self, batch_size: int):
        """
        Returns the number of batches that can be created from the triplets given the batch size.
        The number of batches refers to the number of batches for the neural network, not the number of load stacks.
        :param batch_size: size of the batch
        :return: number of batches
        """
        if not isinstance(batch_size, int) and batch_size > 0:
            raise ValueError("batch_size must be a positive integer")
        return sum(
            int(np.floor(self.triplets_per_stack[i] / batch_size))
            for i in range(len(self.idx_load_stacks))
        )

    @staticmethod
    def sample_triplets(
        label_to_indices: dict,
        label: str,
        all_indices: set,
        leftovers_sequence_indices: list,
        num_triplets: int,
        num_triplets_per_leftover: int,
        max_leftover_triplets: int,
    ) -> list:
        """
        Creates triplets for a given label. The triplets are created in a way that the anchor and positive samples are
        from the same class and the negative sample is from a different class. The number of triplets created for each
        class is limited by examples_per_class_limit.
        :param label_to_indices: dictionary mapping labels to indices
        :param label: label for which to create triplets
        :param all_indices: set of all indices
        :param examples_per_class_limit: maximum number of triplets to create for each class
        :return: list of triplets
        """
        if label == "unknown":
            return []  # do not create triplets for unknown class
        indices = label_to_indices[label]
        other_indices = list(all_indices - set(indices))
        triplets = []
        # sample for leftovers
        if leftovers_sequence_indices:
            ancors = np.array(
                list(leftovers_sequence_indices) * num_triplets_per_leftover
            )
            if len(ancors) > max_leftover_triplets:
                ancors = np.random.choice(
                    a=ancors, size=max_leftover_triplets, replace=False
                )
            positives = np.random.choice(
                a=indices,
                size=len(ancors),
                replace=True,
            )
            negatives = np.random.choice(
                a=other_indices,
                size=len(ancors),
                replace=True,
            )
            for ancor_idx, positive_idx, negative_idx in zip(
                ancors, positives, negatives
            ):
                triplets.append((ancor_idx, positive_idx, negative_idx))

            examples_per_class_limit = max(0, num_triplets - len(triplets))
            logger.info(
                "Created "
                + str(len(triplets))
                + " leftover triplets for label: "
                + str(label)
                + "..."
            )
            if examples_per_class_limit == 0:
                return triplets
        else:
            examples_per_class_limit = num_triplets
        ancors = np.random.choice(
            a=indices, size=examples_per_class_limit, replace=True
        )
        positives = np.random.choice(
            a=indices, size=examples_per_class_limit, replace=True
        )
        negatives = np.random.choice(
            a=other_indices, size=examples_per_class_limit, replace=True
        )
        for ancor_idx, positive_idx, negative_idx in zip(ancors, positives, negatives):
            triplets.append((ancor_idx, positive_idx, negative_idx))
        logger.info(
            "Created "
            + str(len(triplets))
            + " triplets total for label: "
            + str(label)
            + "..."
        )
        return triplets


def prepare_sdfloader(
    sdf_train_path: str,
    leftovers_dir: str,
    nthreads: int,
    triplets_per_class: int,
    triplets_per_leftover: int,
    output_path: str,
    load_stack_size=100000,
) -> None:
    """
    Creates SDFloader object and saves it to a file.
    :param sdf_train_path: path to SparseDataFrame file
    :param nthreads: number of threads to use
    :param num_examples_per_class: maximum number of triplets to create for each class
    :param output_path: path to output file
    :param load_stack_size: number of triplets to load at once
    :return: None
    """
    with open(sdf_train_path, "rb") as f:
        sdf_train = pickle.load(f)
    leftovers_sequence_ids = {}
    if leftovers_dir and triplets_per_leftover > 0:
        for path in [os.path.join(leftovers_dir, f) for f in os.listdir(leftovers_dir)]:
            ids = subprocess.check_output(
                f"seqkit seq -in < {path}", shell=True
            ).decode("utf-8")
            ids = ids.strip().split("\n")
            leftovers_sequence_ids[
                os.path.basename(path).removesuffix(".leftovers.fasta")
            ] = ids

    # logger.info(f"TO DELETE::: Leftover ids example: {str(ids)}")

    sdfloader = SDFloader(
        sdf_train,
        leftovers_sequence_ids=leftovers_sequence_ids,
        triplets_per_class=triplets_per_class,
        triplets_per_leftover=triplets_per_leftover,
        load_stack_size=load_stack_size,
        nthreads=nthreads,
    )

    # sdf_train = pickle.load(
    #     open(
    #         "data/kegg_2021_coverage_0.8/sdf_train.pkl",
    #         "rb",
    #     )
    # )

    # sdfloader = pickle.load(
    #     open(
    #         "/sternadi/nobackup/volume1/guy_shur/kegg_2021_coverage_0.8_sdfloader.pkl",
    #         "rb",
    #     )
    # )

    # import random

    # c = 0
    # for a, p, n in sdfloader.triplet_batch_generator():
    #     a, p, n = a[0], p[0], n[0]
    #     if random.random() > 0.95:
    #         a_p_corr = np.corrcoef(a, p)[0][1]
    #         a_n_corr = np.corrcoef(a, n)[0][1]
    #         logger.info(f"Correlation between anchor and positive: {a_p_corr}")
    #         logger.info(f"Correlation between anchor and negative: {a_n_corr}")

    # for a, p, n in zip(sdfloader.anchors, sdfloader.positives, sdfloader.negatives):
    #     if random.random() > 0.95:
    #         a_idx, p_idx, n_idx = (
    #             sdf_train.index_ids[a],
    #             sdf_train.index_ids[p],
    #             sdf_train.index_ids[n],
    #         )
    #         a_labels = sdf_train.labels[a_idx]
    #         p_labels = sdf_train.labels[p_idx]
    #         n_labels = sdf_train.labels[n_idx]
    #         logger.info(f"Anchor labels: {a_labels}")
    #         logger.info(f"Positive labels: {p_labels}")
    #         logger.info(f"Negative labels: {n_labels}")

    with open(output_path, "wb+") as f:
        pickle.dump(sdfloader, f)
