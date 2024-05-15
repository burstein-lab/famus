import re
import multiprocessing as mp
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import uuid
from collections import namedtuple

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

from app import get_cfg, logger, now
from app.fasta_parser import FastaParser
from app.sdf import from_sparse_dict
from app.utils import concatenate_files, even_split

SUBCLUSTER_SUFFIX_PATTERN = r"\.sub_cluster\.cluster\.\d+\.fasta$"
SPLIT_SUBCLUSTER_SUFFIX_PATTERN = r"\.sub_cluster\.cluster\.\d+\.\d+\.fasta$"
SUBCLUSTER_PROFILE_SUFFIX_PATTERN = r"\.sub_cluster\.cluster\.\d+$"
PreparePhmmArgs = namedtuple("PreparePhmmArgs", "path fasta_path")
SplitHMMSearchArgs = namedtuple(
    "SplitHMMSearchArgs", "profile_path file_for_scoring output_path"
)
cfg = get_cfg()
nthreads = cfg["nthreads"]
mmseq_nthreads = cfg["threads_per_mmseqs_job"]
number_of_sampled_sequences_per_subcluster = cfg[
    "number_of_sampled_sequences_per_subcluster"
]
fraction_of_sampled_unknown_sequences = cfg["fraction_of_sampled_unknown_sequences"]
samples_profiles_product_limit = cfg["samples_profiles_product_limit"]


def validate_dir_paths(paths: list) -> tuple:
    """
    Validates that the directory paths exist and
    returns them as a tuple with a trailing slash if they don't already have one.
    :param paths: list of paths to validate
    :return: tuple of validated paths
    """
    paths = tuple(p + "/" if not p.endswith("/") else p for p in paths)
    validate_paths(paths)
    return paths


def create_paths_if_not_exists(paths: list) -> tuple:
    """
    Creates directories if they do not exist.
    :param paths: list of paths to create
    :return: None
    """
    paths = tuple(p + "/" if not p.endswith("/") else p for p in paths)
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    return paths


def validate_paths(paths: list) -> None:
    """
    Validates that the paths exist.
    :param paths: list of paths to validate
    :return: None
    """
    for path in paths:
        if not os.path.exists(path):
            raise ValueError("path {} does not exist".format(path))


def get_subcluster_fasta_paths(path: str) -> list:
    """
    Returns a list of paths to subcluster fasta files in the given directory.
    :param path: path to directory containing subcluster fasta files
    :return: list of paths to subcluster fasta files
    """
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if re.search(SUBCLUSTER_SUFFIX_PATTERN, f)
    ]


def get_split_subcluster_fasta_paths(path: str) -> list:
    """
    Returns a list of paths to split subcluster fasta files in the given directory.
    :param path: path to directory containing split subcluster fasta files
    :return: list of paths to split subcluster fasta files
    """
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if re.search(SPLIT_SUBCLUSTER_SUFFIX_PATTERN, f)
    ]


def generate_ground_truth(
    input_augmented_subclusters_fastas_dir_path: str,
    input_leftovers_dir_path: str,
    input_unannotated_sequences_fasta_path: str,
    output_ground_truth_path: str,
) -> None:
    """
    Generates a ground truth dictionary for the given orthologs and unannotated sequences.
    The keys are the sequence IDs and the values are the orthology names.
    :param input_orthologs_fasta_dir_path: path to directory containing ortholog fasta files
    :param input_unannotated_sequences_fasta_path: path to fasta file containing unannotated sequences
    :param output_ground_truth_path: path to output ground truth pickle file
    :return: None
    """
    (
        input_augmented_subclusters_fastas_dir_path,
        input_leftovers_dir_path,
    ) = validate_dir_paths(
        [input_augmented_subclusters_fastas_dir_path, input_leftovers_dir_path]
    )
    if input_unannotated_sequences_fasta_path:
        validate_paths([input_unannotated_sequences_fasta_path])

    labeled_sequences_ground_truth = {}
    augmented_subcluster_fasta_paths = [
        os.path.join(input_augmented_subclusters_fastas_dir_path, f)
        for f in os.listdir(input_augmented_subclusters_fastas_dir_path)
    ]
    # leftover_subcluster_fasta_paths = [
    #     os.path.join(input_leftovers_dir_path, f)
    #     for f in os.listdir(input_leftovers_dir_path)
    # ]
    all_fasta_paths = (
        augmented_subcluster_fasta_paths  # + leftover_subcluster_fasta_paths
    )

    assert all(f.endswith(".fasta") for f in all_fasta_paths)
    for fasta_path in all_fasta_paths:
        orthology_name = re.sub(
            SUBCLUSTER_SUFFIX_PATTERN, "", os.path.basename(fasta_path)
        ).removesuffix(".leftovers.fasta")
        with open(fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id not in labeled_sequences_ground_truth:
                    labeled_sequences_ground_truth[record.id] = ""
                else:
                    labeled_sequences_ground_truth[record.id] += ";"
                labeled_sequences_ground_truth[record.id] += orthology_name

    if input_unannotated_sequences_fasta_path:
        unlabeled_sequences_ground_truth = {}
        with open(input_unannotated_sequences_fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id in labeled_sequences_ground_truth:
                    raise ValueError(
                        "Sequence {} is in both annotated and unannotated sequences".format(
                            record.id
                        )
                    )
                if record.id in unlabeled_sequences_ground_truth:
                    logger.warning(
                        "Sequence {} is in the unannotated sequences more than once".format(
                            record.id
                        )
                    )
                unlabeled_sequences_ground_truth[record.id] = "unknown"

    output = labeled_sequences_ground_truth | unlabeled_sequences_ground_truth
    with open(output_ground_truth_path, "wb+") as f:
        pickle.dump(output, f)


def prepare_full_hmmsearch_input(
    data_dir_path: str,
    augmented_subcluster_dir: str,
    leftovers_dir: str,
    tmp_dir_path: str,
    input_unknown_sequences_fasta_path: str,
    output_full_hmmsearch_input_path: str,
    number_of_sampled_sequences_per_subcluster=number_of_sampled_sequences_per_subcluster,
    fraction_of_sampled_unknown_sequences=fraction_of_sampled_unknown_sequences,
    samples_profiles_product_limit=samples_profiles_product_limit,
):
    labeled_ids_per_cluster = []
    labeled_ids_per_leftover = []
    for fasta_path in get_subcluster_fasta_paths(augmented_subcluster_dir):
        labeled_ids_per_cluster.append(get_fasta_ids(fasta_path))
    for fasta_path in get_subcluster_fasta_paths(leftovers_dir):
        labeled_ids_per_leftover.append(get_fasta_ids(fasta_path))

    if not fraction_of_sampled_unknown_sequences == "do_not_use":
        # check for overlap between unknown sequences and labeled sequences
        unlabeled_ids = get_fasta_ids(input_unknown_sequences_fasta_path)
        labeled_ids = [
            item for sublist in labeled_ids_per_cluster for item in sublist
        ] + [item for sublist in labeled_ids_per_leftover for item in sublist]
        both_labeled_and_unlabeled = set(labeled_ids) & set(unlabeled_ids)
        if both_labeled_and_unlabeled:
            raise ValueError(
                f"{both_labeled_and_unlabeled} are in both the unknown label file and in one of the labeled fasta files"
            )

    # calculate the total number of sequences that will be sampled
    if number_of_sampled_sequences_per_subcluster == "use_all":
        total_sampled_sequences = sum(
            [len(ids) for ids in labeled_ids_per_cluster + labeled_ids_per_leftover]
        )

    elif (
        isinstance(number_of_sampled_sequences_per_subcluster, int)
        and number_of_sampled_sequences_per_subcluster > 0
    ):
        total_sampled_sequences = sum(
            [
                min(len(ids), number_of_sampled_sequences_per_subcluster)
                for ids in labeled_ids_per_cluster
            ]
        ) + sum(len(ids) for ids in labeled_ids_per_leftover)
    else:
        raise ValueError(
            "number_of_sampled_sequences_per_subcluster must be 'use_all' or a positive integer"
        )

    if isinstance(fraction_of_sampled_unknown_sequences, float):
        total_sampled_sequences += (
            total_sampled_sequences * fraction_of_sampled_unknown_sequences
        )
    elif fraction_of_sampled_unknown_sequences == "use_all":
        total_sampled_sequences += len(
            get_fasta_ids(input_unknown_sequences_fasta_path)
        )
    number_of_profiles = len(labeled_ids_per_cluster)
    hmmsearch_load_size = total_sampled_sequences * number_of_profiles
    total_number_of_leftover_seqs = sum([len(ids) for ids in labeled_ids_per_leftover])
    number_of_sampled_sequences_per_leftover = (
        None if total_number_of_leftover_seqs > 0 else 0
    )
    # adjust the total number of sequences to be sampled if the product of the number of sampled sequences and the number of profiles is greater than the limit
    if hmmsearch_load_size > samples_profiles_product_limit:
        logger.warning(
            f"the product of the number of sampled sequences and the number of profiles is {hmmsearch_load_size}, which is greater than the limit of {samples_profiles_product_limit}. Automatically choosing sample sizes."
        )
        sampled_sequences_limit = samples_profiles_product_limit // number_of_profiles
        total_number_of_subcluster_seqs = sum(
            [len(ids) for ids in labeled_ids_per_cluster]
        )
        total_number_of_leftover_seqs = sum(
            [len(ids) for ids in labeled_ids_per_leftover]
        )

        total_sampled_leftover_sequences = min(
            total_number_of_leftover_seqs, sampled_sequences_limit * 0.25
        )
        number_of_sampled_sequences_per_leftover = int(
            total_sampled_leftover_sequences // number_of_profiles
        )

        total_sampled_subcluster_sequences = min(
            total_number_of_subcluster_seqs,
            sampled_sequences_limit - total_sampled_leftover_sequences,
        )
        number_of_sampled_sequences_per_subcluster = int(
            total_sampled_subcluster_sequences // number_of_profiles
        )
        if number_of_sampled_sequences_per_subcluster < 6:
            number_of_sampled_sequences_per_subcluster = 6

        if (
            number_of_sampled_sequences_per_leftover < 2
            and number_of_sampled_sequences_per_leftover > 0
        ):
            number_of_sampled_sequences_per_leftover = 2
        # TODO: remove leftover sampling if the current version works out
        # TODO: delete the line below
        number_of_sampled_sequences_per_leftover = 0
        # TODO: delete the line above
        logger.info(
            f"number of sampled sequences per subcluster: {number_of_sampled_sequences_per_subcluster}"
        )
        logger.info(
            f"number of sampled sequences per leftover: {number_of_sampled_sequences_per_leftover}"
        )

    sampled_subclusters_fasta_path = os.path.join(
        data_dir_path, "sampled_subclusters.fasta"
    )
    if isinstance(number_of_sampled_sequences_per_subcluster, int):
        logger.info("sampling subclusters for model")
        sample_subclusters_for_model(
            augmented_subcluster_dir,
            output_sampled_subclusters_fasta_path=sampled_subclusters_fasta_path,
            number_of_sampled_sequences_per_subcluster=number_of_sampled_sequences_per_subcluster,
        )
    else:
        logger.info("using all subclusters for model")
        concatenate_files(
            files=[
                augmented_subcluster_dir + f
                for f in os.listdir(augmented_subcluster_dir)
            ],
            output=sampled_subclusters_fasta_path,
            track_progress=False,
        )
    os.system(
        "cat "
        + sampled_subclusters_fasta_path
        + " > "
        + output_full_hmmsearch_input_path
    )
    n_sampled_subcluster_sequences = int(
        subprocess.check_output(
            "grep -c '>' " + sampled_subclusters_fasta_path, shell=True
        )
    )

    sampled_leftovers_fasta_path = os.path.join(
        data_dir_path, "sampled_leftovers.fasta"
    )
    if number_of_sampled_sequences_per_leftover is None or (
        isinstance(number_of_sampled_sequences_per_leftover, int)
        and number_of_sampled_sequences_per_leftover > 0
    ):
        if (
            isinstance(number_of_sampled_sequences_per_leftover, int)
            and number_of_sampled_sequences_per_leftover > 0
        ):
            logger.info("sampling leftovers for model")
            sample_subclusters_for_model(
                leftovers_dir,
                output_sampled_subclusters_fasta_path=sampled_leftovers_fasta_path,
                number_of_sampled_sequences_per_subcluster=number_of_sampled_sequences_per_leftover,
            )

        else:
            logger.info("using all leftovers for model")

            concatenate_files(
                files=[
                    os.path.join(leftovers_dir, f) for f in os.listdir(leftovers_dir)
                ],
                output=sampled_leftovers_fasta_path,
                track_progress=False,
            )
        os.system(
            "cat "
            + sampled_leftovers_fasta_path
            + " >> "
            + output_full_hmmsearch_input_path
        )

        n_leftover_sequences = int(
            subprocess.check_output(
                "grep -c '>' " + sampled_leftovers_fasta_path, shell=True
            )
        )
    elif number_of_sampled_sequences_per_leftover == 0:
        logger.info("not using leftovers for model")
        n_leftover_sequences = 0
    else:
        raise ValueError(
            f"number_of_sampled_sequences_per_leftover must be a nonnegative integer or None, not {number_of_sampled_sequences_per_leftover}"
        )
    n_total_sequences = n_sampled_subcluster_sequences + n_leftover_sequences

    if fraction_of_sampled_unknown_sequences == "use_all":
        logger.info("using all unknown sequences for model")
        os.system(
            "cat "
            + input_unknown_sequences_fasta_path
            + " >> "
            + output_full_hmmsearch_input_path
        )
    elif fraction_of_sampled_unknown_sequences == "do_not_use":
        logger.info("not using unknown sequences for model")
    else:
        logger.info("sampling unknown sequences for model")
        sampled_unknown_sequences_fasta_path = os.path.join(
            data_dir_path, "sampled_unknown_sequences.fasta"
        )
        sample_unknown_sequences_for_model(
            input_unknown_sequences_fasta_path=input_unknown_sequences_fasta_path,
            n_sequences=int(n_total_sequences * fraction_of_sampled_unknown_sequences),
            tmp_dir_path=tmp_dir_path,
            output_sampled_unknown_sequences_fasta_path=sampled_unknown_sequences_fasta_path,
        )
        os.system(
            "cat "
            + sampled_unknown_sequences_fasta_path
            + " >> "
            + output_full_hmmsearch_input_path
        )


def hmmsearch_results_to_train_sdf(
    input_split_hmmsearch_results_path: str,
    input_split_subcluster_md_path: str,
    input_full_hmmsearch_results_path: str,
    input_full_profiles_dir_path: str,
    input_all_sequences_fasta_path: str,
    input_ground_truth_path: str,
    output_sdf_path: str,
) -> None:
    """
    Creates a sparse dataframe from the hmmsearch results.
    Parses the hmmsearch results to a dictionary, where the keys are tuples of (sequence ID, profile ID) and the values are the scores.
    For sequence x profile pairs where the sequence belongs to the subcluster profile, the score is subsituted with the score on the split profile that did not contain that sequence to prevent overfitting.
    Populates the dictionary with zeros for sequence x profile pairs that are missing from the hmmsearch results.
    Creates a sparse dataframe from the dictionary.
    :param input_split_hmmsearch_results_path: path to hmmsearch results for split profiles
    :param input_split_subcluster_md_path: path to metadata for split profiles
    :param input_full_hmmsearch_results_path: path to hmmsearch results for full profiles
    :param input_full_profiles_dir_path: path to directory containing full profiles
    :param input_all_sequences_fasta_path: path to fasta file containing all sequences
    :param input_ground_truth_path: path to ground truth pickle file
    :param output_sdf_path: path to output sparse dataframe pickle file
    :return: None
    """
    validate_paths(
        [
            input_split_hmmsearch_results_path,
            input_split_subcluster_md_path,
            input_full_hmmsearch_results_path,
            input_full_profiles_dir_path,
            input_all_sequences_fasta_path,
            input_ground_truth_path,
        ]
    )

    sparse_dict = {}
    seq_id_idx = 0
    profile_id_idx = 2
    score_idx = 8
    seen_sequences = set()
    seen_profiles = set()
    logger.info("Loading ground truth dicts")
    sampled_sequences = set(FastaParser(input_all_sequences_fasta_path).get_ids())
    with open(input_split_subcluster_md_path, "rb") as f:
        split_subcluster_md = pickle.load(f)
    with open(input_ground_truth_path, "rb") as f:
        ground_truth = pickle.load(f)

    logger.info("Loading split hmmsearch results")
    with open(input_split_hmmsearch_results_path, "r") as h:
        lines = h.readlines()
    for line in tqdm(lines[1:], desc="split hmmsearch results"):
        line = line.split("\t")
        seq_id = line[seq_id_idx]
        if seq_id not in sampled_sequences:
            continue
        seen_sequences.add(seq_id)
        profile_id = line[profile_id_idx]
        profile_id = profile_id.removesuffix(".fasta")

        score = line[score_idx]
        subcluster_split = int(profile_id[-1])
        full_subcluster = profile_id[:-2]
        ortholog = re.sub(SUBCLUSTER_PROFILE_SUFFIX_PATTERN, "", full_subcluster)
        if (
            subcluster_split == int(split_subcluster_md[ortholog][seq_id]["test_split"])
            and full_subcluster == split_subcluster_md[ortholog][seq_id]["subcluster"]
        ):
            sparse_dict[(seq_id, full_subcluster)] = int(float(score))

        else:
            continue
            raise ValueError(
                f"Sequence {seq_id} was scored on the wrong profile {profile_id}"
            )

    logger.info("Loading full hmmsearch results")
    with open(input_full_hmmsearch_results_path, "r") as h:
        lines = h.readlines()
    for line in tqdm(lines[1:], desc="full hmmsearch results"):
        line = line.split("\t")
        profile_id = line[profile_id_idx]
        profile_id = profile_id.removesuffix(".fasta")
        ortholog = re.sub(SUBCLUSTER_PROFILE_SUFFIX_PATTERN, "", profile_id)
        seen_profiles.add(profile_id)
        score = line[score_idx]
        seq_id = line[seq_id_idx]

        if seq_id not in ground_truth:
            raise ValueError(
                "Sequence {} not in ground truth, but in hmmsearch results".format(
                    seq_id
                )
            )
        if (
            seq_id in split_subcluster_md[ortholog]
            and profile_id == split_subcluster_md[ortholog][seq_id]["subcluster"]
        ):
            continue
        sparse_dict[(seq_id, profile_id)] = int(float(score))
        seen_sequences.add(seq_id)
    logger.info("Adding missing profiles")
    all_profiles = set(
        [p.removesuffix(".hmm") for p in os.listdir(input_full_profiles_dir_path)]
    )
    missing_profiles = all_profiles - seen_profiles
    for m in missing_profiles:
        sparse_dict[(seq_id, m)] = 0
    logger.info("Adding missing sequences")
    missing_sequences = sampled_sequences - seen_sequences
    for m in missing_sequences:
        sparse_dict[(m, profile_id)] = 0
    logger.info("Generating sparse dataframe")
    ground_truth = {
        k: v.split(";") for k, v in ground_truth.items() if k in sampled_sequences
    }
    sdf = from_sparse_dict(sparse_dict, ground_truth, dtype=np.int64)
    logger.info("Saving sparse dataframe")
    with open(output_sdf_path, "wb+") as f:
        pickle.dump(sdf, f)


def hmmsearch_results_to_classification_sdf(
    input_hmmsearch_results_path: str,
    input_train_sdf_path: str,
    input_fasta_path: str,
    output_path: str,
) -> None:
    """
    Creates a sparse dataframe from the hmmsearch results.
    Parses the hmmsearch results to a dictionary, where the keys are tuples of (sequence ID, profile ID) and the values are the scores.
    Populates the dictionary with zeros for sequence x profile pairs that are missing from the hmmsearch results.
    Creates a sparse dataframe from the dictionary.
    :param input_hmmsearch_results_path: path to hmmsearch results
    :param input_train_sdf_path: path to train sparse dataframe
    :param input_fasta_path: path to fasta file containing all sequences
    :param output_path: path to output sparse dataframe pickle file
    :return: None
    """
    sparse_dict = {}
    seq_id_idx = 0
    profile_id_idx = 2
    score_idx = 8
    seen_ids = set()
    seen_profiles = set()
    logger.info("loading hmmsearch results")
    with open(input_hmmsearch_results_path, "r") as h:
        lines = h.readlines()
    for line in lines[1:]:
        line = line.split("\t")
        profile_id = line[profile_id_idx]
        profile_id = profile_id.removesuffix(".fasta")
        seen_profiles.add(profile_id)
        score = line[score_idx]
        seq_id = line[seq_id_idx]
        seen_ids.add(seq_id)
        sparse_dict[(seq_id, profile_id)] = int(float(score))
    sdf_train = pickle.load(open(input_train_sdf_path, "rb"))
    all_profiles = sdf_train.column_names
    missing_profiles = set(all_profiles) - seen_profiles
    with open(input_fasta_path) as h:
        all_ids = [r.id for r in SeqIO.parse(h, "fasta")]
    some_id = all_ids[0]
    for m in list(missing_profiles):
        sparse_dict[(some_id, m)] = 0
    missing_records = set(all_ids) - seen_ids
    some_profile = sdf_train.column_names[0]
    for m in missing_records:
        sparse_dict[(m, some_profile)] = 0

    sdf = from_sparse_dict(sparse_dict, lables=None, dtype=np.int64)
    colnames = list(sdf.column_names)
    new_order = [colnames.index(c) for c in sdf_train.column_names]
    sdf.matrix = sdf.matrix[:, new_order]
    sdf.column_names = sdf_train.column_names

    logger.info("saving sdf")
    with open(output_path, "wb+") as f:
        pickle.dump(sdf, f)


def sample_subclusters_for_model(
    input_subcluster_fastas_dir_path: str,
    output_sampled_subclusters_fasta_path: str,
    number_of_sampled_sequences_per_subcluster: int,
) -> None:
    """
    Samples sequences from subclusters and saves them to a single fasta file.
    :param input_subcluster_fastas_dir_path: path to directory containing subcluster fasta files
    :param output_sampled_subclusters_fasta_path: path to output fasta file
    :param max_sequences_per_subcluster: maximum number of sequences to sample from each subcluster. If the subcluster size is less or equal to this number, all sequences are sampled. Otherwise, this number of sequences is sampled.
    :return: None
    """
    fasta_paths = get_subcluster_fasta_paths(input_subcluster_fastas_dir_path)
    logger.info(f"found {len(fasta_paths)} subclusters")
    logger.info(
        f"sampling up to {number_of_sampled_sequences_per_subcluster} sequences from each subcluster"
    )
    sampled_records = []
    for fasta_path in tqdm(fasta_paths, desc="sampling subclusters"):
        with open(fasta_path, "r") as f:
            curr_records = [record for record in SeqIO.parse(f, "fasta")]
        n_seqs_to_sample = min(
            number_of_sampled_sequences_per_subcluster, len(curr_records)
        )
        sampled_records.extend(random.sample(curr_records, n_seqs_to_sample))
    logger.info(f"sampled {len(sampled_records)} sequences from subclusters")
    with open(output_sampled_subclusters_fasta_path, "w+") as f:
        SeqIO.write(sampled_records, f, "fasta")


def count_sequences_in_fasta(fasta_path: str) -> int:
    """
    Counts the number of sequences in a fasta file.
    :param fasta_path: path to fasta file
    :return: number of sequences in the fasta file
    """
    return int(subprocess.check_output("grep -c '>' " + fasta_path, shell=True))


def get_fasta_ids(fasta_path: str) -> list:
    ids = subprocess.check_output(
        f"seqkit seq -in {fasta_path}".split(" "),
        stderr=subprocess.DEVNULL,
    ).decode(sys.stdout.encoding)
    ids = ids.split("\n")
    if ids[-1] == "":
        ids = ids[:-1]
    return ids


def extract_sequences_from_fasta(
    fasta_file: str, input_id_list_file: str, output_path: str
) -> None:
    """
    Extracts sequences from a fasta file and saves them to a new fasta file.
    :param fasta_path: path to input fasta file
    :param ids: list of sequence IDs to extract
    :param output_path: path to output fasta file
    :return: None
    """
    with open(output_path, "w+") as f:
        subprocess.call(
            f"seqkit grep -f {input_id_list_file} -o {output_path} {fasta_file}".split(
                " "
            ),
            stdout=f,
            stderr=subprocess.DEVNULL,
        )


def sample_unknown_sequences_for_model(
    input_unknown_sequences_fasta_path: str,
    n_sequences: int,
    tmp_dir_path: str,
    output_sampled_unknown_sequences_fasta_path: str,
) -> None:
    """
    Samples sequences that are not in subclusters and saves them to a single fasta file.
    :param input_unknown_sequences_fasta_path: path to fasta file containing unknown sequences
    :param n_sequences: number of sequences to sample
    :param output_sampled_unknown_sequences_fasta_path: path to output fasta file
    :return: None
    """
    unknown_ids = get_fasta_ids(input_unknown_sequences_fasta_path)
    unique_ids = list(set(unknown_ids))
    sampled_ids = random.sample(unique_ids, n_sequences)
    with open(f"{tmp_dir_path}/sampled_unknown_ids.txt", "w+") as f:
        f.write("\n".join(sampled_ids))
    extract_sequences_from_fasta(
        input_unknown_sequences_fasta_path,
        f"{tmp_dir_path}/sampled_unknown_ids.txt",
        output_sampled_unknown_sequences_fasta_path,
    )
    os.remove(f"{tmp_dir_path}/sampled_unknown_ids.txt")


def emit_sequences(profile_path: str, output_path: str, n_sequences: int) -> None:
    """
    Generates sequences from a profile and saves them to a fasta file. Used to augment small subclusters.
    :param profile_path: path to profile
    :param output_path: path to output fasta file
    :param n_sequences: number of sequences to generate
    :return: None
    """
    with open(output_path, "w+") as handle:
        handle.write(
            subprocess.check_output(
                "hmmemit -N {} {}".format(n_sequences, profile_path).split(" ")
            ).decode(sys.stdout.encoding)
        )


def qmafft(input_path: str, output_path: str) -> None:
    """
    Uses mafft to align sequences and saves the alignment to a file.
    :param input_path: path to input fasta file
    :param output_path: path to output alignment file
    :return: None
    """
    output = subprocess.check_output(
        "app/qmafft {} 1 --quiet".format(input_path).split(" ")
    ).decode(sys.stdout.encoding)
    if output or not os.path.exists(output_path):
        with open(output_path, "w+") as handle:
            handle.write(output)
    else:
        raise ValueError(f"{input_path} failed to align with mafft")


def hmmbuild(input_path: str, output_path: str, name: str) -> None:
    """
    Uses hmmbuild to build a profile from an alignment and saves the profile to a file.
    :param input_path: path to input alignment file
    :param output_path: path to output profile file
    :param name: name of profile
    :return: None
    """
    cmd = "hmmbuild -n {} --amino {} {}".format(name, output_path, input_path)
    result = subprocess.run(
        cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise ValueError(
            f"hmmbuild failed with return code {result.returncode} and error message {result.stderr.decode(sys.stdout.encoding)}"
        )


def copy_subcluster_fastas_to_dir(subcluster_fastas_path: str, output_dir: str) -> None:
    """
    Copies subcluster fastas to a directory.
    :param subcluster_fastas_path: path to directory containing subcluster fastas
    :param output_dir: path to output directory
    :return: None
    """
    for path in tqdm(
        get_subcluster_fasta_paths(subcluster_fastas_path),
        desc="copying subcluster fastas",
    ):
        shutil.copy(path, output_dir)


def fasta_is_small(path: str, n: int) -> int | bool:
    """
    Returns the number of sequences in the file if the fasta file contains less than n sequences and False otherwise.
    :param path: path to fasta file
    :param n: number of sequences below which the fasta file is considered small
    :return: The number of sequences in the file if the fasta file contains less than n sequences, False otherwise
    """
    c = 0
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith(">"):
                c += 1
                if c >= n:
                    return False
    return c


def augment_single_subcluster(
    subcluster_path: str,
    output_dir: str,
    tmp_dir_path: str,
    min_sequences_per_subcluster: int,
) -> None:
    """
    Augments a single subcluster fasta file.
    Creates alignment and profile, them emits sequences from the profile and concatenates them with the original subcluster fasta file.
    :param subcluster_path: path to subcluster fasta file
    :param output_dir: path to output directory
    :param sequences_to_emit: number of sequences to emit
    :return: None
    """
    subcluster_name = os.path.basename(subcluster_path).removesuffix(".fasta")

    if os.path.exists(os.path.join(output_dir, subcluster_name + ".fasta")):
        return

    n_sequences_in_fasta = fasta_is_small(subcluster_path, min_sequences_per_subcluster)
    if not n_sequences_in_fasta:
        shutil.copy(subcluster_path, output_dir)
        return

    qmafft(
        input_path=subcluster_path,
        output_path=f"{tmp_dir_path}{subcluster_name}.aln",
    )
    assert os.path.exists(f"{tmp_dir_path}{subcluster_name}.aln"), (
        f"{tmp_dir_path}{subcluster_name}.aln" + " does not exist"
    )
    hmmbuild(
        input_path=f"{tmp_dir_path}{subcluster_name}.aln",
        output_path=f"{tmp_dir_path}{subcluster_name}.hmm",
        name=subcluster_name,
    )
    assert os.path.exists(f"{tmp_dir_path}{subcluster_name}.hmm"), (
        f"{tmp_dir_path}{subcluster_name}.hmm" + " does not exist"
    )
    sequences_to_emit = min_sequences_per_subcluster - n_sequences_in_fasta
    emit_sequences(
        profile_path=f"{tmp_dir_path}{subcluster_name}.hmm",
        output_path=f"{tmp_dir_path}{subcluster_name}_emission.fasta",
        n_sequences=sequences_to_emit,
    )
    assert os.path.exists(f"{tmp_dir_path}{subcluster_name}_emission.fasta"), (
        f"{tmp_dir_path}{subcluster_name}_emission.fasta" + " does not exist"
    )
    concatenate_files(
        files=[subcluster_path, f"{tmp_dir_path}{subcluster_name}_emission.fasta"],
        output=output_dir + subcluster_name + ".fasta",
        track_progress=False,
    )
    assert os.path.exists(output_dir + subcluster_name + ".fasta"), (
        output_dir + subcluster_name + ".fasta" + " does not exist"
    )

    # delete tmp files
    os.remove(f"{tmp_dir_path}{subcluster_name}.aln")
    os.remove(f"{tmp_dir_path}{subcluster_name}.hmm")
    os.remove(f"{tmp_dir_path}{subcluster_name}_emission.fasta")


def augment_small_subclusters(
    subclusters_dir: str,
    output_dir: str,
    tmp_dir: str,
    min_sequences_per_subcluster: int = 6,
    nthreads: int = nthreads,
) -> None:
    """
    Augments small subclusters by creating alignments and profiles, then emitting sequences from the profiles and concatenating them with the original subcluster fasta files.
    :param subclusters_dir: path to directory containing subcluster fasta files
    :param output_dir: path to output directory
    :param min_sequences_per_subcluster: minimum number of sequences to sample from each subcluster. If the subcluster size is less or equal to this number, all sequences are sampled. Otherwise, this number of sequences is sampled.
    :return: total number of sequences added
    """
    subclusters_dir, output_dir = create_paths_if_not_exists(
        [subclusters_dir, output_dir]
    )

    subcluster_files_paths = get_subcluster_fasta_paths(subclusters_dir)
    args = []
    for subcluster_path in subcluster_files_paths:
        args.append(
            (
                subcluster_path,
                output_dir,
                tmp_dir,
                min_sequences_per_subcluster,
            )
        )
    pool = mp.Pool(processes=nthreads)
    pool.starmap(augment_single_subcluster, args)
    pool.close()
    pool.join()


def randomly_split_subclusters(
    input_subclusters_fasta_dir_path: str,
    output_fasta_dir_for_profiles_path: str,
    output_fasta_dir_for_scoring_path: str,
    output_path_md: str,
):
    """
    Randomly splits subclusters into 3 groups of roughly equal size.
    Each group is two thirds of the subcluster.
    Each group is used to get a score for the remaining third, so that each sequence is scored on a profile it is not in.
    Saves which sequences should be scored on which profiles in a pickle file (the ID of the profile it is not in).
    :param input_subclusters_fasta_dir_path: path to directory containing subcluster fasta files
    :param output_fasta_dir_for_profiles_path: path to output directory for split profiles
    :param output_fasta_dir_for_scoring_path: path to output directory for sequences to be scored
    :param output_path_md: path to output pickle file containing metadata
    :return: None
    """
    (input_subclusters_fasta_dir_path,) = validate_dir_paths(
        [input_subclusters_fasta_dir_path]
    )
    (
        output_fasta_dir_for_profiles_path,
        output_fasta_dir_for_scoring_path,
    ) = create_paths_if_not_exists(
        [output_fasta_dir_for_profiles_path, output_fasta_dir_for_scoring_path]
    )
    split_subcluster_md = {}
    random.seed(42)
    for file in tqdm(
        os.listdir(input_subclusters_fasta_dir_path),
        desc="subclusters",
        total=len(os.listdir(input_subclusters_fasta_dir_path)),
    ):
        assert ".sub_cluster.cluster." in file and file.endswith(
            ".fasta"
        ), "file {} is not a subcluster fasta".format(file)
        subcluster = file.removesuffix(".fasta")
        ortholog = re.sub(SUBCLUSTER_PROFILE_SUFFIX_PATTERN, "", subcluster)
        if ortholog not in split_subcluster_md:
            split_subcluster_md[ortholog] = {}
        file_path = os.path.join(input_subclusters_fasta_dir_path, file)
        parser = FastaParser(file_path)
        number_of_sequences = len(parser)
        assert (
            number_of_sequences >= 6
        ), "subcluster {} has less than 6 sequences".format(file)
        seq_ids = parser.get_ids()
        random.shuffle(seq_ids)
        groups = even_split(seq_ids, 3)

        for i in range(3):
            profile_split_file = subcluster + "." + str(i) + ".fasta"
            profile_split_file = output_fasta_dir_for_profiles_path + profile_split_file
            record_split_file = subcluster + "." + str(i) + ".fasta"
            record_split_file = output_fasta_dir_for_scoring_path + record_split_file
            with open(
                record_split_file, "w+"
            ) as f:  # exporting the sequences that will be scored against the profile
                parser.export_sequences(groups[i], f)
            profile_records = []  # the records of the sequences that will make the profile
            for j in range(3):
                if (
                    j != i
                ):  # the group where i == j is the group that will be scored on this profile and is not used to make it
                    profile_records += groups[j]
            for seq_id in groups[
                i
            ]:  # saves the metadata for the sequences that should be scored on the current profile
                split_subcluster_md[ortholog][seq_id] = {
                    "test_split": i,
                    "subcluster": subcluster,
                }
            with open(profile_split_file, "w+") as f:
                parser.export_sequences(profile_records, f)

    with open(output_path_md, "wb+") as f:
        pickle.dump(split_subcluster_md, f)


def _create_subcluster_profile(
    path: str, full_profile_dir: str, tmp_dir_path: str
) -> None:
    """
    Creates a profile from a subcluster fasta file.
    :param path: path to subcluster fasta file
    :param full_profile_dir: path to output directory for profiles
    :return: None
    """
    subcluster_name = os.path.basename(path).removesuffix(".fasta")
    output_hmm_path = os.path.join(full_profile_dir, subcluster_name + ".hmm")
    qmafft(path, f"{tmp_dir_path}{subcluster_name}.full.aln")
    hmmbuild(
        f"{tmp_dir_path}{subcluster_name}.full.aln",
        output_hmm_path,
        subcluster_name,
    )
    os.remove(f"{tmp_dir_path}{subcluster_name}.full.aln")


def create_subcluster_profiles(
    subcluster_dir: str, profile_dir: str, tmp_dir_path: str, nthreads=nthreads
) -> None:
    """
    Creates profiles from subcluster fasta files.
    Will skip fasta files for which a profile already exists.
    :param subcluster_dir: path to directory containing subcluster fasta files
    :param profile_dir: path to output directory for profiles
    :param nthreads: number of threads to use
    :return: None
    """

    subcluster_paths = [
        os.path.join(subcluster_dir, f) for f in os.listdir(subcluster_dir)
    ]
    if not os.path.exists(profile_dir):
        os.mkdir(profile_dir)

    # cleaning empty profiles from possible previous runs
    empty_profile_file_paths = [
        profile_dir + f
        for f in os.listdir(profile_dir)
        if f.endswith(".hmm") and os.path.getsize(profile_dir + f) == 0
    ]
    for path in empty_profile_file_paths:
        os.remove(path)

    logger.info("checking for existing profiles")
    existing_profiles = set([f.removesuffix(".hmm") for f in os.listdir(profile_dir)])
    if len(existing_profiles) == len(subcluster_paths):
        logger.info("all profiles already exist")
        return
    if len(existing_profiles) > 0:
        logger.info(str(len(existing_profiles)) + " profiles already exist")
    subcluster_paths_without_profiles = [
        path
        for path in subcluster_paths
        if os.path.basename(path).removesuffix(".fasta") not in existing_profiles
    ]
    logger.info("creating {} profiles".format(len(subcluster_paths_without_profiles)))
    pool = mp.Pool(processes=nthreads)
    pool.starmap(
        _create_subcluster_profile,
        iterable=[
            (path, profile_dir, tmp_dir_path)
            for path in subcluster_paths_without_profiles
        ],
    )
    pool.close()
    pool.join()


def compile_subclusters_profile_db(
    subcluster_dir: str, full_profile_dir: str, output_path: str, nthreads=nthreads
) -> None:
    """
    Creates a profile database from subcluster fasta files.
    :param subcluster_dir: path to directory containing subcluster fasta files
    :param full_profile_dir: path to output directory for profiles
    :param output_path: path to output profile database
    :param nthreads: number of threads to use
    :return: None
    """
    create_subcluster_profiles(subcluster_dir, full_profile_dir, nthreads=nthreads)
    concatenate_files(
        [
            full_profile_dir + f
            for f in os.listdir(full_profile_dir)
            if f.endswith(".hmm")
        ],
        output_path,
        track_progress=False,
    )


def _hmmsearch_single_profile(
    fasta_file_path: str,
    profiles_db_path: str,
    output_path: str,
    threads: int = 4,
    no_filter: bool = False,
) -> None:
    try:
        dom_t_option = ""
        if no_filter:
            dom_t_option = " -T -10000 --max"
        # run hmmsearch
        cmd = "hmmsearch -o /dev/null --tblout {} --cpu {}{} {} {}".format(
            output_path,
            str(threads),
            dom_t_option,
            fasta_file_path,
            profiles_db_path,
        )
        subprocess.check_output(cmd.split(" ")).decode(sys.stdout.encoding)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(
                f"hmmsearch failed to create results file {output_path}. command: {cmd}"
            )
    except Exception as e:
        logger.error("hmmsearch failed")
        raise e


def tail(path, n=1):
    """
    Returns the last n lines of a file.
    :param path: path to file
    :param n: number of lines to return
    :return: last n lines of file
    """
    return subprocess.check_output("tail -n {} {}".format(n, path).split(" ")).decode(
        sys.stdout.encoding
    )


def full_hmmsearch(
    input_full_profiles_dir_path: str,
    input_fasta_path: str,
    output_full_hmmsearch_results_path: str,
    tmp_dir_path: str,
    n_processes: int,
    no_filter: bool = False,
) -> None:
    """
    Runs hmmsearch on a fasta file using a profile database.
    :param input_full_profiles_dir_path: path to profiles directory
    :param input_fasta_path: path to fasta file
    :param output_full_hmmsearch_results_path: path to output hmmsearch results
    :param n_processes: number of hmmsearch processes to run in parallel
    :return: None
    """
    if not os.path.exists(input_fasta_path):
        raise ValueError("input_fasta_path does not exist")

    if not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer")

    (input_full_profiles_dir_path,) = validate_dir_paths([input_full_profiles_dir_path])

    # split fasta_files
    if not tmp_dir_path or not os.path.exists(tmp_dir_path):
        tmp_dir_path = "tmp/{}".format(now()) + "/"
        os.mkdir(tmp_dir_path)
    profile_paths = [
        input_full_profiles_dir_path + f
        for f in os.listdir(input_full_profiles_dir_path)
    ]
    args = []
    output_file_paths = []
    for profile_path in profile_paths:
        search_output_path = tmp_dir_path + "{}.hmmsearch".format(
            profile_path.split("/")[-1].removesuffix(".hmm")
        )
        output_file_paths.append(search_output_path)
        if not (
            os.path.exists(search_output_path)
            # and tail(search_output_path) == "# [ok]\n"
        ):  # check if hmmsearch has not already been run successfully
            args.append(
                (
                    profile_path,
                    input_fasta_path,
                    search_output_path,
                    1,
                    no_filter,
                )
            )
    if len(args) == 0:
        logger.info("all hmmsearch results already exist")
    else:
        pool = mp.Pool(processes=n_processes)
        logger.info(f"running hmmsearch for {len(args)} profiles")
        pool.starmap(_hmmsearch_single_profile, iterable=args)
        pool.close()
        pool.join()
        logger.info("hmmsearch done")
    tmp_output_path = os.path.join(tmp_dir_path, "full_hmmsearch.txt")
    if os.path.exists(tmp_output_path) and "# [done]" in tail(tmp_output_path):
        logger.info("full hmmsearch results already exist")
    else:
        logger.info("concatenating hmmsearch results")

        logger.info("concatenating {} files".format(len(output_file_paths)))
        concatenate_files(output_file_paths, tmp_output_path)
        with open(tmp_output_path, "a") as f:
            f.write("# [done]")

    try:
        logger.info("postprocessing hmmsearch results")
        with open(tmp_output_path, "r") as input_handle, open(
            output_full_hmmsearch_results_path, "w+"
        ) as output_handle:
            while line := input_handle.readline():
                if not line.startswith("#"):
                    line = line.strip("\n").split(" ")
                    line = [token for token in line if token]
                    non_desc_tokens = line[:22]
                    desc_tokens = line[22:]
                    description = " ".join(desc_tokens)
                    non_desc_tokens.append(description)
                    output_handle.write("\t".join(non_desc_tokens) + "\n")
    except Exception as e:
        logger.error("hmmsearch postprocessing failed")
        raise e
    # for file in output_file_paths:
    #     os.remove(file)
    # os.remove(tmp_output_path)


def single_split_search(
    profile_path: str, file_for_scoring: str, output_path: str
) -> None:
    """
    Runs hmmsearch on a fasta file using a profile.
    :param profile_path: path to profile
    :param file_for_scoring: path to fasta file
    :param output_path: path to output hmmsearch results
    :return: None
    """
    # create fasta file with relevant sequences
    subcluster_split_name = os.path.basename(file_for_scoring).removesuffix(".fasta")
    tsv_output_path = os.path.join(
        output_path, subcluster_split_name + ".hmmsearch.tsv"
    )
    if os.path.exists(tsv_output_path):
        logger.debug("skipping {}".format(subcluster_split_name))
        return
    logger.debug("starting {}".format(subcluster_split_name))
    # run hmmsearch
    hmmsearch_output_path = output_path + subcluster_split_name + ".hmmsearch"
    cmd = "hmmsearch -o /dev/null --tblout {}".format(hmmsearch_output_path)
    cmd += " {} {}".format(profile_path, file_for_scoring)
    logger.debug("Running CMD: " + cmd)
    cmd = cmd.split(" ")
    cmd = [s for s in cmd if s]
    completed_process = subprocess.run(cmd, stdout=subprocess.PIPE)
    if completed_process.returncode != 0:
        raise ValueError(
            f"hmmsearch failed with return code {completed_process.returncode}. command: {cmd}"
        )
    if (
        not os.path.exists(hmmsearch_output_path)
        or os.path.getsize(hmmsearch_output_path) == 0
    ):
        raise ValueError(
            f"hmmsearch failed to create results file {hmmsearch_output_path}. command: {cmd}"
        )

    logger.info("{} hmmsearch done, postprocessing".format(subcluster_split_name))
    with open(hmmsearch_output_path, "r") as input_handle, open(
        tsv_output_path, "w+"
    ) as output_handle:
        for line in input_handle.readlines():
            if line and not line.startswith("#"):
                line = line.strip("\n").split(" ")
                line = [token for token in line if token]
                non_desc_tokens = line[:22]
                desc_tokens = line[22:]
                description = " ".join(desc_tokens)
                non_desc_tokens.append(description)
                output_handle.write("\t".join(non_desc_tokens) + "\n")
    logger.info("Done {}".format(subcluster_split_name))


def split_profiles_hmmsearch(
    input_split_profiles_dir_path: str,
    input_fasta_dir_for_scoring_path: str,
    output_split_hmmsearch_results_path: str,
    tmp_path="",
    nthreads=nthreads,
) -> None:
    """
    Runs hmmsearch on a split fasta file using a split profile.
    The profile is roughly 2/3 of the subcluster, and the fasta file is the remaining 1/3.
    The output is used to get a score for the remaining 1/3, so that each sequence is scored on a profile it is not in.
    :param input_split_profiles_dir_path: path to directory containing split profiles
    :param input_fasta_dir_for_scoring_path: path to directory containing fasta files to be scored
    :param input_split_sequence_md_path: path to pickle file containing metadata
    :param output_split_hmmsearch_results_path: path to output hmmsearch results
    :param tmp_path: path to directory where results are saved temporarily - if this method stops before finishing, use the directory from the previous run to continue
    :param nthreads: number of threads to use
    :return: None
    """
    (
        input_split_profiles_dir_path,
        input_fasta_dir_for_scoring_path,
    ) = validate_dir_paths(
        [input_split_profiles_dir_path, input_fasta_dir_for_scoring_path]
    )

    if tmp_path == "":
        tmp_path = "tmp/" + str(uuid.uuid4()) + "/"
        os.mkdir(tmp_path)

    logger.info("saving data to " + tmp_path)

    pool = mp.Pool(processes=nthreads)

    split_fastas_files_paths = [
        input_fasta_dir_for_scoring_path + f
        for f in os.listdir(input_fasta_dir_for_scoring_path)
    ]
    logger.info("generating args")
    existing_results = set(os.listdir(tmp_path))
    args = [
        SplitHMMSearchArgs(
            profile_path=input_split_profiles_dir_path
            + os.path.basename(path).removesuffix(".fasta")
            + ".hmm",
            file_for_scoring=path,
            output_path=tmp_path,
        )
        for path in split_fastas_files_paths
        if os.path.basename(path).removesuffix(".fasta") + ".hmmsearch.tsv"
        not in existing_results
    ]
    logger.info("running hmmsearch on {} files".format(len(args)))
    pool.starmap(single_split_search, args)
    pool.close()
    pool.join()

    # concatenate results
    logger.info("concatenating results")
    hmmsearch_results = [
        tmp_path + f for f in os.listdir(tmp_path) if f.endswith(".tsv")
    ]
    concatenate_files(
        hmmsearch_results, output_split_hmmsearch_results_path, track_progress=False
    )


def _convert_fasta_headers(
    input_fasta_path: str, output_fasta_path: str, output_mapping_file: str
) -> None:
    """
    Writes a fasta file with new headers and a mapping file from the old headers to the new ones.
    Used because mmseqs has ambiguity in how it handles fasta headers so this standardizes them.
    After clustering, the mapping file is used to map the substitute headers back to the original
    with _convert_fasta_headers_back.
    :param input_fasta_path: path to input fasta file
    :param output_fasta_path: path to output fasta file
    :param output_mapping_file: path to output mapping file
    :return: None
    """
    c = 0
    with open(input_fasta_path, "r") as input_handle, open(
        output_fasta_path, "w+"
    ) as output_handle, open(output_mapping_file, "w+") as mapping_handle:
        for line in input_handle.readlines():
            if line.startswith(">"):
                c += 1
                ind = str(c)
                mapping_handle.write("{}{}".format(ind, line))
                output_handle.write(">{}\n".format(ind))
            else:
                output_handle.write(line)


def _convert_fasta_headers_back(
    input_fasta_path: str, mapping_file: str, output_fasta_path: str
) -> None:
    """
    Writes a fasta file with old headers using a mapping file from the old headers to the new ones.
    Used because mmseqs has ambiguity in how it handles fasta headers so this maps them back to the original after changing them using _convert_fasta_headers.
    :param input_fasta_path: path to input fasta file
    :param mapping_file: path to mapping file
    :param output_fasta_path: path to output fasta file
    :return: None
    """
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            mapping[line[: line.index(">")]] = line[line.index(">") + 1 :]
    with open(input_fasta_path, "r") as input_handle, open(
        output_fasta_path, "w+"
    ) as output_handle:
        for line in input_handle.readlines():
            if line.startswith(">"):
                ind = line.strip(" \n").removeprefix(">")
                output_handle.write(">{}\n".format(mapping[ind]))
            else:
                output_handle.write(line)


def _subclusters_fully_contain_rep_seqs(
    subcluster_files_paths: list[str], rep_seq_file_path: str
) -> bool:
    subcluster_records = []
    for path in subcluster_files_paths:
        with open(path, "r") as f:
            records = [record.id for record in SeqIO.parse(f, "fasta")]
            subcluster_records.extend(records)
    rep_seq_records = [r.id for r in SeqIO.parse(rep_seq_file_path, "fasta")]
    return set(rep_seq_records) == set(subcluster_records)


def has_duplicate_ids(fasta_path: str) -> bool:
    """
    Checks if a fasta file has duplicate sequence IDs.
    :param fasta_path: path to fasta file
    :return: True if there are duplicate sequence IDs, False otherwise
    """
    ids = set()
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id in ids:
            return True
        ids.add(record.id)
    return False


def single_ortholog_to_subclusters(
    file_path: str,
    subclusters_output_path: str,
    leftovers_output_path: str,
    tmp_nr90: str,
    tmp_clu: str,
    tmp_leftovers: str,
    tmp_hmmsearch: str,
    output_rep_seq_dir: str,
) -> None:
    """
    Clusters a single ortholog fasta file into subclusters.
    :param file_path: path to ortholog fasta file
    :param output_path: path to output directory
    :param tmp_nr90: path to temporary directory
    :param tmp_clu: path to temporary directory
    :param output_rep_seq_dir: path to output directory for representative sequences
    :param output_subcluster_dir: path to output directory for subclusters
    :return: None
    """
    try:
        fname = os.path.basename(file_path).removesuffix(".fasta")

        # check if the file has already been processed

        existing_subclusters_in_output = [
            f
            for f in os.listdir(subclusters_output_path)
            if re.sub(SUBCLUSTER_SUFFIX_PATTERN, "", f) == fname
        ]
        existing_subclusters_in_output = [
            os.path.join(subclusters_output_path, f)
            for f in existing_subclusters_in_output
        ]
        if len(existing_subclusters_in_output) > 0:
            if _subclusters_fully_contain_rep_seqs(
                existing_subclusters_in_output, file_path
            ):
                # All sequences are in subclusters - therefore the file has already been processed
                logger.debug(f"{fname} has already been processed, skipping")
                return
            for path in existing_subclusters_in_output:
                os.remove(path)

        # process the file
        # assure no duplicates
        if has_duplicate_ids(file_path):
            raise ValueError(f"{file_path} contains duplicate sequence IDs")
        # rename the fasta headers to numbers so that mmseqs handles them uniformly
        _convert_fasta_headers(
            file_path,
            f"{tmp_nr90}{fname}.fasta",
            f"{tmp_nr90}{fname}.mapping",
        )

        # initial clustering to remove redundancy
        cmd = f"mmseqs easy-cluster {tmp_nr90}{fname}.fasta {tmp_nr90}{fname} {tmp_nr90}{fname}_tmp --threads {mmseq_nthreads} -s 7.5 -c 0.8 --min-seq-id 0.9"
        result = subprocess.run(
            cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            logger.warning(
                f"warning: mmseqs failed on clustering {fname}, retrying with single step clustering"
            )
            cmd = f"mmseqs easy-cluster {tmp_nr90}{fname}.fasta {tmp_nr90}{fname} {tmp_nr90}{fname}_tmp --threads {mmseq_nthreads} --single-step-clustering -s 7.5 -c 0.8 --min-seq-id 0.9"
            result = subprocess.run(
                cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                raise ValueError(
                    f"cmd failed: {cmd} with return code {result.returncode} and error message {result.stderr.decode(sys.stdout.encoding)}"
                )
            logger.info(f"single-step clustering successful for {fname}")
        if (
            not os.path.exists(f"{tmp_nr90}{fname}_rep_seq.fasta")
            or os.path.getsize(f"{tmp_nr90}{fname}_rep_seq.fasta") == 0
        ):
            raise ValueError(
                f"mmseqs easy-cluster failed to create representative sequences for {fname}"
            )

        shutil.copyfile(
            f"{tmp_nr90}{fname}_rep_seq.fasta",
            f"{output_rep_seq_dir}{fname}.fasta",
        )

        # if not doing subclusters
        if True:
            _convert_fasta_headers_back(
                f"{tmp_nr90}{fname}_rep_seq.fasta",
                f"{tmp_nr90}{fname}.mapping",
                f"{subclusters_output_path}{fname}.sub_cluster.cluster.1.fasta",
            )
            return

        # clustering to get subclusters out of the representative sequences

        cmd = f"mmseqs easy-cluster {tmp_nr90}{fname}_rep_seq.fasta {tmp_clu}{fname} -s 7.5 -c 0.5 {tmp_clu}{fname}_tmp --threads {mmseq_nthreads}"
        result = subprocess.run(
            cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            logger.warning(
                f"warning: subclustering mmseqs failed on clustering {fname}, retrying with single step clustering"
            )
            cmd = f"mmseqs easy-cluster {tmp_nr90}{fname}_rep_seq.fasta {tmp_clu}{fname} --single-step-clustering -s 7.5 -c 0.5 {tmp_clu}{fname}_tmp --threads {mmseq_nthreads}"
            result = subprocess.run(
                cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                raise ValueError(
                    f"cmd failed: {cmd} with return code {result.returncode} and error message {result.stderr.decode(sys.stdout.encoding)}"
                )
            logger.info(f"single-step clustering successful for {fname}")

        if (
            not os.path.exists(f"{tmp_clu}{fname}_cluster.tsv")
            or os.path.getsize(f"{tmp_clu}{fname}_cluster.tsv") == 0
        ):
            raise ValueError(
                f"mmseqs easy-cluster failed to create subclusters for {fname}"
            )

        # get the subclusters
        cluster_seq_ids = {}
        with open(f"{tmp_clu}{fname}_cluster.tsv", "r") as f:
            for line in f.readlines():
                line = line.strip("\n").split("\t")
                cluster, seq_id = line[0], line[1]
                if cluster not in cluster_seq_ids:
                    cluster_seq_ids[cluster] = []
                cluster_seq_ids[cluster].append(seq_id)

        # get the clusters that contain at least x% of the sequences
        total_seqs = sum([len(v) for v in cluster_seq_ids.values()])
        cluster_names_and_sizes = [(k, len(v)) for k, v in cluster_seq_ids.items()]
        cluster_names_and_sizes = sorted(
            cluster_names_and_sizes, key=lambda x: x[1], reverse=True
        )
        coverage = 0.9
        i = 0
        total_seqs_covered = 0
        while total_seqs_covered < total_seqs * coverage:
            total_seqs_covered += cluster_names_and_sizes[i][1]
            i += 1
        index_of_first_leftover_subcluster = i
        logger.debug(f"{fname} has {index_of_first_leftover_subcluster} subclusters")
        # get the records for all sequences
        with open(f"{tmp_nr90}{fname}_rep_seq.fasta", "r") as f:
            records = [record for record in SeqIO.parse(f, "fasta")]
        records = {record.id: record for record in records}

        # write the subclusters to files
        profiles_dir = os.path.join(tmp_hmmsearch, "profiles/")
        alignments_dir = os.path.join(tmp_hmmsearch, "alignments/")
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(alignments_dir, exist_ok=True)
        subcluster_counter = 1
        relevant_profiles = []
        for i, cluster_name_and_size in enumerate(
            cluster_names_and_sizes[:index_of_first_leftover_subcluster]
        ):
            cluster_name, cluster_size = cluster_name_and_size
            seq_ids = cluster_seq_ids[cluster_name]

            with open(
                f"{tmp_clu}{fname}.sub_cluster.cluster.{subcluster_counter}.fasta", "w+"
            ) as subcluster_f:
                SeqIO.write(
                    [records[seq_id] for seq_id in seq_ids], subcluster_f, "fasta"
                )
            _convert_fasta_headers_back(
                f"{tmp_clu}{fname}.sub_cluster.cluster.{subcluster_counter}.fasta",
                f"{tmp_nr90}{fname}.mapping",
                f"{subclusters_output_path}{fname}.sub_cluster.cluster.{subcluster_counter}.fasta",
            )
            os.remove(
                f"{tmp_clu}{fname}.sub_cluster.cluster.{subcluster_counter}.fasta"
            )

            # make profiles
            logger.debug(f"doing {fname} qmafft")
            qmafft(
                input_path=f"{subclusters_output_path}{fname}.sub_cluster.cluster.{subcluster_counter}.fasta",
                output_path=f"{alignments_dir}{fname}.sub_cluster.cluster.{subcluster_counter}.aln",
            )
            logger.debug(f"doing {fname} hmmbuild")
            hmmbuild(
                input_path=f"{alignments_dir}{fname}.sub_cluster.cluster.{subcluster_counter}.aln",
                output_path=f"{profiles_dir}{fname}.sub_cluster.cluster.{subcluster_counter}.hmm",
                name=f"{fname}.sub_cluster.cluster.{subcluster_counter}.fasta",
            )
            relevant_profiles.append(
                f"{profiles_dir}{fname}.sub_cluster.cluster.{subcluster_counter}.hmm"
            )
            subcluster_counter += 1

        concatenate_files(
            relevant_profiles,
            f"{profiles_dir}{fname}.hmm.db",
            track_progress=False,
        )

        leftover_file_path = f"{leftovers_output_path}{fname}.leftovers.fasta"
        leftover_records = []
        for cluster_name_and_size in cluster_names_and_sizes[
            index_of_first_leftover_subcluster:
        ]:
            cluster_name, cluster_size = cluster_name_and_size
            seq_ids = cluster_seq_ids[cluster_name]
            leftover_records.extend([records[seq_id] for seq_id in seq_ids])
        shutil.copyfile(
            f"{output_rep_seq_dir}{fname}.fasta",
            f"{output_rep_seq_dir}{fname}.fasta.tmp",
        )
        _convert_fasta_headers_back(
            f"{output_rep_seq_dir}{fname}.fasta.tmp",
            f"{tmp_nr90}{fname}.mapping",
            f"{output_rep_seq_dir}{fname}.fasta",
        )
        os.remove(
            f"{output_rep_seq_dir}{fname}.fasta.tmp",
        )
        if len(leftover_records) > 0:
            logger.debug(f"{fname} has {len(leftover_records)} leftovers")
            with open(
                f"{tmp_leftovers}{fname}.leftovers.fasta.tmp", "w+"
            ) as leftover_f:
                SeqIO.write(leftover_records, leftover_f, "fasta")
            _convert_fasta_headers_back(
                f"{tmp_leftovers}{fname}.leftovers.fasta.tmp",
                f"{tmp_nr90}{fname}.mapping",
                leftover_file_path,
            )
            leftover_records = [
                record for record in SeqIO.parse(leftover_file_path, "fasta")
            ]
            # assert the leftover_file_path size is not 0
            assert (
                os.path.getsize(leftover_file_path) > 0
            ), f"{leftover_file_path} is empty"
            logger.debug(
                f"{fname} has {index_of_first_leftover_subcluster} subclusters"
            )
            if index_of_first_leftover_subcluster == 1:
                logger.debug(f"{fname} has only one subcluster")
                num_sequences_before = count_sequences_in_fasta(
                    f"{subclusters_output_path}{fname}.sub_cluster.cluster.1.fasta"
                )
                with open(
                    f"{subclusters_output_path}{fname}.sub_cluster.cluster.1.fasta", "a"
                ) as f:
                    SeqIO.write(leftover_records, f, "fasta")
                num_sequences_after = count_sequences_in_fasta(
                    f"{subclusters_output_path}{fname}.sub_cluster.cluster.1.fasta"
                )
                logger.debug(
                    f"{fname} num sequences before: {num_sequences_before}, num sequences after: {num_sequences_after}"
                )
                if not num_sequences_after == num_sequences_before + len(
                    leftover_records
                ):
                    logger.debug(
                        f"{fname} num sequences before: {num_sequences_before}, num sequences after: {num_sequences_after}"
                    )
                    raise ValueError(
                        f"{fname} num sequences before: {num_sequences_before}, num sequences after: {num_sequences_after}"
                    )
            else:
                # hmmsearch leftovers

                cmd = "hmmsearch -o /dev/null --tblout {} --max -E 10000 --cpu {} {} {}".format(
                    f"{tmp_hmmsearch}{fname}.hmmsearch",
                    str(nthreads),
                    f"{profiles_dir}{fname}.hmm.db",
                    leftover_file_path,
                )

                result = subprocess.run(
                    cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if result.returncode != 0:
                    raise ValueError(
                        f"cmd failed: {cmd} with return code {result.returncode} and error message {result.stderr.decode(sys.stdout.encoding)}"
                    )

                # parse the results
                best_hits = {}
                try:
                    with open(f"{tmp_hmmsearch}{fname}.hmmsearch", "r") as input_handle:
                        for line in input_handle.readlines():
                            if line and not line.startswith("#"):
                                line = line.strip("\n").split(" ")
                                line = [token for token in line if token]
                                seq_id = line[0]
                                profile_id = line[2]
                                dom_score = float(line[8])
                                if seq_id not in best_hits:
                                    best_hits[seq_id] = (profile_id, dom_score)
                                else:
                                    if dom_score > best_hits[seq_id][1]:
                                        best_hits[seq_id] = (profile_id, dom_score)
                    logger.debug(fname + " best hits: " + str(best_hits))
                except Exception as e:
                    logger.error(f"{fname} hmmsearch postprocessing failed")
                    raise e

                # append leftovers back into subcluster files
                leftover_record_ids = {record.id: record for record in leftover_records}
                if not set(best_hits.keys()) == set(leftover_record_ids.keys()):
                    missing = set(leftover_record_ids.keys()).difference(
                        set(best_hits.keys())
                    )
                    logger.warning(
                        f"{fname} not all leftovers have a best hit in the subclusters: {str(missing)}. Creating subcluster for them."
                    )
                    relevant_records = [
                        leftover_record_ids[seq_id] for seq_id in missing
                    ]
                    with open(
                        f"{subclusters_output_path}{fname}.sub_cluster.cluster.{subcluster_counter}.fasta",
                        "w+",
                    ) as f:
                        SeqIO.write(relevant_records, f, "fasta")

                for seq_id, (subcluster_name, _) in best_hits.items():
                    with open(f"{subclusters_output_path}{subcluster_name}", "a") as f:
                        SeqIO.write([leftover_record_ids[seq_id]], f, "fasta")

            # assert all leftovers are in subclusters
            leftover_ids = [
                record.id for record in SeqIO.parse(leftover_file_path, "fasta")
            ]
            logger.debug(f"{fname} leftover records: {len(leftover_ids)}")
            subcluster_ids = []
            existing_subclusters_in_output = [
                f
                for f in os.listdir(subclusters_output_path)
                if re.sub(SUBCLUSTER_SUFFIX_PATTERN, "", f) == fname
            ]
            logger.debug(
                f"{fname} existing subclusters: {existing_subclusters_in_output}"
            )
            existing_subclusters_in_output = [
                os.path.join(subclusters_output_path, f)
                for f in existing_subclusters_in_output
            ]
            for path in existing_subclusters_in_output:
                with open(path, "r") as f:
                    record_ids = [record.id for record in SeqIO.parse(f, "fasta")]
                    subcluster_ids.extend(record_ids)
            subcluster_ids = sorted(subcluster_ids)
            logger.debug(f"{fname} subcluster records: {len(subcluster_ids)}")
            leftover_ids = sorted(leftover_ids)
            logger.debug(f"{fname} leftover records: {str(leftover_ids)}")
            logger.debug(f"{fname} subcluster records: {str(subcluster_ids)}")
            logger.debug(
                f"{fname} difference: {str(set(leftover_ids).difference(set(subcluster_ids)))}"
            )

            try:
                if not set(leftover_ids).issubset(set(subcluster_ids)):
                    raise ValueError(f"{fname} leftovers are not in subclusters")
                else:
                    logger.debug(f"success - {fname} leftovers are in subclusters")
            except Exception as e:
                logger.error(f"{fname} ERROR: {str(e)}")
                raise e
        else:
            logger.debug(f"{fname} has no leftovers")
        # clean up

        try:
            shutil.rmtree(f"{tmp_nr90}{fname}_tmp")
            shutil.rmtree(f"{tmp_clu}{fname}_tmp")
        except Exception:
            pass
        os.remove(f"{tmp_nr90}{fname}_cluster.tsv")
        os.remove(f"{tmp_nr90}{fname}_rep_seq.fasta")
        os.remove(f"{tmp_nr90}{fname}_all_seqs.fasta")
        os.remove(f"{tmp_clu}{fname}_cluster.tsv")
        os.remove(f"{tmp_clu}{fname}_rep_seq.fasta")
        os.remove(f"{tmp_clu}{fname}_all_seqs.fasta")
        os.remove(f"{tmp_nr90}{fname}.fasta")
        os.remove(f"{tmp_nr90}{fname}.mapping")
    except Exception as e:
        logger.error(f"{fname} failed - {str(e)}")
        raise e


def orthologs_to_subclusters(
    ortholog_fasta_dir: str,
    output_subcluster_dir: str,
    output_leftovers_dir: str,
    output_rep_seq_dir: str,
    tmp_dir_path: str,
    nthreads=nthreads,
) -> None:
    """
    Clusters ortholog fasta files into subclusters.
    First clusters the orthologs into representative sequences, then clusters the representative sequences into subclusters.
    :param ortholog_fasta_dir: path to directory containing ortholog fasta files
    :param output_subcluster_dir: path to output directory for subclusters
    :param output_rep_seq_dir: path to output directory for representative sequences
    :param nthreads: number of threads to use
    :return: None
    """
    tmp_clu = os.path.join(tmp_dir_path, "CLU/")
    tmp_nr90 = os.path.join(tmp_dir_path, "NR90/")
    tmp_leftovers = os.path.join(tmp_dir_path, "leftovers/")
    tmp_hmmsearch = os.path.join(tmp_dir_path, "hmmsearch_leftovers/")
    (
        tmp_clu,
        tmp_nr90,
        tmp_leftovers,
        output_rep_seq_dir,
        ortholog_fasta_dir,
        output_subcluster_dir,
        output_leftovers_dir,
    ) = create_paths_if_not_exists(
        [
            tmp_clu,
            tmp_nr90,
            tmp_leftovers,
            output_rep_seq_dir,
            ortholog_fasta_dir,
            output_subcluster_dir,
            output_leftovers_dir,
        ]
    )
    input_fasta_file_paths = [
        os.path.join(ortholog_fasta_dir, f)
        for f in os.listdir(ortholog_fasta_dir)
        if f.endswith(".fasta")
    ]
    if len(input_fasta_file_paths) == 0:
        raise ValueError(f"no files with .fasta suffix found in {ortholog_fasta_dir}")
    # assure no duplicates

    pool = mp.Pool(processes=nthreads // mmseq_nthreads)
    for fp in input_fasta_file_paths:
        pool.apply_async(
            single_ortholog_to_subclusters,
            args=(
                fp,
                output_subcluster_dir,
                output_leftovers_dir,
                tmp_nr90,
                tmp_clu,
                tmp_leftovers,
                tmp_hmmsearch,
                output_rep_seq_dir,
            ),
        )
        time.sleep(0.2)  # prevents mmseqs from crashing
    pool.close()
    pool.join()
