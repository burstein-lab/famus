import os
import pickle
from typing import Any, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, pairwise_distances
from torch import cdist

from app import get_cfg, logger
from app import now as now_func
from app.model import MLP
from app.sdf import SparseDataFrame, load
from app.utils import even_split
from app.model import load_from_state

cfg = get_cfg()
user_device = cfg["user_device"]
n_processes = cfg["n_processes"]
threshold = cfg["threshold"]
chunksize = cfg["chunksize"]  # reduce here or in cfg.yaml if GPU RAM becomes an issue


def load_sparse_dataframes(
    sdf_train_path: str, sdf_classify_path: str, load_sdf_from_pickle=False
) -> tuple:
    """
    Loads the SparseDataFrames from the given paths
    """
    sdf_train: SparseDataFrame
    sdf_classify: SparseDataFrame
    if load_sdf_from_pickle:
        with open(sdf_train_path, "rb") as f:
            sdf_train = pickle.load(f)
    else:
        sdf_train = load(sdf_train_path)
    with open(sdf_classify_path, "rb") as f:
        sdf_classify = pickle.load(f)
    sdf_train.matrix = sdf_train.matrix.astype(np.float32)
    sdf_classify.matrix = sdf_classify.matrix.astype(np.float32)
    return sdf_train, sdf_classify


def _min_dist_ind(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    device,
    n_processes=n_processes,
) -> Tuple[np.ndarray, np.ndarray]:
    if device == "cpu":
        a_b_distances = pairwise_distances(
            embeddings_a.cpu().numpy(), embeddings_b.cpu().numpy(), n_jobs=n_processes
        )
        a_b_distances = torch.tensor(a_b_distances, dtype=torch.float32)
    else:
        a_b_distances = cdist(embeddings_a, embeddings_b)
        min_distances: torch.Tensor
        min_indices: torch.Tensor
    min_distances, min_indices = torch.min(a_b_distances, dim=1)
    output = min_distances.cpu().numpy(), min_indices.cpu().numpy()
    return output


def _calc_embeddings(
    sdf, model: MLP, device: Any, chunksize=chunksize, n_processes=n_processes
):
    logger.info("Calculating embeddings from scratch")
    chunksize = min(chunksize, len(sdf))
    num_chunks = int(np.ceil(len(sdf) / chunksize))
    embeddings = []
    if device == "cpu":
        torch.set_num_threads(n_processes)
    with torch.no_grad():
        for i in range(num_chunks):
            chunk = sdf.matrix[
                i * chunksize : min((i + 1) * chunksize, len(sdf))
            ].todense()
            chunk_tensor = torch.tensor(
                chunk, requires_grad=False, dtype=torch.float32
            ).to(device)
            embeddings.append(model.forward_once(chunk_tensor))
            del chunk_tensor

    # return a numpy array of the embeddings
    concatenated = torch.cat(embeddings, dim=0)
    return concatenated.cpu().numpy()


def get_embeddings(
    sdf: SparseDataFrame,
    model: MLP,
    embeddings_path: str,
    device=user_device,
    chunksize=chunksize,
    use_saved_embeddings=True,
    n_processes=n_processes,
) -> torch.Tensor:
    if embeddings_path is None or embeddings_path == "":
        use_saved_embeddings = False
    if use_saved_embeddings and os.path.exists(embeddings_path):
        logger.info("Loading embeddings")
        embeddings: np.ndarray = np.load(embeddings_path)
    else:
        logger.info("Calculating embeddings")
        embeddings = _calc_embeddings(sdf, model, device, chunksize, n_processes)
        if use_saved_embeddings:
            logger.info("Saving embeddings")
            np.save(embeddings_path, embeddings)
    embeddings: torch.Tensor = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.to(device)
    return embeddings


def calculate_threshold(
    sdf_train_path: str,
    model_path: str,
    train_embeddings_path: str,
    device=user_device,
    chunksize=chunksize,
    n_processes=n_processes,
) -> float:
    """
    Calculates a threshold for classification.
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            raise torch.cuda.CudaError("CUDA is not available")
        else:
            device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
        torch.set_num_threads(n_processes)
    else:
        device = torch.device(device)

    logger.info("device: " + str(device))
    logger.info("Loading model")

    snn_model: MLP = load_from_state(model_path)
    snn_model.eval()
    snn_model.to(device)
    sdf_train: SparseDataFrame
    logger.info("Loading dataframes")
    sdf_train = load(sdf_train_path)

    logger.info("Calculating threshold")
    threshold = calc_thresh(
        sdf_train=sdf_train,
        model=snn_model,
        train_embeddings_path=train_embeddings_path,
        device=device,
        chunksize=chunksize,
        n_processes=n_processes,
    )
    return threshold


def calc_thresh(
    sdf_train,
    model,
    train_embeddings_path,
    device=user_device,
    chunksize=chunksize,
    n_processes=n_processes,
) -> None:
    """
    Generates a distance threshold for classification.
    """
    model.eval()

    logger.info("Getting embeddings")
    train_embeddings = get_embeddings(
        sdf_train, model, train_embeddings_path, device, chunksize
    )
    indices = np.arange(len(train_embeddings))
    np.random.shuffle(indices)
    split_indices = even_split(indices, 3)
    max_thresholds = []
    max_f1_scores = []
    labels = sdf_train.labels
    for fold in range(3):
        train_indices = []
        for i in range(3):
            if not i == fold:
                train_indices.extend(split_indices[i])
        curr_train_embeddings = train_embeddings[train_indices]
        train_labels = [labels[k] for k in sdf_train.index_ids[train_indices]]

        test_indices = split_indices[fold]
        curr_test_embeddings = train_embeddings[test_indices]
        closest_distances, closest_indices = get_closest_distances_and_indices(
            curr_train_embeddings, curr_test_embeddings, device, n_processes
        )

        y_pred = [train_labels[k] for k in closest_indices]

        ground_truth = [labels[k] for k in sdf_train.index_ids[test_indices]]

        for i, preds_array in enumerate(y_pred):
            curr_ground_truth_array = ground_truth[i]
            if (len(curr_ground_truth_array) > 1 or len(preds_array) > 1) and len(
                set(preds_array) & set(curr_ground_truth_array)
            ) > 0:
                y_pred[i] = curr_ground_truth_array[0]
            else:
                y_pred[i] = preds_array[0]
            ground_truth[i] = curr_ground_truth_array[0]
        y_true, y_pred = np.array(ground_truth), np.array(y_pred)

        max_f = 0
        best_thresh = 0
        for i in range(50, 1, -1):
            thresh = i / 100
            logger.info("Current threshold: " + str(thresh))
            y_pred[closest_distances >= thresh] = "unknown"

            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            logger.info("weighted fscore: " + str(weighted_f1))
            if weighted_f1 > max_f:
                max_f = weighted_f1
                best_thresh = thresh
        max_f1_scores.append(max_f)
        max_thresholds.append(best_thresh)
    avg_max_f1_score = np.mean(max_f1_scores)
    avg_threshold = np.mean(max_thresholds)
    logger.info("average fscore: " + str(avg_max_f1_score))
    logger.info("selected threshold: " + str(avg_threshold))
    return avg_threshold


def classify(
    sdf_train_path: str,
    sdf_classify_path: str,
    model_path: str,
    train_embeddings_path: None | str,
    classification_embeddings_path: None | str,
    output_path: str,
    device=user_device,
    chunksize=chunksize,
    threshold=threshold,
    n_processes=n_processes,
    load_sdf_from_pickle=False,
) -> None:
    if os.path.exists(output_path):
        logger.warning("Output file already exists. Overwriting.")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise torch.cuda.CudaError("CUDA is not available")
        else:
            device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
        torch.set_num_threads(n_processes)
    else:
        device = torch.device(device)

    logger.info("device: " + str(device))
    logger.info("Loading model")

    snn_model: MLP = load_from_state(model_path)
    snn_model.to(device)
    snn_model.eval()
    sdf_train: SparseDataFrame
    sdf_classify: SparseDataFrame
    logger.info("Loading dataframes")
    sdf_train, sdf_classify = load_sparse_dataframes(
        sdf_train_path, sdf_classify_path, load_sdf_from_pickle
    )

    if threshold == "bootstrap":
        threshold = calc_thresh(
            sdf_train=sdf_train,
            model=snn_model,
            train_embeddings_path=train_embeddings_path,
            device=device,
            chunksize=chunksize,
            n_processes=n_processes,
        )
    else:
        try:
            threshold = float(threshold)
        except ValueError:
            raise ValueError(
                "threshold must be either 'bootstrap' or a float, but was {}".format(
                    threshold
                )
            )
    logger.info("threshold: " + str(threshold))
    logger.info("Getting embeddings")
    train_embeddings = get_embeddings(
        sdf_train, snn_model, train_embeddings_path, device, chunksize
    )
    classifiy_embeddings = get_embeddings(
        sdf_classify, snn_model, classification_embeddings_path, device, chunksize
    )

    train_data_labels = sdf_train.labels
    train_data_labels = [sdf_train.labels[k] for k in sdf_train.index_ids]
    train_data_labels = [";".join(labels) for labels in train_data_labels]
    train_data_labels = np.array(train_data_labels, dtype="object")

    closest_distances, closest_indices = get_closest_distances_and_indices(
        train_embeddings, classifiy_embeddings, device, n_processes
    )

    classifications = train_data_labels[closest_indices]
    classifications[closest_distances >= threshold] = "unknown"

    logger.info("Saving predictions")
    with open(output_path, "w+") as f:
        for i, label in enumerate(classifications):
            f.write(sdf_classify.index_ids[i] + "\t" + label + "\n")


def get_closest_distances_and_indices(
    train_embeddings, classifiy_embeddings, device, n_processes, chunksize=chunksize
):
    closest_distances = np.array([], dtype=float)
    closest_indices = np.array([], dtype=int)
    with torch.no_grad():
        for i, classify_chunk in enumerate(
            embedding_chunk_iterator(classifiy_embeddings, chunksize, "classify")
        ):
            classify_chunk.to(device)

            curr_closest_distances = np.array(
                [np.inf for _ in range(len(classify_chunk))]
            )
            curr_closest_indices = np.array([0 for _ in range(len(classify_chunk))])
            indices_offset = 0
            for j, train_chunk in enumerate(
                embedding_chunk_iterator(train_embeddings, chunksize, "train")
            ):
                train_chunk.to(device)

                min_distances, min_indices = _min_dist_ind(
                    classify_chunk, train_chunk, device, n_processes
                )
                min_indices = min_indices + indices_offset
                is_below_curr_closest_distance = min_distances < curr_closest_distances
                curr_closest_distances[is_below_curr_closest_distance] = min_distances[
                    is_below_curr_closest_distance
                ]
                curr_closest_indices[is_below_curr_closest_distance] = min_indices[
                    is_below_curr_closest_distance
                ]
                indices_offset += len(train_chunk)

            closest_distances = np.concatenate(
                (closest_distances, curr_closest_distances)
            )
            closest_indices = np.concatenate((closest_indices, curr_closest_indices))
    return closest_distances, closest_indices


def embedding_chunk_iterator(embeddings, chunksize, name=""):
    num_chunks = int(np.ceil(len(embeddings) / chunksize))
    if name:
        name += " "
    for i in range(num_chunks):
        yield embeddings[i * chunksize : min((i + 1) * chunksize, len(embeddings))]
