import os
import pickle
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, pairwise_distances


from famus.logging import logger
from famus.model import MLP
from famus.sdf import SparseDataFrame, load
from famus.utils import even_split

try:
    import torch
except ImportError:
    logger.warning(
        "PyTorch is not installed. Please install PyTorch to use the classification module."
    )


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


def _calc_embeddings(sdf, model: MLP, device: Any, chunksize, n_processes):
    logger.info("Calculating embeddings from scratch")
    chunksize = min(chunksize, len(sdf))
    num_chunks = int(np.ceil(len(sdf) / chunksize))
    logger.debug(f"Number of chunks: {num_chunks}")
    embeddings = []
    if device == "cpu":
        torch.set_num_threads(n_processes)
    with torch.no_grad():
        for i in range(num_chunks):
            logger.debug(f"Calculating chunk {i + 1} / {num_chunks}")
            chunk = sdf.matrix[
                i * chunksize : min((i + 1) * chunksize, len(sdf))
            ].todense()
            chunk_tensor = torch.tensor(
                chunk, requires_grad=False, dtype=torch.float32
            ).to(device)
            embeddings.append(model(chunk_tensor).cpu().numpy())
            del chunk_tensor

    # return a numpy array of the embeddings
    concatenated = np.concatenate(embeddings)
    return concatenated


def get_embeddings(
    sdf: SparseDataFrame,
    model: MLP,
    embeddings_path: str,
    device,
    chunksize,
    use_saved_embeddings,
    n_processes,
):
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
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.to(device)
    return embeddings


def calculate_threshold(
    sdf_train_path: str,
    model_path: str,
    train_embeddings_path: str,
    device,
    chunksize,
    n_processes,
    load_sdf_from_pickle=False,
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

    model = MLP.load_from_state(model_path, device=device)
    model.eval()
    model.to(device)
    logger.info("Loading dataframes")
    if load_sdf_from_pickle:
        with open(sdf_train_path, "rb") as f:
            sdf_train = pickle.load(f)
    else:
        sdf_train = load(sdf_train_path)

    logger.info("Calculating threshold")
    threshold = calc_thresh(
        sdf_train=sdf_train,
        model=model,
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
    device,
    chunksize,
    n_processes,
) -> None:
    """
    Generates a distance threshold for classification.
    Assumes there are at least a few thousand labeled and unlabeled points in sdf_train.
    """
    model.eval()
    logger.info("Balancing labeled and unknown samples in training set")
    unlabeled_indices, singly_labeled_indices = [], []
    for i, index_id in enumerate(sdf_train.index_ids):
        if len(sdf_train.labels[index_id]) > 1:
            continue
        if sdf_train.labels[index_id][0] == "unknown":
            unlabeled_indices.append(i)
        else:
            singly_labeled_indices.append(i)

    n_unlabeled = len(unlabeled_indices)
    n_labeled = len(singly_labeled_indices)

    if n_unlabeled > n_labeled:
        sampled_labeled_indices = np.random.choice(
            singly_labeled_indices, size=n_unlabeled, replace=True
        ).tolist()
        balanced_indices = set(unlabeled_indices + sampled_labeled_indices)
        mask = np.array(
            [i in balanced_indices for i in range(len(sdf_train))], dtype=bool
        )
    else:
        sampled_unlabeled_indices = np.random.choice(
            unlabeled_indices, size=n_labeled, replace=True
        ).tolist()
        balanced_indices = set(sampled_unlabeled_indices + singly_labeled_indices)
        mask = np.array(
            [i in balanced_indices for i in range(len(sdf_train))], dtype=bool
        )
    sdf_train = sdf_train._keep_indices(mask)
    logger.info("Getting embeddings")
    train_embeddings = get_embeddings(
        sdf_train,
        model,
        train_embeddings_path,
        device,
        chunksize,
        use_saved_embeddings=True,
        n_processes=n_processes,
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
            curr_train_embeddings,
            curr_test_embeddings,
            device,
            n_processes,
            chunksize=chunksize,
        )

        y_pred = np.array([train_labels[k] for k in closest_indices])

        y_true = np.array([labels[k] for k in sdf_train.index_ids[test_indices]])

        max_f = 0
        best_thresh = 0
        for i in range(200, 0, -1):
            thresh = i / 100
            logger.debug("Current threshold: " + str(thresh))
            y_pred[closest_distances >= thresh] = "unknown"

            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            logger.debug("weighted fscore: " + str(weighted_f1))
            if weighted_f1 > max_f:
                max_f = weighted_f1
                best_thresh = thresh
        for i in range(99, 0, -1):
            thresh = i / 1000
            logger.debug("Current threshold: " + str(thresh))
            y_pred[closest_distances >= thresh] = "unknown"

            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            logger.debug("weighted fscore: " + str(weighted_f1))
            if weighted_f1 > max_f:
                max_f = weighted_f1
                best_thresh = thresh
        max_f1_scores.append(max_f)
        max_thresholds.append(best_thresh)
    avg_max_f1_score = np.mean(max_f1_scores)
    avg_threshold = np.mean(max_thresholds)
    logger.debug("average fscore: " + str(avg_max_f1_score))
    logger.info("selected threshold: " + str(avg_threshold))
    return avg_threshold


def classify(
    sdf_train_path: str,
    sdf_classify_path: str,
    model_path: str,
    train_embeddings_path: None | str,
    classification_embeddings_path: None | str,
    output_path: str,
    device,
    chunksize,
    threshold,
    n_processes,
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

    model = MLP.load_from_state(model_path, device=device)
    model.to(device)
    model.eval()
    sdf_train: SparseDataFrame
    sdf_classify: SparseDataFrame
    logger.info("Loading dataframes")
    sdf_train, sdf_classify = load_sparse_dataframes(
        sdf_train_path, sdf_classify_path, load_sdf_from_pickle
    )

    if threshold == "bootstrap":
        threshold = calc_thresh(
            sdf_train=sdf_train,
            model=model,
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
        sdf=sdf_train,
        model=model,
        embeddings_path=train_embeddings_path,
        device=device,
        chunksize=chunksize,
        use_saved_embeddings=True,
        n_processes=n_processes,
    )
    classifiy_embeddings = get_embeddings(
        sdf=sdf_classify,
        model=model,
        embeddings_path=classification_embeddings_path,
        device=device,
        chunksize=chunksize,
        use_saved_embeddings=False,
        n_processes=n_processes,
    )
    logger.debug("Train embeddings shape: " + str(train_embeddings.shape))
    logger.debug("Classify embeddings shape: " + str(classifiy_embeddings.shape))
    train_data_labels = sdf_train.labels
    train_data_labels = [sdf_train.labels[k] for k in sdf_train.index_ids]
    train_data_labels = [";".join(labels) for labels in train_data_labels]
    train_data_labels = np.array(train_data_labels, dtype="object")
    logger.debug("Train data labels shape: " + str(train_data_labels.shape))
    closest_distances, closest_indices = get_closest_distances_and_indices(
        train_embeddings, classifiy_embeddings, device, n_processes, chunksize=chunksize
    )

    classifications = train_data_labels[closest_indices]
    classifications[closest_distances >= threshold] = "unknown"

    logger.info("Saving predictions")
    with open(output_path, "w+") as f:
        for i, label in enumerate(classifications):
            f.write(sdf_classify.index_ids[i] + "\t" + label + "\n")


def get_closest_distances_and_indices(
    train_embeddings, classify_embeddings, device, n_processes, chunksize
):
    """
    For each embedding in classify_embeddings, finds the closest embedding in train_embeddings.
    closest_distances[i] is the distance between classify_embeddings[i] and its closest embedding in train_embeddings.
    closest_indices[i] is the index of the closest embedding in train_embeddings to classify_embeddings[i].
    Returns:
        closest_distances: np.ndarray of shape (len(classify_embeddings),)
        closest_indices: np.ndarray of shape (len(classify_embeddings),)

    """
    if device.type == "cpu":
        return _get_closest_cpu(
            train_embeddings, classify_embeddings, n_processes, chunksize
        )

    # GPU optimized version
    classify_chunksize = chunksize * 5
    train_chunksize = chunksize

    n_classify = len(classify_embeddings)
    closest_distances = torch.full(
        (n_classify,), float("inf"), device=device, dtype=torch.float32
    )
    closest_indices = torch.zeros(n_classify, dtype=torch.long, device=device)

    classify_offset = 0
    i = 0
    num_classify_chunks = int(np.ceil(len(classify_embeddings) / classify_chunksize))
    with torch.no_grad():
        for classify_chunk in _chunks(classify_embeddings, classify_chunksize):
            logger.debug(f"get closest: classify chunk {i + 1} / {num_classify_chunks}")
            chunk_size = len(classify_chunk)
            classify_chunk = classify_chunk.to(device)
            chunk_min_dists = torch.full(
                (chunk_size,), float("inf"), device=device, dtype=torch.float32
            )
            chunk_min_indices = torch.zeros(chunk_size, dtype=torch.long, device=device)

            train_offset = 0
            for train_chunk in _chunks(train_embeddings, train_chunksize):
                train_chunk = train_chunk.to(device)
                distances = torch.cdist(classify_chunk, train_chunk)
                min_dists, min_idxs = torch.min(distances, dim=1)
                min_idxs = min_idxs + train_offset
                mask = min_dists < chunk_min_dists
                chunk_min_dists = torch.where(mask, min_dists, chunk_min_dists)
                chunk_min_indices = torch.where(mask, min_idxs, chunk_min_indices)
                train_offset += len(train_chunk)

            closest_distances[classify_offset : classify_offset + chunk_size] = (
                chunk_min_dists
            )
            closest_indices[classify_offset : classify_offset + chunk_size] = (
                chunk_min_indices
            )

            classify_offset += chunk_size
            i += 1

    return closest_distances.cpu().numpy(), closest_indices.cpu().numpy()


def _chunks(tensor, chunk_size):
    """Generator for tensor chunks"""
    for i in range(0, len(tensor), chunk_size):
        yield tensor[i : i + chunk_size]


def _get_closest_cpu(train_embeddings, classify_embeddings, n_processes, chunksize):
    train_np = (
        train_embeddings.numpy()
        if torch.is_tensor(train_embeddings)
        else train_embeddings
    )
    classify_np = (
        classify_embeddings.numpy()
        if torch.is_tensor(classify_embeddings)
        else classify_embeddings
    )

    n_classify = len(classify_np)
    closest_distances = np.full(n_classify, np.inf, dtype=np.float32)
    closest_indices = np.zeros(n_classify, dtype=np.int64)
    num_classify_chunks = int(np.ceil(n_classify / chunksize))
    num_train_chunks = int(np.ceil(len(train_np) / chunksize))
    chunk_count = 0
    for i in range(0, n_classify, chunksize):
        chunk_count += 1
        classify_chunk = classify_np[i : i + chunksize]
        logger.debug(
            f"get closest: classify chunk {chunk_count} / {num_classify_chunks}"
        )
        train_chunk_count = 0
        for j in range(0, len(train_np), chunksize):
            train_chunk_count += 1
            logger.debug(
                f"get closest: train chunk {train_chunk_count} / {num_train_chunks}"
            )
            train_chunk = train_np[j : j + chunksize]

            distances = pairwise_distances(
                classify_chunk, train_chunk, n_jobs=n_processes
            )

            min_idxs = np.argmin(distances, axis=1)
            min_dists = distances[np.arange(len(distances)), min_idxs]
            min_idxs = min_idxs + j

            # Update minimums
            mask = min_dists < closest_distances[i : i + len(classify_chunk)]
            closest_distances[i : i + len(classify_chunk)][mask] = min_dists[mask]
            closest_indices[i : i + len(classify_chunk)][mask] = min_idxs[mask]

    return closest_distances, closest_indices
