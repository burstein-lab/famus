import os
import pickle
from typing import Any, Tuple

import numpy as np
from sklearn.metrics import f1_score, pairwise_distances


from famus import get_cfg, logger
from famus import now as now_func
from famus.model import MLP
from famus.sdf import SparseDataFrame, load
from famus.utils import even_split
from famus.model import load_from_state

try:
    import torch
    from torch import cdist
except ImportError:
    logger.warning(
        "PyTorch is not installed. Please install PyTorch to use the classification module."
    )

cfg = get_cfg()
user_device = cfg["user_device"]
n_processes = cfg["n_processes"]
threshold = cfg["threshold"]
chunksize = cfg["chunksize"]  # reduce here or in cfg.yaml if GPU RAM becomes an issue


def update_unknown_labels(
    sdf: SparseDataFrame,
    threshold: float,
    model: MLP,
    device=user_device,
    n_processes=n_processes,
    chunksize=chunksize,
    embeddings_path: str | None = None,
) -> SparseDataFrame:
    """
    Updates the labels of unknown samples in the given SparseDataFrame.
    This function identifies samples labeled as "unknown", checks their embeddings' nearest neighbor,
    and updates their labels based on the nearest neighbor's label if the distance is below the threshold.
    If the distance is above the threshold, the sample remains labeled as "unknown".
    """
    embeddings = get_embeddings(
        sdf,
        model=model,
        embeddings_path=embeddings_path,
        device=device,
        chunksize=chunksize,
    )
    labeled_samples_labels = [
        sdf.labels[k] for k in sdf.index_ids if "unknown" not in sdf.labels[k]
    ]  # labels of known samples in the same order as the embeddings. It's length is the same as the length of labeled_embeddings
    unknown_sample_indices = [
        i for i, labels in enumerate(sdf.labels.values()) if "unknown" in labels
    ]  # indices of unknown samples in the same order as the embeddings
    known_sample_indices = [
        i
        for i, sample_index_ids in enumerate(sdf.index_ids)
        if "unknown" not in sdf.labels[sample_index_ids]
    ]  # indices of known samples in the same order as the embeddings
    unknown_sdf = sdf._keep_indices(unknown_sample_indices)

    unknown_embeddings = embeddings[unknown_sample_indices]
    labeled_embeddings = embeddings[known_sample_indices]
    logger.info("Calculating distances between unknown and labeled samples")
    closest_distances, closest_indices = get_closest_distances_and_indices(
        labeled_embeddings,
        unknown_embeddings,
        device=device,
        n_processes=n_processes,
    )
    passed_indices = np.where(closest_distances < threshold)[0]
    logger.info(
        f"Found {len(passed_indices)} / {len(unknown_sdf)} unknown samples with distances below the threshold of {threshold}"
    )
    for i, (closest_distance, closest_index) in enumerate(
        zip(closest_distances, closest_indices)
    ):
        if closest_distance < threshold:
            unknown_sdf.labels[unknown_sdf.index_ids[i]] = labeled_samples_labels[
                closest_index
            ]
    logger.info("Finished updating unknown labels in SparseDataFrame")
    return sdf


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
    embeddings_a,
    embeddings_b,
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
    logger.info("average labelled/unknown fscore: " + str(avg_max_f1_score))
    logger.info("selected threshold: " + str(avg_threshold))
    return avg_threshold


def classify_2(
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
    """
    Classification using weighted voting from all neighbors within threshold.
    Weight = threshold - distance + 1e-10 for each neighbor.
    Label with highest total weight is assigned.
    """
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

    snn_model: MLP = load_from_state(model_path, device=device)
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
    classify_embeddings = get_embeddings(
        sdf_classify, snn_model, classification_embeddings_path, device, chunksize
    )

    train_data_labels = sdf_train.labels
    train_data_labels = [sdf_train.labels[k] for k in sdf_train.index_ids]
    train_data_labels = [";".join(labels) for labels in train_data_labels]
    train_data_labels = np.array(train_data_labels, dtype="object")

    logger.info("Finding all neighbors within threshold")
    neighbor_indices, neighbor_distances = (
        get_nearest_neighbors_and_distances_optimized(
            train_embeddings,
            classify_embeddings,
            threshold,
            device,
            n_processes,
            chunksize,
        )
    )

    logger.info("Performing weighted voting classification")
    classifications = []

    for i, (indices, distances) in enumerate(zip(neighbor_indices, neighbor_distances)):
        if len(indices) == 0:
            # No neighbors within threshold
            classifications.append("unknown")
        else:
            # Calculate weights and perform voting
            label_weights = {}

            for neighbor_idx, distance in zip(indices, distances):
                weight = ((threshold - distance) + 1e-10 / threshold + 1e-10) ** 10
                neighbor_label_string = train_data_labels[neighbor_idx]

                # Split multi-labels and add weight to each individual label
                individual_labels = neighbor_label_string.split(";")
                for label in individual_labels:
                    label = label.strip()  # Remove any whitespace
                    if label in label_weights:
                        label_weights[label] += weight
                    else:
                        label_weights[label] = weight

            # Find label with maximum weight
            best_label = max(label_weights.keys(), key=lambda x: label_weights[x])
            classifications.append(best_label)

    logger.info("Saving predictions")
    with open(output_path, "w+") as f:
        for i, label in enumerate(classifications):
            f.write(sdf_classify.index_ids[i] + "\t" + label + "\n")


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

    snn_model: MLP = load_from_state(model_path, device=device)
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


def get_nearest_neighbors_and_distances(
    target_embeddings,
    query_embeddings,
    threshold,
    device,
    n_processes,
    chunksize=chunksize,
):
    """
    For each query embedding, finds all target embeddings within the threshold distance.

    Args:
        target_embeddings: Embeddings to search in (the "database")
        query_embeddings: Embeddings to find neighbors for
        threshold: Maximum distance for neighbors
        device: Device to use for computation
        n_processes: Number of processes for CPU computation
        chunksize: Size of chunks for processing

    Returns:
        neighbor_indices: List of arrays, where neighbor_indices[i] contains indices of
                         all target embeddings within threshold of query i
        neighbor_distances: List of arrays, where neighbor_distances[i] contains distances
                           of all target embeddings within threshold of query i
    """
    neighbor_indices = []
    neighbor_distances = []

    with torch.no_grad():
        for i, query_chunk in enumerate(
            embedding_chunk_iterator(query_embeddings, chunksize, "query")
        ):
            query_chunk = query_chunk.to(device)

            # For each query in this chunk, collect all neighbors within threshold
            chunk_neighbor_indices = [[] for _ in range(len(query_chunk))]
            chunk_neighbor_distances = [[] for _ in range(len(query_chunk))]

            target_indices_offset = 0
            for j, target_chunk in enumerate(
                embedding_chunk_iterator(target_embeddings, chunksize, "target")
            ):
                target_chunk = target_chunk.to(device)

                # Compute all pairwise distances between query_chunk and target_chunk
                if device == "cpu":
                    distances = pairwise_distances(
                        query_chunk.cpu().numpy(),
                        target_chunk.cpu().numpy(),
                        n_jobs=n_processes,
                    )
                    distances = torch.tensor(distances, dtype=torch.float32)
                else:
                    distances = cdist(query_chunk, target_chunk)

                # For each query in the chunk, find targets within threshold
                for query_idx in range(len(query_chunk)):
                    query_distances = distances[query_idx]
                    within_threshold_mask = query_distances < threshold

                    if torch.any(within_threshold_mask):
                        # Get indices and distances of neighbors within threshold
                        local_target_indices = torch.where(within_threshold_mask)[0]
                        global_target_indices = (
                            local_target_indices + target_indices_offset
                        )
                        corresponding_distances = query_distances[within_threshold_mask]

                        # Add to the lists for this query
                        chunk_neighbor_indices[query_idx].extend(
                            global_target_indices.cpu().numpy()
                        )
                        chunk_neighbor_distances[query_idx].extend(
                            corresponding_distances.cpu().numpy()
                        )

                target_indices_offset += len(target_chunk)

            # Convert lists to numpy arrays for each query in the chunk
            for query_idx in range(len(query_chunk)):
                neighbor_indices.append(
                    np.array(chunk_neighbor_indices[query_idx], dtype=int)
                )
                neighbor_distances.append(
                    np.array(chunk_neighbor_distances[query_idx], dtype=float)
                )

    return neighbor_indices, neighbor_distances


def get_nearest_neighbors_and_distances_optimized(
    target_embeddings,
    query_embeddings,
    threshold,
    device,
    n_processes,
    chunksize=chunksize,
):
    """
    Optimized version that processes the threshold filtering more efficiently.
    This version is better for cases where you expect many neighbors within threshold.
    """
    neighbor_indices = []
    neighbor_distances = []

    with torch.no_grad():
        for i, query_chunk in enumerate(
            embedding_chunk_iterator(query_embeddings, chunksize, "query")
        ):
            query_chunk = query_chunk.to(device)

            # Initialize result containers for this chunk
            chunksize = len(query_chunk)
            chunk_neighbor_indices = [[] for _ in range(chunksize)]
            chunk_neighbor_distances = [[] for _ in range(chunksize)]

            target_indices_offset = 0
            for j, target_chunk in enumerate(
                embedding_chunk_iterator(target_embeddings, chunksize, "target")
            ):
                target_chunk = target_chunk.to(device)

                # Compute distances
                if device == "cpu":
                    distances = pairwise_distances(
                        query_chunk.cpu().numpy(),
                        target_chunk.cpu().numpy(),
                        n_jobs=n_processes,
                    )
                    distances = torch.tensor(
                        distances, dtype=torch.float32, device=device
                    )
                else:
                    distances = cdist(query_chunk, target_chunk)

                # Find all pairs within threshold using vectorized operations
                within_threshold_mask = distances < threshold
                query_indices, target_indices = torch.where(within_threshold_mask)

                if len(query_indices) > 0:
                    # Get the actual distances for the pairs within threshold
                    threshold_distances = distances[within_threshold_mask]

                    # Adjust target indices to global indices
                    global_target_indices = target_indices + target_indices_offset

                    # Group by query index
                    for qi, ti, dist in zip(
                        query_indices.cpu().numpy(),
                        global_target_indices.cpu().numpy(),
                        threshold_distances.cpu().numpy(),
                    ):
                        chunk_neighbor_indices[qi].append(ti)
                        chunk_neighbor_distances[qi].append(dist)

                target_indices_offset += len(target_chunk)

            # Convert to numpy arrays
            for query_idx in range(chunksize):
                neighbor_indices.append(
                    np.array(chunk_neighbor_indices[query_idx], dtype=int)
                )
                neighbor_distances.append(
                    np.array(chunk_neighbor_distances[query_idx], dtype=float)
                )

    return neighbor_indices, neighbor_distances
