import os
import pickle
from typing import Any, Tuple

import numpy as np
import torch
from sklearn.metrics import pairwise_distances, precision_recall_curve
from torch import cdist

from app import get_cfg, logger
from app.model import VariableNet
from app.sdf import SparseDataFrame
from app.utils import even_split

cfg = get_cfg()
user_device = cfg["user_device"]
nthreads = cfg["nthreads"]
threshold = cfg["threshold"]
chunksize = cfg['chunksize']  # reduce here or in cfg.yaml if GPU RAM becomes an issue


def load_sparse_dataframes(sdf_train_path: str, sdf_classify_path: str) -> tuple:
    """
    Loads the SparseDataFrames from the given paths
    :param sdf_train_path: path to the training SparseDataFrame
    :param sdf_classify_path: path to the classification SparseDataFrame
    :return: tuple of the training and classification SparseDataFrames
    """
    sdf_train: SparseDataFrame
    sdf_classify: SparseDataFrame
    logger.info("Loading dataframes")
    sdf_train = pickle.load(open(sdf_train_path, "rb"))
    sdf_classify = pickle.load(open(sdf_classify_path, "rb"))
    sdf_train.matrix = sdf_train.matrix.astype(np.float32)
    sdf_classify.matrix = sdf_classify.matrix.astype(np.float32)
    return sdf_train, sdf_classify


def calculate_threshold(
    sdf_train: SparseDataFrame,
    model: VariableNet,
    train_embeddings_path: str,
    device,
    chunksize=chunksize,
    nthreads=nthreads,
) -> float:
    """
    computes the threshold for classifying a label as unknown
    :param sdf_train: training data
    :param model: model to use for classification
    :param train_embeddings_path: path to load/save the training embeddings to
    :param device: device to use for classification
    :param chunksize: chunksize to use for data
    :return: threshold for classifying a label as unknown
    """
    model.eval()
    sdf_train.matrix = sdf_train.matrix.astype(np.float32)
    labels = sdf_train.labels
    all_embeddings = get_embeddings(sdf_train, model, train_embeddings_path, device)
    embedding_splits = []
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    split_indices = even_split(indices, 3)
    for fold in range(3):
        embedding_splits.append(all_embeddings[split_indices[fold]])
    max_thresholds = []
    max_f1_scores = []
    for fold in range(3):
        train_indices = []
        for i in range(3):
            if not i == fold:
                train_indices.extend(split_indices[i])
        train_embeddings = torch.concatenate(
            tuple([embedding_splits[i] for i in range(3) if not i == fold])
        )
        test_embeddings = embedding_splits[fold]
        train_labels = labels[train_indices]
        y_test = labels[split_indices[fold]]
        y_test = np.array([1 if label == "unknown" else 0 for label in y_test])
        closest_distances = np.array([])
        known_mask = train_labels != "unknown"
        train_embeddings = train_embeddings[known_mask]
        train_labels = train_labels[known_mask]

        train_chunksize = min(chunksize, len(train_embeddings))
        test_chunksize = min(chunksize, len(test_embeddings))

        num_train_chunks = int(np.ceil(len(train_embeddings) / train_chunksize))
        num_test_chunks = int(np.ceil(len(test_embeddings) / test_chunksize))
        with torch.no_grad():
            for i in range(num_test_chunks):
                logger.info(
                    "Predicting test chunk for threshold calculation {}/{}, eval round {}/{}".format(
                        i + 1, num_test_chunks, fold + 1, 3
                    )
                )
                test_chunk = test_embeddings[
                    i * test_chunksize : min(
                        (i + 1) * test_chunksize, len(test_embeddings)
                    )
                ].to(device)
                curr_closest_distances = np.array(
                    [np.inf for _ in range(len(test_chunk))]
                )
                for j in range(num_train_chunks):
                    train_chunk = train_embeddings[
                        j * train_chunksize : min(
                            (j + 1) * train_chunksize, len(train_embeddings)
                        )
                    ].to(device)
                    min_distances, _ = get_min_distances(
                        test_chunk, train_chunk, device, nthreads
                    )
                    below_curr_closest_distance = min_distances < curr_closest_distances
                    curr_closest_distances[below_curr_closest_distance] = min_distances[
                        below_curr_closest_distance
                    ]
                closest_distances = np.concatenate(
                    (closest_distances, curr_closest_distances)
                )
        precision, recall, thresholds = precision_recall_curve(
            y_test, closest_distances
        )
        f1_scores = np.array(
            [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(precision, recall)]
        )
        max_f1_score = np.max(f1_scores)
        max_f1_score_idx = np.argmax(f1_scores)
        distance_threshold = thresholds[max_f1_score_idx]
        logger.info("current fold threshold: " + str(distance_threshold))
        logger.info("current fold max fscore: " + str(max_f1_score))
        max_thresholds = np.append(max_thresholds, distance_threshold)
        max_f1_scores = np.append(max_f1_scores, max_f1_score)
    avg_max_f1_score = np.mean(max_f1_scores)
    avg_threshold = np.mean(max_thresholds)
    logger.info("average fscore: " + str(avg_max_f1_score))
    logger.info("selected threshold: " + str(avg_threshold))
    return avg_threshold


def calculate_threshold_no_unknowns(
    sdf_train: SparseDataFrame,
    model: VariableNet,
    train_embeddings_path: str,
    device,
    chunksize=chunksize,
    nthreads=nthreads,
) -> float:
    """
    computes the threshold for classifying a label as unknown, using known labels of other classes as negative examples
    :param sdf_train: training data
    :param model: model to use for classification
    :param train_embeddings_path: path to load/save the training embeddings to
    :param device: device to use for classification
    :param chunksize: chunksize to use for data
    :return: threshold for classifying a label as unknown
    """
    model.eval()
    sdf_train.matrix = sdf_train.matrix.astype(np.float32)
    labels = sdf_train.labels
    all_embeddings = get_embeddings(sdf_train, model, train_embeddings_path, device)
    assert len(all_embeddings) == len(labels)
    indices = np.arange(len(labels))

    np.random.shuffle(indices)
    split_indices = even_split(indices, 3)
    split_indices = [split_indices[i] for i in range(len(split_indices))]
    max_thresholds = []
    max_f1_scores = []
    with torch.no_grad():
        for fold in range(3):
            train_indices = []
            for i in range(3):
                if not i == fold:
                    train_indices.extend(split_indices[i])
            train_indices = sorted(train_indices)
            validation_indices = sorted(split_indices[fold])
            train_embeddings = all_embeddings[train_indices]
            train_data_labels = labels[train_indices]
            validation_data_labels = labels[validation_indices]
            test_embeddings = all_embeddings[validation_indices]
            train_chunksize = min(chunksize, len(train_embeddings))
            test_chunksize = min(chunksize, len(test_embeddings))
            num_train_chunks = int(np.ceil(len(train_embeddings) / train_chunksize))
            num_test_chunks = int(np.ceil(len(test_embeddings) / test_chunksize))
            classifications_final = np.array([])
            shortest_distances_final = np.array([])
            for i in range(num_test_chunks):
                logger.info(
                    "Predicting test chunk for threshold calculation {}/{}, eval round {}/{}".format(
                        i + 1, num_test_chunks, fold + 1, 3
                    )
                )
                test_chunk = test_embeddings[
                    i * test_chunksize : min(
                        (i + 1) * test_chunksize, len(test_embeddings)
                    )
                ].to(device)
                shortest_distances_so_far = np.array(
                    [np.inf for _ in range(len(test_chunk))]
                )
                temporary_labels = np.array(["unknown" for _ in range(len(test_chunk))])
                for j in range(num_train_chunks):
                    train_chunk = train_embeddings[
                        j * train_chunksize : min(
                            (j + 1) * train_chunksize, len(train_embeddings)
                        )
                    ].to(device)
                    curr_train_chunk_labels = train_data_labels[
                        j * train_chunksize : min(
                            (j + 1) * train_chunksize, len(train_embeddings)
                        )
                    ]
                    (
                        curr_shortest_distances,
                        curr_shortest_distances_indices,
                    ) = get_min_distances(test_chunk, train_chunk, device, nthreads)

                    is_new_closest = curr_shortest_distances < shortest_distances_so_far
                    shortest_distances_so_far[is_new_closest] = curr_shortest_distances[
                        is_new_closest
                    ]
                    new_labels_indices = curr_shortest_distances_indices[is_new_closest]
                    new_labels = curr_train_chunk_labels[new_labels_indices]
                    temporary_labels[is_new_closest] = new_labels

                classifications_final = np.concatenate(
                    (classifications_final, temporary_labels)
                )
                shortest_distances_final = np.concatenate(
                    (shortest_distances_final, shortest_distances_so_far)
                )
            is_closest_to_correct_label = (
                classifications_final == validation_data_labels
            )
            is_closest_to_correct_label = np.array(
                [0 if x else 1 for x in is_closest_to_correct_label]
            )

            precision, recall, thresholds = precision_recall_curve(
                is_closest_to_correct_label, shortest_distances_final
            )
            f1_scores = np.array(
                [
                    2 * p * r / (p + r) if p + r > 0 else 0
                    for p, r in zip(precision, recall)
                ]
            )
            max_f1_score = np.max(f1_scores)
            max_f1_score_idx = np.argmax(f1_scores)
            distance_threshold = thresholds[max_f1_score_idx]
            logger.info("distance_threshold: " + str(distance_threshold))
            logger.info("max_f1_score: " + str(max_f1_score))
            max_thresholds = np.append(max_thresholds, distance_threshold)
            max_f1_scores = np.append(max_f1_scores, max_f1_score)

    avg_max_f1_score = np.mean(max_f1_scores)
    avg_threshold = np.mean(max_thresholds)
    logger.info("avg_max_f1_score: " + str(avg_max_f1_score))
    logger.info("avg_threshold: " + str(avg_threshold))
    return avg_threshold


def get_min_distances(
    embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, device, nthreads=nthreads
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the minimum distances between the embeddings in embeddings_a and embeddings_b
    :param embeddings_a: embeddings to calculate distances from
    :param embeddings_b: embeddings to calculate distances to
    :return: tuple of the minimum distances and the indices of the embeddings in embeddings_b that are closest to the embeddings in embeddings_a
    """
    if device == "cpu":
        a_b_distances = pairwise_distances(
            embeddings_a.cpu().numpy(), embeddings_b.cpu().numpy(), n_jobs=nthreads
        )
        a_b_distances = torch.tensor(a_b_distances, dtype=torch.float32)
    else:
        a_b_distances = cdist(embeddings_a, embeddings_b)
        #a_b_distances = CosineSimilarity()(embeddings_a, embeddings_b)
        min_distances: torch.Tensor
        min_indices: torch.Tensor
    min_distances, min_indices = torch.min(a_b_distances, dim=1)
    output = min_distances.cpu().numpy(), min_indices.cpu().numpy()
    return output


def _calc_embeddings(sdf, model: VariableNet, device: Any, chunksize=chunksize, nthreads=nthreads):
    chunksize = min(chunksize, len(sdf))
    num_chunks = int(np.ceil(len(sdf) / chunksize))
    embeddings = []
    if device == "cpu":
        torch.set_num_threads(nthreads)
    with torch.no_grad():
        for i in range(num_chunks):
            logger.info(
                "Getting chunk {}/{} for embedding calculation".format(
                    i + 1, num_chunks
                )
            )
            chunk = sdf.matrix[
                i * chunksize : min((i + 1) * chunksize, len(sdf))
            ].todense()
            chunk_tensor = torch.tensor(
                chunk, requires_grad=False, dtype=torch.float32
            ).to(device)
            logger.info("embedding")
            embeddings.append(model.forward_once(chunk_tensor))
            del chunk_tensor

    return torch.cat(embeddings, dim=0)


def get_embeddings(
    sdf: SparseDataFrame,
    model: VariableNet,
    embeddings_path: str,
    device=user_device,
    chunksize=chunksize,
    use_saved_embeddings=True,
    nthreads=nthreads,
) -> torch.Tensor:
    """Calculates the embeddings for the given data
    :param sdf: data to calculate embeddings for
    :param model: model to use for calculating embeddings
    :param embeddings_path: path to load/save the embeddings to
    :param device: device to use for calculating embeddings
    :param chunksize: chunksize to use for data
    :param use_saved_embeddings: whether to use saved embeddings if they exist and save them if they don't
    :return: embeddings for the given data
    """
    if embeddings_path is None or embeddings_path == "":
        use_saved_embeddings = False
    if use_saved_embeddings and os.path.exists(embeddings_path):
        logger.info("Loading embeddings")
        embeddings: torch.Tensor = torch.load(embeddings_path)
        embeddings = embeddings.to(device)
    else:
        logger.info("Calculating embeddings")
        embeddings = _calc_embeddings(sdf, model, device, chunksize, nthreads)
        if use_saved_embeddings:
            logger.info("Saving embeddings")
            torch.save(embeddings, embeddings_path)
    return embeddings

def precalculate_embeddings(
    sdf_path: str,
    model_path: str,
    embeddings_path: str,
    device=user_device,
    chunksize=chunksize,
    nthreads=nthreads,
) -> None:
    """
    Calculates the embeddings for the given data
    :param sdf_path: path to the SparseDataFrame
    :param model_path: path to the pytorch model to use for classification
    :param embeddings_path: path to load/save the embeddings to
    :param device: device to use for calculating embeddings
    :param chunksize: chunksize to use for data
    """
    sdf = pickle.load(open(sdf_path, "rb"))
    embeddings = get_embeddings(
        sdf=sdf,
        model=torch.load(model_path, map_location=device),
        embeddings_path=embeddings_path,
        device=device,
        chunksize=chunksize,
        use_saved_embeddings=False,
        nthreads=nthreads,
    )
    # verify that embeddings are in RAM
    embeddings.cpu()
    torch.save(embeddings, embeddings_path)

def classify(
    sdf_train_path: str,
    sdf_classify_path: str,
    model_path: str,
    train_embeddings_path: str,
    classification_embeddings_path: str,
    output_path: str,
    device=user_device,
    chunksize=chunksize,
    threshold=threshold,
    nthreads=nthreads,
) -> None:
    """
    Generates predictions for the given data.
    Loads the model and data, loads the embeddings if they exist or calculates them if they don't.
    Then uses the embeddings to classify the unseen data by finding the closest embeddings in the data used to train the model.
    :param sdf_train_path: path to the training SparseDataFrame
    :param sdf_classify_path: path to the classification SparseDataFrame
    :param model_path: path to the pytorch model to use for classification
    :param train_embeddings_path: path to load/save the training embeddings to
    :param classification_embeddings_path: path to load/save the classification embeddings to
    :param output_path: path to the file to write the classifications to
    :param device: device to use for classification
    :param chunksize: chunksize to use for data
    :param threshold: threshold to use for classifying a label as unknown - either a float or "bootstrap" to calculate the threshold
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
        torch.set_num_threads(nthreads)
    else:
        device = torch.device(device)

    logger.info("device: " + str(device))
    logger.info("Loading model")
    snn_model: VariableNet = torch.load(model_path, map_location=device)
    snn_model.eval()
    sdf_train: SparseDataFrame
    sdf_classify: SparseDataFrame
    logger.info("Loading dataframes")
    sdf_train, sdf_classify = load_sparse_dataframes(sdf_train_path, sdf_classify_path)
    if threshold == "bootstrap":
        if np.any(sdf_train.labels == "unknown"):
            logger.info("Calculating threshold using unknown labels in training data")
            threshold = calculate_threshold(
                sdf_train=sdf_train,
                model=snn_model,
                train_embeddings_path=train_embeddings_path,
                device=device,
                chunksize=chunksize,
                nthreads=nthreads,
            )
        else:
            logger.warn(
                "No unknown labels in training data. Using known labels of other classes as negative examples"
            )
            threshold = calculate_threshold_no_unknowns(
                sdf_train=sdf_train,
                model=snn_model,
                train_embeddings_path=train_embeddings_path,
                device=device,
                chunksize=chunksize,
                nthreads=nthreads,
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
    train_chunksize = min(chunksize, len(sdf_train))
    classify_chunksize = min(chunksize, len(sdf_classify))
    num_train_chunks = int(np.ceil(len(sdf_train) / train_chunksize))
    num_classify_chunks = int(np.ceil(len(sdf_classify) / classify_chunksize))
    classifications = np.array([])
    train_data_labels = np.array(sdf_train.labels)

    train_embeddings = get_embeddings(
        sdf_train, snn_model, train_embeddings_path, device, chunksize
    )
    classifiy_embeddings = get_embeddings(
        sdf_classify, snn_model, classification_embeddings_path, device, chunksize
    )

    with torch.no_grad():
        for i in range(num_classify_chunks):
            logger.info(
                "Predicting chunk for classification {}/{}".format(
                    i + 1, num_classify_chunks
                )
            )
            classify_chunk = classifiy_embeddings[
                i * classify_chunksize : min(
                    (i + 1) * classify_chunksize, len(sdf_classify)
                )
            ].to(device)
            curr_closest_distances = np.array(
                [np.inf for _ in range(len(classify_chunk))]
            )
            curr_labels = np.array(["unknown" for _ in range(len(classify_chunk))])
            for j in range(num_train_chunks):
                train_chunk = train_embeddings[
                    j * train_chunksize : min((j + 1) * train_chunksize, len(sdf_train))
                ].to(device)
                min_distances, min_indices = get_min_distances(
                    classify_chunk, train_chunk, device, nthreads
                )

                min_indices_labels = train_data_labels[
                    [j * train_chunksize + idx for idx in min_indices]
                ]
                passed_threshold_and_below_curr_closest_distance = np.logical_and(
                    min_distances < threshold, min_distances < curr_closest_distances
                )
                curr_closest_distances[
                    passed_threshold_and_below_curr_closest_distance
                ] = min_distances[passed_threshold_and_below_curr_closest_distance]
                curr_labels[
                    passed_threshold_and_below_curr_closest_distance
                ] = min_indices_labels[passed_threshold_and_below_curr_closest_distance]

            classifications = np.concatenate((classifications, curr_labels))

    logger.info("Saving predictions")
    with open(output_path, "w+") as f:
        for i, label in enumerate(classifications):
            f.write(sdf_classify.index_ids[i] + "\t" + label + "\n")
