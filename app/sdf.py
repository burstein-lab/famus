"""
A class for working with sparse data for data science projects.
Although pandas dataframes can store sparse data in the form of sparse pandas series, operations on them are still slow.
In the common case of data science projects, they also usually still store labels for each row inside the dataframe.
The SparseDataFrame class is a wrapper around a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.
It can also store index identifiers in addition to positional indices and column names like a pandas dataframe, but in a seperate data structure. 
In addition it can strore labels for each row, and perform operations on the data based on the annotation of each row and label.
All operations are performed on the underlying matrix, so they are very fast on large and sparse matrices.
"""
import random
from dataclasses import dataclass
from math import ceil
from typing import Iterable, Tuple

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm


def from_dense_array(
    data: np.ndarray,
    index_ids: list | set | np.ndarray | None = None,
    labels: list | set | np.ndarray | None = None,
    column_names: list | set | np.ndarray | None = None,
    dtype=np.float32,
    matrix_type=csc_matrix,
):
    """
    Create a SparseDataFrame instance from a dense numpy array.
    :param data: the data to store in the dataframe.
    :param index_ids: the index ids (not necessarily numeric) of the rows in the dataframe.
    :param labels: Optional. the labels of the rows in the dataframe.
    :param column_names: Optional. the names of the columns in the dataframe.
    :param dtype: the dtype of the data. Defaults to np.float32.
    :param matrix_type: the type of the underlying matrix. Defaults to scipy.sparse.csc_matrix.
    """
    assert isinstance(data, np.ndarray), "data must be a a numpy array"
    assert isinstance(
        index_ids, Iterable
    ), "index_ids must be an iterable object such as a list, tuple or a numpy array"
    assert isinstance(
        labels, Iterable
    ), "labels must be an iterable object such as a list, tuple or a numpy array"
    assert (
        len(index_ids) == data.shape[0]
    ), "index_ids must have the same length as the number of rows in the matrix"
    assert (
        len(labels) == data.shape[0]
    ), "labels must have the same length as the number of rows in the matrix"
    assert len(index_ids) == len(set(index_ids)), "index_ids must be unique"
    assert matrix_type in [
        csr_matrix,
        csc_matrix,
    ], "matrix_type must be either scipy.sparse.csr_matrix or scipy.sparse.csc_matrix"
    matrix = matrix_type(data.astype(dtype))
    return SparseDataFrame(matrix, index_ids, labels, column_names, dtype=dtype)


def from_sparse_dict(data: dict, lables: dict | None = None, dtype=np.float32):
    """
    Create a SparseDataFrame from a dict of (index_id, feature_name) -> value and a dict of index_id -> label.
    """
    print("Getting unique index ids")
    index_ids = list(
        set([x[0] for x in list(data.keys())])
    )  # unordered list of unique ids
    features_to_indices = {}  # where we will store the mapping from feature name to index of the feature in the matrix
    feature_index_counter = 0  # used to assign indices to features
    index_ids_to_indices = {
        index_id: i for i, index_id in enumerate(index_ids)
    }  # maps index ids to indices in the matrix
    rows, cols, values = [], [], []
    for id_and_feature, value in tqdm(
        data.items(), total=len(data), desc="reading sparse dict"
    ):  # input is map of (index_id, feature_name) -> value, goal is to unpack to list of index_id indices, list of feature indices and list of values
        feature = id_and_feature[1]
        if feature not in features_to_indices:
            features_to_indices[
                feature
            ] = feature_index_counter  # assign index to new feature
            feature_index_counter += 1
        # get indices of id and feature in the matrix, and append them to the lists
        rows.append(index_ids_to_indices[id_and_feature[0]])
        cols.append(features_to_indices[feature])
        values.append(value)
    print("Creating sparse matrix")
    sm = csr_matrix((values, (rows, cols)))
    # get sorted index_ids by their index
    print("Sorting index ids")
    sorted_index_ids = [
        index_id
        for index_id in sorted(index_ids_to_indices, key=index_ids_to_indices.get)
    ]
    # get sorted labels by their index
    if lables is not None:
        sorted_labels = [lables[index_id] for index_id in sorted_index_ids]
    else:
        sorted_labels = None
    # get sorted features by their index
    print("Sorting features")
    sorted_columns = [
        feature for feature in sorted(features_to_indices, key=features_to_indices.get)
    ]
    return SparseDataFrame(
        sm, sorted_index_ids, sorted_labels, sorted_columns, dtype=dtype
    )


@dataclass
class SparseDataFrame(object):
    """
    A class for storing sparse data with many rows and columns in a dataframe-like format
    and performing fast operations on it.
    """

    def __init__(
        self,
        matrix: csr_matrix | csc_matrix,
        index_ids: Iterable,
        labels: Iterable | None = None,
        column_names: Iterable | None = None,
        dtype=np.float32,
    ):
        assert (
            type(matrix) == csr_matrix or type(matrix) == csc_matrix
        ), "matrix must be a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix"
        assert isinstance(
            index_ids, Iterable
        ), "index_ids must be an iterable object such as a list, tuple or a numpy array"
        assert (
            len(index_ids) == matrix.shape[0]
        ), "index_ids must have the same length as the number of rows in the matrix"
        if labels is not None:
            assert (
                len(labels) == matrix.shape[0]
            ), "labels must have the same length as the number of rows in the matrix"
            assert isinstance(
                labels, Iterable
            ), "labels must be None or an iterable object such as a list, tuple or a numpy array"
        assert len(index_ids) == len(set(index_ids)), "index_ids must be unique"
        matrix = matrix.astype(dtype)
        self.dtype = dtype
        self.matrix = matrix
        self.index_ids = np.array(index_ids)
        self.labels = np.array(labels) if labels is not None else None
        self.column_names = np.array(column_names) if column_names is not None else None

    def __len__(self):
        return len(self.index_ids)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.select_by_index_ids([key])
        elif isinstance(key, Iterable):
            return self.select_by_index_ids(key)
        else:
            raise TypeError("Invalid argument type")

    def _keep_indices(self, mask: Iterable):
        """
        Returns a new SparseDataFrame that only contains the rows that are True in the mask.
        """
        return SparseDataFrame(
            self.matrix[mask, :],
            self.index_ids[mask],
            self.labels[mask],
            self.column_names,
            self.dtype,
        )

    def _keep_columns(self, mask: Iterable):
        """
        Returns a new SparseDataFrame that only contains the columns that are True in the mask.
        """
        return SparseDataFrame(
            self.matrix[:, mask],
            self.index_ids,
            self.labels,
            self.column_names[mask],
            self.dtype,
        )

    def get_label(self, label):
        """
        Returns a new SparseDataFrame that only contains the rows with the given label.
        Raises an AssertionError if no rows with the given label are found.
        """
        mask = self.labels == label
        assert any(mask), "No rows with label {} found".format(label)
        return self._keep_indices(mask)

    def get_debug_dataframe(self, max_rows=1000):
        """
        Returns a new SparseDataFrame with the first max_rows rows of the original dataframe.
        """
        n_rows = min(max_rows, len(self))
        return self.select_by_index_ids(self.index_ids[:n_rows])

    def matrix_iat(self, indices: Iterable):
        """
        Returns the values in the matrix at the given indices.
        """
        return self.matrix[indices, :]

    def unique_labels(self) -> set:
        """
        Returns the unique labels in the dataframe.
        """
        return set(self.labels)

    def remove_label(self, label: str) -> "SparseDataFrame":
        """
        Removes all rows with the given label from the dataframe.
        Raises an AssertionError if no rows with the given label are found.
        """
        mask = self.labels != label
        assert any(mask), "No rows with label {} found".format(label)
        return self._keep_indices(mask)

    def select_by_index_ids(self, index_ids: Iterable) -> "SparseDataFrame":
        """
        Returns a new SparseDataFrame that only contains the rows with the given index ids.
        """
        pool = set(index_ids)
        mask = [idx in pool for idx in self.index_ids]
        assert any(mask), "No rows with index ids {} found".format(index_ids)
        return self._keep_indices(mask)

    def select_by_labels(self, labels: Iterable) -> "SparseDataFrame":
        """
        Returns a new SparseDataFrame that only contains the rows with the given labels.
        """
        pool = set(labels)
        mask = [label in pool for label in self.labels]
        assert any(mask), "No rows with labels {} found".format(labels)
        return SparseDataFrame(
            self.matrix[mask, :], self.index_ids[mask], self.labels[mask]
        )

    def drop_singleton_labels(self) -> None:
        """
        Drops all labels (and their samples) that only occur once in the dataframe.
        """
        counts: dict = self.get_label_counts()
        mask = [counts[label] > 1 for label in self.labels]
        self.matrix = self.matrix[mask, :]
        self.labels = self.labels[mask]
        self.index_ids = self.index_ids[mask]

    def drop_labels_below_threshold(self, threshold: int) -> None:
        """
        Drops all labels (and their samples) that only occur once in the dataframe.
        """
        counts: dict = self.get_label_counts()
        mask = [counts[label] >= threshold for label in self.labels]
        self.matrix = self.matrix[mask, :]
        self.labels = self.labels[mask]
        self.index_ids = self.index_ids[mask]

    def get_label_counts(self) -> dict:
        """
        Returns a dict of counts for each label.
        """
        result = {label: 0 for label in self.unique_labels()}
        for label in self.labels:
            result[label] += 1
        return result

    def get_label_to_index_ids(self) -> dict:
        """
        Returns a dict mapping each label to the ids of the dataframe that have that label.
        """
        result = {label: [] for label in self.unique_labels()}
        for idx_id, label in zip(self.index_ids, self.labels):
            result[label].append(idx_id)
        return result

    def get_label_to_indices(self) -> dict:
        """
        Returns a dict mapping each label to the indices of the dataframe that have that label.
        """
        result = {label: [] for label in self.unique_labels()}
        for i, label in zip(range(len(self.labels)), self.labels):
            result[label].append(i)
        return result

    def get_stratified_train_test_split(self, test_size: float) -> tuple:
        """
        Returns a stratified train-test split of the dataframe.
        :param test_size: the fraction of the dataframe to use for testing.
        """
        assert 0 < test_size < 1
        counts: dict = self.get_label_counts()
        test_counts = {
            label: ceil(count * test_size) for label, count in counts.items()
        }
        train_counts = {
            label: count - test_counts[label] for label, count in counts.items()
        }
        assert all(
            [count > 0 for count in test_counts.values()]
        ), "Not enough data for split. call drop_singleton_labels before splitting."
        assert all(
            [count > 0 for count in train_counts.values()]
        ), "Not enough data for split. call drop_singleton_labels before splitting."
        label_to_index_ids = self.get_label_to_index_ids()
        test_split_ids = []
        for label, count in test_counts.items():
            test_split_ids += random.sample(label_to_index_ids[label], count)
        train_split_ids = set(self.index_ids) - set(test_split_ids)
        test_split_ids = set(test_split_ids)
        # rearrange the index ids so that they are in the same order as the original dataframe, to not scramble the data
        train_split_ids = [idx for idx in self.index_ids if idx in train_split_ids]
        test_split_ids = [idx for idx in self.index_ids if idx in test_split_ids]
        return self.select_by_index_ids(train_split_ids), self.select_by_index_ids(
            test_split_ids
        )

    def concat(self, other: "SparseDataFrame") -> "SparseDataFrame":
        """
        Concatenates two SparseDataFrames vertically.
        """
        assert isinstance(other, SparseDataFrame), "other must be a SparseDataFrame"
        assert (
            self.matrix.shape[1] == other.matrix.shape[1]
        ), "other must have the same number of columns as self"
        assert all(
            self.column_names == other.column_names
        ), "other must have the same column names as self"
        assert all(self.dtype == other.dtype), "other must have the same dtype as self"
        assert set(self.index_ids).isdisjoint(
            set(other.index_ids)
        ), "other must not contain any index ids that are already in self"

        matrix = csr_matrix(np.vstack((self.matrix, other.matrix)))
        index_ids = np.hstack((self.index_ids, other.index_ids))
        labels = np.hstack((self.labels, other.labels))
        return SparseDataFrame(matrix, index_ids, labels, self.column_names)

    def sample_n_rows(self, n: int) -> "SparseDataFrame":
        """
        Returns a new SparseDataFrame with n random rows from the original dataframe.
        """
        selected_indices = np.random.choice(self.index_ids, size=n, replace=False)
        return self.select_by_index_ids(selected_indices)

    def negative_sampling(self, label: str, n_samples: int) -> "SparseDataFrame":
        """
        samples n rows with a label that is different from the given label.
        """
        assert (
            label in self.unique_labels()
        ), "label must be one of the labels in the dataframe"
        pool = [
            idx
            for idx, sdf_label in zip(self.index_ids, self.labels)
            if sdf_label != label
        ]
        chosen_ids = random.sample(pool, n_samples)
        return self.select_by_index_ids(chosen_ids)

    def shuffle(self):
        """
        Returns a new SparseDataFrame with the rows shuffled.
        """
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        return self.select_by_index_ids(indices)

    def sample_n_random_labels(self, n: int) -> "SparseDataFrame":
        """
        Returns a new SparseDataFrame with n random labels from the original dataframe.
        """
        label_pool = set(random.sample(self.unique_labels(), n))
        mask = [label in label_pool for label in self.labels]
        return SparseDataFrame(
            self.matrix[mask, :],
            self.index_ids[mask],
            self.labels[mask],
            self.column_names,
        )

    def drop_low_frequency_labels(self, n: int) -> "SparseDataFrame":
        """
        Drops all labels that occur less than n times in the dataframe.
        """
        counts = self.get_label_counts()
        mask = [counts[label] >= n for label in self.labels]
        return self._keep_rows(mask)

    def drop_zero_columns(self) -> "SparseDataFrame":
        """
        Drops all columns that are only zeros.
        """
        csc = self.matrix.tocsc()
        mask = [
            True if ri > le else False
            for le, ri in zip(csc.indptr[:-1], csc.indptr[1:])
        ]  # don't change "true if ri > le else false" to "ri > le" - for some reason this causes data loss
        return self._keep_columns(mask)

    def random_sample_frac(self, fraction: float) -> "SparseDataFrame":
        """
        Returns a new SparseDataFrame with a random sample of the rows from the original dataframe.
        """
        assert 0 < fraction <= 1, "fraction must be above 0 and up to 1. got {}".format(
            fraction
        )
        selected_indices = np.random.choice(
            self.index_ids, size=int(len(self.index_ids) * fraction), replace=False
        )
        return self.select_by_index_ids(selected_indices)

    def toy_train_test_split(
        self, test_size: float, fraction=0.1
    ) -> Tuple["SparseDataFrame", "SparseDataFrame"]:
        """
        Returns a stratified train-test split of a random sample of the dataframe.
        Drops low frequency labels before splitting - may not contain every label in the original dataframe.
        """
        toy_sdf = self.random_sample_frac(fraction)
        toy_sdf.drop_singleton_labels()
        toy_train, toy_test = toy_sdf.get_stratified_train_test_split(test_size)
        return toy_train, toy_test
