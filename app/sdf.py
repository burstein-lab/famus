"""
A class for working with sparse data for data science projects.
Although pandas dataframes can store sparse data in the form of sparse pandas series, operations on them are still slow.
In the common case of data science projects, they also usually still store labels for each row inside the dataframe.
The SparseDataFrame class is a wrapper around a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.
It can also store index identifiers in addition to positional indices and column names like a pandas dataframe, but in a seperate data structure.
In addition it can strore labels for each row, and perform operations on the data based on the annotation of each row and label.
All operations are performed on the underlying matrix, so they are very fast on large and sparse matrices.
"""

from __future__ import annotations
from dataclasses import dataclass

from typing import Iterable
import json

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from tqdm import tqdm


@dataclass
class SparseDataFrame(object):
    """
    A class for storing sparse data with many rows and columns in a dataframe-like format
    and performing fast operations on it.

    """

    def __init__(
        self,
        matrix: csr_matrix | csc_matrix,
        index_ids: Iterable[str],
        labels: dict | None = None,
        column_names: Iterable | None = None,
        dtype=np.float32,
    ) -> None:
        """
        SparseDataFrame constructor.
        """
        if not isinstance(matrix, csr_matrix) and not isinstance(matrix, csc_matrix):
            raise TypeError(
                "matrix must be a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix"
            )
        if not isinstance(index_ids, Iterable):
            raise TypeError(
                "index_ids must be an iterable object such as a list, tuple or a numpy array"
            )
        if labels is not None and not isinstance(labels, dict):
            raise TypeError("labels must be a dict")
        if column_names is not None and not isinstance(column_names, Iterable):
            raise TypeError(
                "column_names must be an iterable object such as a list, tuple or a numpy array"
            )

        if not len(index_ids) == len(set(index_ids)):
            raise ValueError("index_ids must be unique")
        if not matrix.shape[0] == len(index_ids):
            raise ValueError(
                "The number of rows in the matrix must be equal to the length of index_ids"
            )

        if column_names is not None and not len(column_names) == matrix.shape[1]:
            raise ValueError(
                "The number of columns in the matrix must be equal to the length of column_names"
            )
        if labels:
            for label_list in labels.values():
                if not isinstance(label_list, Iterable):
                    raise TypeError("labels values must be an iterable object")
            labeled_but_not_in_index_ids = set(labels) - set(index_ids)
            if len(labeled_but_not_in_index_ids) > 0:
                print(
                    f"Some index ids appear in the labels dict but not in the data: {labeled_but_not_in_index_ids}"
                )
                labels = {
                    k: v
                    for k, v in labels.items()
                    if k not in labeled_but_not_in_index_ids
                }

            index_ids_without_labels = set(index_ids) - set(labels)
            if len(index_ids_without_labels) > 0:
                raise ValueError(
                    f"labels dict must contain all index ids. ids without labels: {index_ids_without_labels}"
                )
        matrix = matrix.astype(dtype)
        self.dtype = dtype
        self.matrix = matrix
        self.index_ids = np.array(index_ids)
        if labels:
            self.labels = {k: np.array(v) for k, v in labels.items()}
        else:
            self.labels = None
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

    def save(self, path: str) -> None:
        """
        Save the SparseDataFrame to a file.
        """
        coo = self.matrix.tocoo()
        data = {
            "row_indices": coo.row.tolist(),
            "col_indices": coo.col.tolist(),
            "values": coo.data.tolist(),
            "index_ids": self.index_ids.tolist(),
            "labels": {k: v.tolist() for k, v in self.labels.items()}
            if self.labels
            else None,
            "column_names": self.column_names.tolist(),
            "dtype": str(self.dtype),
        }
        with open(path, "w") as f:
            f.write(json.dumps(data))

    def _keep_indices(self, mask: Iterable[bool]) -> SparseDataFrame:
        """
        Returns a new SparseDataFrame that only contains the rows that are True in the mask.
        """
        index_ids_to_keep = set(self.index_ids[mask])
        if self.labels:
            labels = {
                index_id: label
                for index_id, label in self.labels.items()
                if index_id in index_ids_to_keep
            }
        else:
            labels = None
        return SparseDataFrame(
            matrix=self.matrix[mask, :],
            index_ids=self.index_ids[mask],
            labels=labels,
            column_names=self.column_names,
            dtype=self.dtype,
        )

    def _keep_columns(self, mask: Iterable[bool]) -> SparseDataFrame:
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

    def get_label(self, label) -> SparseDataFrame:
        """
        Returns a new SparseDataFrame that only contains the rows with the given label.
        Raises IndexError if no rows with the given label are found.
        """
        labels = self.get_label_to_index_ids()
        if label not in labels:
            raise IndexError(f"Label {label} not found in labels")
        return self.select_by_index_ids(labels[label])

    def get_num_labels(self) -> int:
        """
        Returns the number of unique labels in the dataframe.
        """
        if not self.labels:
            return 0
        return len(self.unique_labels())

    def get_debug_dataframe(self, max_rows=1000) -> SparseDataFrame:
        """
        Returns a new SparseDataFrame with the first max_rows rows of the original dataframe.
        """
        n_rows = min(max_rows, len(self))
        return self.select_by_index_ids(self.index_ids[:n_rows])

    def matrix_iat(self, indices: Iterable) -> np.ndarray:
        """
        Returns the values in the matrix at the given indices.
        """
        return self.matrix[indices, :]

    def unique_labels(self) -> list:
        """
        Returns the unique labels in the dataframe.
        """
        labels = set()
        for label_list in self.labels.values():
            labels.update(label_list)
        return list(labels)

    def remove_label(self, label: str) -> SparseDataFrame:
        """
        Removes all rows with the given label from the dataframe.
        Raises IndexError if no rows with the given label are found.
        """
        labels = self.get_label_to_index_ids()
        if label not in labels:
            raise IndexError(f"Label {label} not found in labels")
        index_ids_to_remove = set(labels[label])
        mask = [idx not in index_ids_to_remove for idx in self.index_ids]
        return self._keep_indices(mask)

    def select_by_index_ids(self, index_ids: Iterable) -> SparseDataFrame:
        """
        Returns a new SparseDataFrame that only contains the rows with the given index ids.
        """
        if len(index_ids) == 0:
            raise ValueError("index_ids must contain at least one index id")
        if not all([idx in set(self.index_ids) for idx in index_ids]):
            raise ValueError(
                "index_ids must contain only index ids that are in the dataframe"
            )
        mask = [idx in index_ids for idx in self.index_ids]
        return self._keep_indices(mask)

    def select_by_labels(self, labels: Iterable) -> SparseDataFrame:
        """
        Returns a new SparseDataFrame that only contains the rows with the given labels.
        """

        new_labels = {}
        labels = np.array(labels)
        for index_id, label_list in self.labels.items():
            remaining = np.intersect1d(label_list, labels)
            if len(remaining) > 0:
                new_labels[index_id] = remaining
        index_ids = set(new_labels.keys())
        mask = [idx in index_ids for idx in self.index_ids]
        return SparseDataFrame(
            self.matrix[mask, :],
            self.index_ids[mask],
            new_labels,
            self.column_names,
            self.dtype,
        )

    def get_label_to_index_ids(self) -> dict:
        """
        Returns a dict mapping each label to the ids of the dataframe that have that label.
        """
        if not self.labels:
            raise ValueError("No labels in dataframe")
        output = {}
        for index_id, labels in self.labels.items():
            for label in labels:
                if label not in output:
                    output[label] = []
                output[label].append(index_id)
        return {k: np.array(v) for k, v in output.items()}

    def get_label_to_indices(self) -> dict:
        """
        Returns a dict mapping each label to the indices of the dataframe that have that label.
        """
        index_id_to_index = {
            index_id: idx for idx, index_id in enumerate(self.index_ids)
        }
        labels = self.get_label_to_index_ids()
        return {
            label: np.array([index_id_to_index[index_id] for index_id in index_ids])
            for label, index_ids in labels.items()
        }


def load(path: str) -> SparseDataFrame:
    """
    Load a SparseDataFrame from a file that was saved using the save function.
    """
    with open(path, "rb") as f:
        data = json.loads(f.read())
    matrix = csr_matrix((data["values"], (data["row_indices"], data["col_indices"])))
    return SparseDataFrame(
        matrix,
        data["index_ids"],
        {k: np.array(v) for k, v in data["labels"].items()} if data["labels"] else None,
        data["column_names"],
        dtype=np.dtype(data["dtype"].split(".")[1].removesuffix("'>")),
    )


def from_sparse_dict(
    data: dict, lables: dict | None = None, dtype=np.float32
) -> SparseDataFrame:
    """
    Create a SparseDataFrame from a dict of (index_id, feature_name) -> value and a dict of index_id -> label.
    """
    index_ids = list(set([x[0] for x in list(data.keys())]))
    features_to_indices = {}
    feature_index_counter = 0
    index_ids_to_indices = {index_id: i for i, index_id in enumerate(index_ids)}
    rows, cols, values = [], [], []
    for id_and_feature, value in tqdm(
        data.items(), total=len(data), desc="reading sparse dict"
    ):
        feature = id_and_feature[1]
        if feature not in features_to_indices:
            features_to_indices[feature] = feature_index_counter
            feature_index_counter += 1
        rows.append(index_ids_to_indices[id_and_feature[0]])
        cols.append(features_to_indices[feature])
        values.append(value)
    sm = csr_matrix((values, (rows, cols)))
    sorted_index_ids = [
        index_id
        for index_id in sorted(index_ids_to_indices, key=index_ids_to_indices.get)
    ]
    sorted_columns = [
        feature for feature in sorted(features_to_indices, key=features_to_indices.get)
    ]
    return SparseDataFrame(sm, sorted_index_ids, lables, sorted_columns, dtype=dtype)
