"""
A class for working with sparse data for data science projects.
Although pandas dataframes can store sparse data in the form of sparse pandas series, operations on them are still slow.
In the common case of data science projects, they also usually still store labels for each row inside the dataframe.
The SparseDataFrame class is a wrapper around a scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.
It can also store index identifiers in addition to positional indices and column names like a pandas dataframe, but in a separate data structure.
In addition it can store labels for each row, and perform operations on the data based on the annotation of each row and label.
All operations are performed on the underlying matrix, so they are very fast on large and sparse matrices.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from tqdm import tqdm

from famus.logging import logger


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
        if not isinstance(matrix, (csr_matrix, csc_matrix)):
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
        """Save the SparseDataFrame to a file."""
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
        """Returns a new SparseDataFrame that only contains the rows that are True in the mask."""
        logger.debug("Called _keep_indices with mask of length " + str(len(mask)))
        index_ids_to_keep = set(self.index_ids[mask])
        if self.labels:
            labels = {
                index_id: label
                for index_id, label in self.labels.items()
                if index_id in index_ids_to_keep
            }
        else:
            labels = None
        logger.debug("Keeping " + str(len(index_ids_to_keep)) + " rows...")
        return SparseDataFrame(
            matrix=self.matrix[mask, :],
            index_ids=self.index_ids[mask],
            labels=labels,
            column_names=self.column_names,
            dtype=self.dtype,
        )

    def _keep_columns(self, mask: Iterable[bool]) -> SparseDataFrame:
        """Returns a new SparseDataFrame that only contains the columns that are True in the mask."""
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
        """Returns the number of unique labels in the dataframe."""
        if not self.labels:
            return 0
        return len(self.unique_labels())

    def get_debug_dataframe(self, max_rows=1000) -> SparseDataFrame:
        """Returns a new SparseDataFrame with the first max_rows rows of the original dataframe."""
        n_rows = min(max_rows, len(self))
        return self.select_by_index_ids(self.index_ids[:n_rows])

    def matrix_iat(self, indices: Iterable) -> np.ndarray:
        """Returns the values in the matrix at the given indices."""
        return self.matrix[indices, :]

    def unique_labels(self) -> list:
        """Returns the unique labels in the dataframe."""
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
        """Returns a new SparseDataFrame that only contains the rows with the given index ids."""
        logger.debug(
            "Called select_by_index_ids with " + str(len(index_ids)) + " index ids"
        )
        if len(index_ids) == 0:
            raise ValueError("index_ids must contain at least one index id")
        if not all(np.isin(index_ids, self.index_ids)):
            raise IndexError("Some index_ids not found in dataframe")
        mask = np.isin(self.index_ids, index_ids)
        return self._keep_indices(mask)

    def select_by_labels(self, labels: Iterable) -> SparseDataFrame:
        """Returns a new SparseDataFrame that only contains the rows with the given labels."""
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
        """Returns a dict mapping each label to the ids of the dataframe that have that label."""
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
        """Returns a dict mapping each label to the indices of the dataframe that have that label."""
        index_id_to_index = {
            index_id: idx for idx, index_id in enumerate(self.index_ids)
        }
        labels = self.get_label_to_index_ids()
        return {
            label: np.array([index_id_to_index[index_id] for index_id in index_ids])
            for label, index_ids in labels.items()
        }

    def merge(self, other: SparseDataFrame) -> SparseDataFrame:
        """
        Merges two SparseDataFrames with the same columns but different rows.
        Ignores index id duplicates in other.
        If only one dataframe has labels, raises a ValueError.
        """
        if not np.array_equal(self.column_names, other.column_names):
            raise ValueError("Dataframes must have the same columns to be merged")
        if (self.labels is None and other.labels is not None) or (
            self.labels is not None and other.labels is None
        ):
            raise ValueError("Both dataframes must have labels or neither")

        combined_index_ids = list(self.index_ids)
        rows, cols, values = (
            self.matrix.tocoo().row.tolist(),
            self.matrix.tocoo().col.tolist(),
            self.matrix.tocoo().data.tolist(),
        )
        existing_index_ids = set(self.index_ids)
        for i, index_id in enumerate(other.index_ids):
            if index_id in existing_index_ids:
                continue
            target_row_idx = len(
                combined_index_ids
            )  # Capture position before appending
            combined_index_ids.append(index_id)
            other_row = other.matrix.getrow(i).tocoo()
            rows.extend((other_row.row + target_row_idx).tolist())
            cols.extend(other_row.col.tolist())
            values.extend(other_row.data.tolist())
        combined_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(combined_index_ids), self.matrix.shape[1]),
        )
        if self.labels and other.labels:
            combined_labels = {**self.labels, **other.labels}
        else:
            combined_labels = None
        return SparseDataFrame(
            combined_matrix,
            combined_index_ids,
            combined_labels,
            self.column_names,
            self.dtype,
        )


def load(path: str) -> SparseDataFrame:
    """Load a SparseDataFrame from a file that was saved using the save function."""
    with open(path, "rb") as f:
        data = json.loads(f.read())
    matrix = csr_matrix((data["values"], (data["row_indices"], data["col_indices"])))
    try:
        dtype = np.dtype(data["dtype"].split(".")[1].removesuffix("'>"))
    except Exception as e:
        dtype = np.dtype(data["dtype"])
    return SparseDataFrame(
        matrix,
        data["index_ids"],
        {k: np.array(v) for k, v in data["labels"].items()} if data["labels"] else None,
        data["column_names"],
        dtype=dtype,
    )


def from_sparse_dict(
    data: dict, labels: dict | None = None, dtype=np.float32
) -> SparseDataFrame:
    """
    Create a SparseDataFrame from a dict of (index_id, feature_name) -> value
    and a dict of index_id -> label.
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
    return SparseDataFrame(sm, sorted_index_ids, labels, sorted_columns, dtype=dtype)


class Normalizer:
    """
    Normalizes columns of a SparseDataFrame by dividing each column by the mean
    value of that column among rows with the "correct" label for that column.

    Uses vectorized diagonal matrix multiplication for efficient normalization.
    """

    def __init__(self):
        self.column_to_mean_correct_bitscore: dict[str, float] = {}
        self._pat = re.compile(r"\.sub_cluster\.cluster\.\d+")

    def fit(self, sdf: SparseDataFrame) -> Normalizer:
        """
        Compute mean bitscores for each column based on rows with the correct label.

        Args:
            sdf: SparseDataFrame with labels

        Returns:
            self (for method chaining)
        """

        if sdf.labels is None:
            raise ValueError("SparseDataFrame must have labels for fitting")

        index_id_to_index = {
            index_id: idx for idx, index_id in enumerate(sdf.index_ids)
        }
        label_to_indices: dict[str, list[int]] = {}

        for index_id, curr_labels in tqdm(
            sdf.labels.items(), desc="Mapping labels to indices"
        ):
            if curr_labels[0] == "unknown":
                continue
            idx = index_id_to_index[index_id]
            for curr_label in curr_labels:
                if curr_label not in label_to_indices:
                    label_to_indices[curr_label] = []
                label_to_indices[curr_label].append(idx)

        logger.debug(f"found {len(label_to_indices)} unique labels")
        # for efficient column access during mean computation

        matrix_csc = (
            sdf.matrix.tocsc() if not isinstance(sdf.matrix, csc_matrix) else sdf.matrix
        )
        matrix_csc = matrix_csc.astype(np.float32)
        column_to_mean = {}
        for col_idx, colname in enumerate(
            tqdm(sdf.column_names, desc="Computing mean bitscores")
        ):
            correct_label = self._pat.sub("", colname)
            relevant_indices = label_to_indices.get(correct_label, [])
            if relevant_indices:
                bit_scores = matrix_csc[relevant_indices, col_idx].toarray().flatten()
                column_to_mean[colname] = float(np.mean(bit_scores))
            else:
                column_to_mean[colname] = 0.0

        self.column_to_mean_correct_bitscore = column_to_mean
        return self

    def transform(self, sdf: SparseDataFrame) -> SparseDataFrame:
        if not self.column_to_mean_correct_bitscore:
            raise ValueError("Normalizer has not been fitted yet.")

        missing_cols = set(sdf.column_names) - set(
            self.column_to_mean_correct_bitscore.keys()
        )
        if missing_cols:
            raise ValueError(
                f"Dataframe contains {len(missing_cols)} columns not seen during fitting."
            )

        matrix_float = sdf.matrix.astype(np.float32)  # Don't mutate input

        scale_factors = np.array(
            [
                1.0 / self.column_to_mean_correct_bitscore[col]
                if self.column_to_mean_correct_bitscore[col] != 0
                else 1.0
                for col in sdf.column_names
            ],
            dtype=np.float32,  # Hardcode float32, not sdf.dtype
        )

        scale_matrix = diags(scale_factors, offsets=0, format="csr")
        normalized_matrix = matrix_float @ scale_matrix

        return SparseDataFrame(
            normalized_matrix,
            sdf.index_ids,
            sdf.labels,
            sdf.column_names,
            dtype=np.float32,
        )

    def fit_transform(self, sdf: SparseDataFrame) -> SparseDataFrame:
        """
        Fit the normalizer and transform the data in one call.

        Args:
            sdf: SparseDataFrame with labels

        Returns:
            New normalized SparseDataFrame
        """
        return self.fit(sdf).transform(sdf)

    def save(self, path: str) -> None:
        """Save the Normalizer to a file."""
        if not self.column_to_mean_correct_bitscore:
            raise ValueError("Normalizer has not been fitted yet.")
        with open(path, "w") as f:
            json.dump(self.column_to_mean_correct_bitscore, f)

    @staticmethod
    def load(path: str) -> Normalizer:
        """
        Load the Normalizer from a file.

        Returns:
            self (for method chaining)
        """
        with open(path, "r") as f:
            column_to_mean_correct_bitscore = json.load(f)
        norm = Normalizer()
        norm.column_to_mean_correct_bitscore = column_to_mean_correct_bitscore
        return norm
