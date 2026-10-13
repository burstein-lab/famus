"""Tests for the train-metadata sidecar: a small `<table>.meta` file next to a
training SparseDataFrame holding only what classification needs (column names
and per-row labels), so classification can skip parsing the full table.

These tests never set mtimes by hand: on GPFS a utime issued right after a
write can be silently dropped, which made such tests flaky."""

import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from famus import sdf as sdf_module
from famus.sdf import SparseDataFrame

LABELS = {"t1": ["famA"], "t2": ["famA", "famB"], "t3": ["unknown"]}
EXPECTED_COLUMNS = ["p3", "p1", "p2"]
EXPECTED_ROW_LABELS = ["famA", "famA;famB", "unknown"]
DECOY_LABELS = dict(LABELS, t1=["famZ"])


def make_train_sdf(labels=LABELS):
    matrix = csr_matrix(np.array([[5, 0, 0], [0, 7, 1], [0, 0, 9]]))
    return SparseDataFrame(
        matrix, ["t1", "t2", "t3"], labels, ["p3", "p1", "p2"], dtype=np.int64
    )


def make_unreadable(path):
    """chmod 000 keeps the file's size and mtime (only ctime changes), so a
    reader can still succeed only by trusting the sidecar."""
    os.chmod(path, 0)


def write_decoy_sidecar(table_path, size_delta=0, mtime_delta=0):
    """A sidecar holding DECOY_LABELS, stamped with the table's stat shifted by the
    given deltas: trusted iff both deltas are 0."""
    st = os.stat(table_path)
    stamp = SimpleNamespace(
        st_size=st.st_size + size_delta, st_mtime_ns=st.st_mtime_ns + mtime_delta
    )
    sdf_module.save_train_metadata(make_train_sdf(DECOY_LABELS), table_path, stamp)


@pytest.fixture
def json_table(tmp_path):
    path = str(tmp_path / "sdf_train.json")
    make_train_sdf().save(path)
    return path


@pytest.fixture
def pickle_table(tmp_path):
    path = str(tmp_path / "sdf_train.pkl")
    with open(path, "wb") as f:
        pickle.dump(make_train_sdf(), f)
    return path


def test_metadata_from_json_table(json_table):
    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.column_names) == EXPECTED_COLUMNS
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS


def test_metadata_from_pickle_table(pickle_table):
    meta = sdf_module.load_train_metadata(pickle_table, load_from_pickle=True)
    assert list(meta.column_names) == EXPECTED_COLUMNS
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS


def test_later_loads_read_sidecar_instead_of_table(json_table):
    sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    make_unreadable(json_table)
    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.column_names) == EXPECTED_COLUMNS
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS


def test_rewritten_table_rebuilds_sidecar(json_table):
    sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    old = os.stat(json_table)
    make_train_sdf(dict(LABELS, t1=["famC"])).save(json_table)  # e.g. retraining
    assert os.stat(json_table).st_mtime_ns != old.st_mtime_ns

    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.row_labels) == ["famC", "famA;famB", "unknown"]


def test_sidecar_ignored_when_recorded_size_differs(json_table):
    write_decoy_sidecar(json_table, size_delta=1)
    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS


def test_sidecar_ignored_when_recorded_mtime_differs(json_table):
    write_decoy_sidecar(json_table, mtime_delta=1)
    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS


def test_rebuilds_corrupt_sidecar(json_table):
    with open(json_table + ".meta", "w") as f:
        f.write("{not json")
    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS

    make_unreadable(json_table)  # the rebuilt sidecar must now stand alone
    meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS


def test_training_writes_sidecar_with_the_table(tmp_path):
    from famus.data_preprocessing import hmmsearch_results_to_train_sdf

    ground_truth = {"s1": "famA", "s2": "famA;famB", "s3": "unknown"}
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for profile in ["pA", "pB"]:
        (profiles_dir / f"{profile}.hmm").write_text("")
    (tmp_path / "split.tsv").write_text("header\n")
    (tmp_path / "full.tsv").write_text(
        "header\n"
        "s1\t-\tpA.fasta\t-\t-\t-\t-\t-\t10.0\t-\n"
        "s2\t-\tpB.fasta\t-\t-\t-\t-\t-\t20.0\t-\n"
    )
    (tmp_path / "seqs.fasta").write_text(">s1\nMK\n>s2\nMK\n>s3\nMK\n")
    with open(tmp_path / "md.pkl", "wb") as f:
        pickle.dump({"pA": {}, "pB": {}}, f)
    with open(tmp_path / "gt.pkl", "wb") as f:
        pickle.dump(ground_truth, f)
    table = str(tmp_path / "sdf_train.pkl")

    hmmsearch_results_to_train_sdf(
        input_split_hmmsearch_results_path=str(tmp_path / "split.tsv"),
        input_split_subcluster_md_path=str(tmp_path / "md.pkl"),
        input_full_hmmsearch_results_path=str(tmp_path / "full.tsv"),
        input_full_profiles_dir_path=str(profiles_dir),
        input_all_sequences_fasta_path=str(tmp_path / "seqs.fasta"),
        input_ground_truth_path=str(tmp_path / "gt.pkl"),
        output_sdf_path=table,
        save_as_pickle=True,
    )

    assert os.path.exists(table + ".meta")
    with open(table, "rb") as f:
        saved = pickle.load(f)  # row/column order is set by the builder
    make_unreadable(table)
    meta = sdf_module.load_train_metadata(table, load_from_pickle=True)
    assert list(meta.row_labels) == [ground_truth[i] for i in saved.index_ids]
    assert list(meta.column_names) == list(saved.column_names)
    assert sorted(meta.column_names) == ["pA", "pB"]


def test_classification_preprocessing_uses_sidecar_column_order(json_table, tmp_path):
    from famus.data_preprocessing import hmmsearch_results_to_classification_sdf

    sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    make_unreadable(json_table)
    (tmp_path / "hits.tsv").write_text(
        "header\n"
        "q1\t-\tp1.fasta\t-\t-\t-\t-\t-\t12.0\t-\n"
        "q2\t-\tp2.fasta\t-\t-\t-\t-\t-\t7.0\t-\n"
    )
    (tmp_path / "query.fasta").write_text(">q1\nMK\n>q2\nMK\n>q3\nMK\n")
    out = str(tmp_path / "sdf_classify.pkl")

    hmmsearch_results_to_classification_sdf(
        input_hmmsearch_results_path=str(tmp_path / "hits.tsv"),
        input_train_sdf_path=json_table,
        input_fasta_path=str(tmp_path / "query.fasta"),
        output_path=out,
        load_sdf_from_pickle=False,
    )

    with open(out, "rb") as f:
        sdf_classify = pickle.load(f)
    assert list(sdf_classify.column_names) == ["p3", "p1", "p2"]
    rows = dict(zip(sdf_classify.index_ids, sdf_classify.matrix.toarray().tolist()))
    assert rows == {"q1": [0, 12, 0], "q2": [0, 0, 7], "q3": [0, 0, 0]}


@pytest.fixture
def classify_inputs(json_table, tmp_path):
    """A tiny real model, its embeddings of the training rows, and two queries that
    copy training rows t2 and t1 exactly, so each must match that row's labels."""
    torch = pytest.importorskip("torch")
    from famus.model import MLP

    torch.manual_seed(0)
    model = MLP(3, 320)  # MLP.load_from_state assumes embedding_dim=320
    model_path = str(tmp_path / "state.pt")
    model.save_state(model_path)
    with torch.no_grad():
        train_rows = torch.tensor(make_train_sdf().matrix.toarray(), dtype=torch.float32)
        embeddings = model(train_rows).numpy()
    embeddings_path = str(tmp_path / "train_embeddings.npy")
    np.save(embeddings_path, embeddings)

    query = SparseDataFrame(
        csr_matrix(np.array([[0, 7, 1], [5, 0, 0]])),
        ["q1", "q2"],
        None,
        ["p3", "p1", "p2"],
        dtype=np.int64,
    )
    query_path = str(tmp_path / "sdf_classify.pkl")
    with open(query_path, "wb") as f:
        pickle.dump(query, f)
    return SimpleNamespace(
        table=json_table,
        model=model_path,
        embeddings=embeddings_path,
        query=query_path,
        out=str(tmp_path / "predictions.tsv"),
    )


def run_classify(inputs, train_embeddings_path):
    from famus.classification import classify

    classify(
        sdf_train_path=inputs.table,
        sdf_classify_path=inputs.query,
        model_path=inputs.model,
        train_embeddings_path=train_embeddings_path,
        classification_embeddings_path=None,
        output_path=inputs.out,
        device="cpu",
        chunksize=10,
        threshold=0.5,
        n_processes=1,
        load_sdf_from_pickle=False,
    )
    with open(inputs.out) as f:
        return f.read()


def test_classify_uses_sidecar_labels_without_reading_table(classify_inputs):
    sdf_module.load_train_metadata(classify_inputs.table, load_from_pickle=False)
    make_unreadable(classify_inputs.table)
    predictions = run_classify(classify_inputs, classify_inputs.embeddings)
    assert predictions == "q1\tfamA;famB\nq2\tfamA\n"


def test_classify_without_saved_embeddings_still_uses_full_table(classify_inputs):
    predictions = run_classify(classify_inputs, train_embeddings_path=None)
    assert predictions == "q1\tfamA;famB\nq2\tfamA\n"


def test_classify_rejects_saved_embeddings_with_wrong_row_count(classify_inputs):
    sdf_module.load_train_metadata(classify_inputs.table, load_from_pickle=False)
    np.save(classify_inputs.embeddings, np.load(classify_inputs.embeddings)[:2])
    with pytest.raises(ValueError, match="stale"):
        run_classify(classify_inputs, classify_inputs.embeddings)


def test_unwritable_directory_still_returns_metadata(json_table, tmp_path):
    os.chmod(tmp_path, 0o555)
    try:
        meta = sdf_module.load_train_metadata(json_table, load_from_pickle=False)
    finally:
        os.chmod(tmp_path, 0o755)
    assert list(meta.column_names) == EXPECTED_COLUMNS
    assert list(meta.row_labels) == EXPECTED_ROW_LABELS
    assert os.listdir(tmp_path) == ["sdf_train.json"]  # no sidecar, no temp file
