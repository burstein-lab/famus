import os
import random

from Bio import SeqIO

from app import get_cfg
from app.utils import concatenate_files

only_2023_but_in_2021_kos_path = (
    "/davidb/guyshur/kegg_data/diff_23_21/labeled_only_23_in_21_kos.fasta"
)
only_2023_unlabeled = "/davidb/guyshur/kegg_data/diff_23_21/only_23_unlabeled.fasta"

cfg = get_cfg()


def sample_fasta(
    n_labeled: int,
    n_unlabeled: int,
    output_path: str,
    tmp_path: str,
) -> None:
    assert isinstance(n_labeled, int) and n_labeled > 0
    assert isinstance(n_unlabeled, int) and n_unlabeled >= 0
    assert isinstance(output_path, str)

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    sample_knowns(n_labeled, tmp_path + "sampled_knowns.fasta")
    sample_unknowns(n_unlabeled, tmp_path + "sampled_unknowns.fasta")
    concatenate_files(
        [
            tmp_path + "sampled_knowns.fasta",
            tmp_path + "sampled_unknowns.fasta",
        ],
        output_path,
    )


def _random_sample(input_path: str, output_path: str, n: int) -> None:
    records = list(SeqIO.parse(input_path, "fasta"))
    sampled_records = random.sample(records, n)
    with open(output_path, "w+") as f:
        SeqIO.write(sampled_records, f, "fasta")


def sample_unknowns(
    n_unlabeled: int,
    output_path: str,
    unknowns_path=only_2023_unlabeled,
) -> None:
    assert isinstance(n_unlabeled, int) and n_unlabeled > 0
    assert isinstance(output_path, str) and os.path.exists(os.path.dirname(output_path))
    assert os.path.exists(unknowns_path)
    _random_sample(unknowns_path, output_path, n_unlabeled)


def sample_knowns(
    n_labeled: int,
    output_path: str,
    labeled_path=only_2023_but_in_2021_kos_path,
) -> None:
    assert isinstance(n_labeled, int) and n_labeled > 0
    assert os.path.exists(labeled_path)
    assert isinstance(output_path, str)
    assert os.path.exists(os.path.dirname(output_path))
    _random_sample(labeled_path, output_path, n_labeled)
