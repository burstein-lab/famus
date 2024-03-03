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
        if ".sub_cluster.cluster." in f and f.endswith(".fasta")
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

    output = {}
    augmented_subcluster_fasta_paths = [
        os.path.join(input_augmented_subclusters_fastas_dir_path, f)
        for f in os.listdir(input_augmented_subclusters_fastas_dir_path)
    ]
    assert all(f.endswith(".fasta") for f in augmented_subcluster_fasta_paths)
    for fasta_path in tqdm(augmented_subcluster_fasta_paths, desc="subclusters"):
        orthology_name = re.sub(
            SUBCLUSTER_SUFFIX_PATTERN, "", os.path.basename(fasta_path)
        )

        with open(fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id not in output:
                    output[record.id] = ""
                else:
                    output[record.id] += ";"
                output[record.id] += orthology_name

    leftovers_fasta_paths = [
        os.path.join(input_leftovers_dir_path, f)
        for f in os.listdir(input_leftovers_dir_path)
    ]
    assert all(f.endswith(".fasta") for f in leftovers_fasta_paths)
    for fasta_path in tqdm(leftovers_fasta_paths, desc="leftovers"):
        orthology_name = os.path.basename(fasta_path).removesuffix(".leftovers.fasta")
        with open(fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id not in output:
                    output[record.id] = ""
                else:
                    output[record.id] += ";"
                output[record.id] += orthology_name
    if input_unannotated_sequences_fasta_path:
        with open(input_unannotated_sequences_fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id in output:
                    raise ValueError(
                        "Sequence {} is in both annotated and unannotated sequences".format(
                            record.id
                        )
                    )
                output[record.id] = "unknown"
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
):
    if isinstance(number_of_sampled_sequences_per_subcluster, int):
        logger.info("sampling subclusters for model")
        sample_subclusters_for_model(
            augmented_subcluster_dir,
            output_sampled_subclusters_fasta_path=data_dir_path
            + "sampled_subclusters.fasta",
            number_of_sampled_sequences_per_subcluster=number_of_sampled_sequences_per_subcluster,
        )
        os.system(
            "cat "
            + data_dir_path
            + "sampled_subclusters.fasta > "
            + output_full_hmmsearch_input_path
        )
    else:
        logger.info("using all subclusters for model")
        concatenate_files(
            files=[
                augmented_subcluster_dir + f
                for f in os.listdir(augmented_subcluster_dir)
            ],
            output=output_full_hmmsearch_input_path,
            track_progress=False,
        )

    n_sampled_subcluster_sequences = int(
        subprocess.check_output(
            "grep -c '>' " + output_full_hmmsearch_input_path, shell=True
        )
    )
    leftover_paths = [os.path.join(leftovers_dir, f) for f in os.listdir(leftovers_dir)]
    concatenate_files(leftover_paths, os.path.join(data_dir_path, "leftovers.fasta"))
    n_leftover_sequences = int(
        subprocess.check_output(
            "grep -c '>' " + os.path.join(data_dir_path, "leftovers.fasta"), shell=True
        )
    )
    n_total_sequences = n_sampled_subcluster_sequences + n_leftover_sequences
    os.system(
        "cat "
        + os.path.join(data_dir_path, "leftovers.fasta")
        + " >> "
        + output_full_hmmsearch_input_path
    )

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
        sample_unknown_sequences_for_model(
            input_unknown_sequences_fasta_path=input_unknown_sequences_fasta_path,
            n_sequences=int(n_total_sequences * fraction_of_sampled_unknown_sequences),
            tmp_dir_path=tmp_dir_path,
            output_sampled_unknown_sequences_fasta_path=data_dir_path
            + "sampled_unknown_sequences.fasta",
        )
        os.system(
            "cat "
            + data_dir_path
            + "sampled_unknown_sequences.fasta >> "
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
    sampled_records = []
    for fasta_path in tqdm(fasta_paths, desc="sampling subclusters"):
        with open(fasta_path, "r") as f:
            records = [record for record in SeqIO.parse(f, "fasta")]
        n_seqs = min(number_of_sampled_sequences_per_subcluster, len(records))
        sampled_records.extend(random.sample(records, n_seqs))
    with open(output_sampled_subclusters_fasta_path, "w+") as f:
        SeqIO.write(sampled_records, f, "fasta")


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
    with open(output_path, "w+") as handle:
        handle.write(
            subprocess.check_output(
                "app/qmafft {} 1 --quiet".format(input_path).split(" ")
            ).decode(sys.stdout.encoding)
        )


def hmmbuild(input_path: str, output_path: str, name: str) -> None:
    """
    Uses hmmbuild to build a profile from an alignment and saves the profile to a file.
    :param input_path: path to input alignment file
    :param output_path: path to output profile file
    :param name: name of profile
    :return: None
    """
    subprocess.call(
        "hmmbuild -n {} --amino {} {}".format(name, output_path, input_path).split(" "),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    subcluster_paths = get_subcluster_fasta_paths(subcluster_dir)
    if not os.path.exists(profile_dir):
        os.mkdir(profile_dir)
    profile_file_paths = [
        profile_dir + f for f in os.listdir(profile_dir) if f.endswith(".hmm")
    ]
    for fullname in profile_file_paths:
        if os.path.getsize(fullname) == 0:
            os.remove(fullname)
    logger.info("checking for existing profiles")
    existing_profiles = [f for f in os.listdir(profile_dir) if f.endswith(".hmm")]
    existing_profiles = set([f.removesuffix(".hmm") for f in os.listdir(profile_dir)])
    subcluster_paths = [
        path
        for path in subcluster_paths
        if path.split("/")[-1].removesuffix(".faa").removesuffix(".fasta")
        not in existing_profiles
    ]
    logger.info("creating {} profiles".format(len(subcluster_paths)))
    pool = mp.Pool(processes=nthreads)
    pool.starmap(
        _create_subcluster_profile,
        iterable=[(path, profile_dir, tmp_dir_path) for path in subcluster_paths],
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
) -> None:
    try:
        # run hmmsearch
        cmd = "hmmsearch -o /dev/null --tblout {} --cpu {} {} {}".format(
            output_path,
            str(threads),
            fasta_file_path,
            profiles_db_path,
        )
        logger.info(cmd)
        completed_process = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)
        if completed_process.returncode != 0:
            raise ValueError(
                f"hmmsearch failed with return code {completed_process.returncode}. command: {cmd}"
            )
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
        logger.info("skipping {}".format(subcluster_split_name))
        return
    logger.info("starting {}".format(subcluster_split_name))
    # run hmmsearch
    hmmsearch_output_path = output_path + subcluster_split_name + ".hmmsearch"
    cmd = "hmmsearch -o /dev/null --tblout {}".format(hmmsearch_output_path)
    cmd += " {} {}".format(profile_path, file_for_scoring)
    logger.info("Running CMD: " + cmd)
    cmd = cmd.split(" ")
    cmd = [s for s in cmd if s]
    subprocess.run(cmd, stdout=subprocess.PIPE)
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
            line = line.strip("\n").split(">")
            mapping[line[0]] = line[1]
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


def single_ortholog_to_subclusters(
    file_path: str,
    subclusters_output_path: str,
    leftovers_output_path: str,
    tmp_nr90: str,
    tmp_clu: str,
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

    fname = os.path.basename(file_path).removesuffix(".fasta")

    # check if the file has already been processed
    if os.path.exists(f"{leftovers_output_path}{fname}.leftovers.fasta"):
        # leftovers are created last so if they exist, the subclusters exist
        logger.info(f"{fname} has already been processed, skipping")
        return
    existing_subclusters_in_output = [
        os.path.join(subclusters_output_path, f)
        for f in os.listdir(subclusters_output_path)
        if fname in f
    ]
    if len(existing_subclusters_in_output) > 0:
        if _subclusters_fully_contain_rep_seqs(
            existing_subclusters_in_output, file_path
        ):
            # no leftovers, but all sequences are in subclusters - therefore the file has already been processed
            logger.info(f"{fname} has already been processed, skipping")
            return
        # no leftovers, and some (but not all) sequences are in subclusters - therefore the file started processing but did not finish
        for path in existing_subclusters_in_output:
            os.remove(path)

    # process the file
    _convert_fasta_headers(
        file_path,
        f"{tmp_nr90}{fname}.fasta",
        f"{tmp_nr90}{fname}.mapping",
    )
    subprocess.call(
        f"mmseqs easy-cluster {tmp_nr90}{fname}.fasta {tmp_nr90}{fname} {tmp_nr90}{fname}_tmp --threads {mmseq_nthreads} -s 7.5 -c 0.8 --min-seq-id 0.9".split(
            " "
        ),
        stdout=subprocess.DEVNULL,
    )
    shutil.copyfile(
        f"{tmp_nr90}{fname}_rep_seq.fasta",
        f"{output_rep_seq_dir}{fname}.fasta",
    )
    subprocess.call(
        f"mmseqs easy-cluster {tmp_nr90}{fname}_rep_seq.fasta {tmp_clu}{fname} -s 7.5 -c 0.5 {tmp_clu}{fname}_tmp --threads {mmseq_nthreads}".split(
            " "
        ),
        stdout=subprocess.DEVNULL,
    )
    clusters = {}
    seq_id_to_cluster = {}
    with open(f"{tmp_clu}{fname}_cluster.tsv", "r") as f:
        for line in f.readlines():
            line = line.strip("\n").split("\t")
            cluster, seq_id = line[0], line[1]
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(seq_id)
            seq_id_to_cluster[seq_id] = cluster
    total_seqs = sum([len(v) for v in clusters.values()])
    c = 5
    while c > 0:
        coverage = sum([len(v) for v in clusters.values() if len(v) >= c]) / total_seqs
        if coverage >= 0.9:
            break
        c -= 1
    with open(f"{tmp_nr90}{fname}_rep_seq.fasta", "r") as f:
        records = [record for record in SeqIO.parse(f, "fasta")]
    records = {record.id: record for record in records}
    subcluster_counter = 1
    records_to_leftovers = []

    for cluster, seq_ids in clusters.items():
        if len(seq_ids) >= c:
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
            subcluster_counter += 1
        else:
            records_to_leftovers += [records[seq_id] for seq_id in seq_ids]

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
    if len(records_to_leftovers) > 0:
        with open(f"{tmp_clu}{fname}.leftovers.fasta", "w+") as f:
            SeqIO.write(records_to_leftovers, f, "fasta")
        _convert_fasta_headers_back(
            f"{tmp_clu}{fname}.leftovers.fasta",
            f"{tmp_nr90}{fname}.mapping",
            f"{leftovers_output_path}{fname}.leftovers.fasta",
        )
        os.remove(f"{tmp_clu}{fname}.sub_cluster.cluster.0.fasta")
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
    (
        tmp_clu,
        tmp_nr90,
        output_rep_seq_dir,
        ortholog_fasta_dir,
        output_subcluster_dir,
        output_leftovers_dir,
    ) = create_paths_if_not_exists(
        [
            tmp_clu,
            tmp_nr90,
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
                output_rep_seq_dir,
            ),
        )
        time.sleep(0.2)  # prevents mmseqs from crashing
    pool.close()
    pool.join()
