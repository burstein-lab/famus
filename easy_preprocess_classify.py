import argparse
import os

import yaml

from app import data_preprocessing as dp
from app import get_cfg, logger
from app.utils import now


def main(
    input_fasta_file_path: str,
    input_full_profiles_dir_path: str,
    input_sdf_train_path: str,
    data_dir_path: str,
    nthreads: int = None,
) -> None:
    """
    Runs preprocessing before classification.
    :param input_fasta_file_path: path to input fasta file
    :param input_full_profiles_db_path: path to input full profiles db
    :param input_sdf_train_path: path to input train SparseDataFrame
    :param data_dir_path: path to directory where data will be stored.
    If directory does not exist, it will be created.
    If directory exists and was previously used for preprocessing that was stopped before finishing,
    the script will attempt to use the existing data to save time.
    :param nthreads: number of threads to use for parallel processing.
    If not specified, will use cfg.yaml parameter.
    :return: None
    """
    logger.info("Starting preprocessing")
    logger.info("Input fasta: {}".format(input_fasta_file_path))
    logger.info("Input full profiles dir: {}".format(input_full_profiles_dir_path))
    logger.info("Input sdf train: {}".format(input_sdf_train_path))
    logger.info("Data directory: {}".format(data_dir_path))

    cfg = get_cfg()
    if not nthreads:
        nthreads = cfg["nthreads"]
    logger.info("Number of threads: {}".format(nthreads))
    state_file_path = os.path.join(data_dir_path, ".classify_preprocessing_state")

    if not data_dir_path.endswith("/"):
        data_dir_path += "/"

    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)
        state = {
            "hmmsearch_tmp_path": None,
            "full_hmmsearch_results": None,
            "sdf_classify": None,
        }
        yaml.dump(
            state,
            open(state_file_path, "w+"),
        )
        skip = False
    else:
        if not os.path.exists(state_file_path):
            raise ValueError(
                "Data directory already exists but does not contain .classify_preprocessing_state."
            )
        state = yaml.full_load(open(state_file_path, "r"))
        skip = True

    if not state["full_hmmsearch_results"] or not skip:
        logger.info("Running full hmmsearch")
        full_hmmsearch_results = os.path.join(
            data_dir_path, "full_hmmsearch_results.txt"
        )
        if not (skip and state["hmmsearch_tmp_path"]):
            hmmsearch_tmp_path = os.path.join(data_dir_path, "hmmsearch/")
            os.makedirs(hmmsearch_tmp_path, exist_ok=True)
            state["hmmsearch_tmp_path"] = hmmsearch_tmp_path
            yaml.dump(state, open(state_file_path, "w+"))
        else:
            hmmsearch_tmp_path = state["hmmsearch_tmp_path"]

        dp.full_hmmsearch(
            input_full_profiles_dir_path=input_full_profiles_dir_path,
            input_fasta_path=input_fasta_file_path,
            output_full_hmmsearch_results_path=full_hmmsearch_results,
            tmp_dir_path=hmmsearch_tmp_path,
            n_processes=nthreads,
        )
        state["full_hmmsearch_results"] = full_hmmsearch_results
        yaml.dump(state, open(state_file_path, "w+"))
        skip = False
    else:
        logger.info("Using existing full hmmsearch results")
        full_hmmsearch_results = state["full_hmmsearch_results"]

    if not state["sdf_classify"] or not skip:
        logger.info("Running sdf_classify")
        skip = False
        sdf_classify_path = data_dir_path + "sdf_classify.pkl"
        dp.hmmsearch_results_to_classification_sdf(
            input_hmmsearch_results_path=full_hmmsearch_results,
            input_train_sdf_path=input_sdf_train_path,
            input_fasta_path=input_fasta_file_path,
            output_path=sdf_classify_path,
        )
        state["sdf_classify"] = sdf_classify_path
        with open(state_file_path, "w+") as f:
            yaml.dump(state, f)
    else:
        logger.info("Skipping sdf_classify")
        sdf_classify_path = state["sdf_classify"]

    logger.info("Finished preprocessing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fasta_file_path",
        type=str,
        help="[REQUIRED] Path to input fasta file",
        required=True,
    )
    parser.add_argument(
        "--input_full_profiles_dir_path",
        type=str,
        help="[REQUIRED] Path to input full profiles directory",
        required=True,
    )
    parser.add_argument(
        "--input_sdf_train_path",
        type=str,
        help="[REQUIRED] Path to input train SparseDataFrame",
        required=True,
    )
    parser.add_argument(
        "--data_dir_path",
        type=str,
        help="[REQUIRED] Path to directory where data will be stored. If directory does not exist, it will be created. If directory exists and was previously used for preprocessing that was stopped before finishing, the script will attempt to use the existing data to save time.",
        required=True,
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="Number of threads to use for parallel processing. If not specified, will use cfg.yaml parameter.s",
        required=False,
    )
    args = parser.parse_args()
    if args.nthreads:
        nthreads = args.nthreads
    else:
        nthreads = None
    main(
        input_fasta_file_path=args.input_fasta_file_path,
        input_full_profiles_dir_path=args.input_full_profiles_dir_path,
        input_sdf_train_path=args.input_sdf_train_path,
        data_dir_path=args.data_dir_path,
        nthreads=args.nthreads,
    )
