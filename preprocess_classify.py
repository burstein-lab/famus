import argparse
import os

import yaml

from app import data_preprocessing as dp
from app import get_cfg, logger


def main(
    input_fasta_file_path: str,
    input_full_profiles_dir_path: str,
    input_sdf_train_path: str,
    data_dir_path: str,
    n_processes: int = None,
    load_sdf_from_pickle: bool = False,
) -> None:
    logger.info("Starting preprocessing")
    logger.info("Input fasta: {}".format(input_fasta_file_path))
    logger.info("Input full profiles dir: {}".format(input_full_profiles_dir_path))
    logger.info("Input sdf train: {}".format(input_sdf_train_path))
    logger.info("Data directory: {}".format(data_dir_path))

    cfg = get_cfg()

    if not n_processes:
        n_processes = cfg["n_processes"]
    logger.info("Number of processes: {}".format(n_processes))
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
        if not os.path.exists(input_fasta_file_path):
            raise ValueError(f"{input_fasta_file_path} does not exist")
        if not os.path.exists(input_full_profiles_dir_path):
            raise ValueError(f"{input_full_profiles_dir_path} does not exist")
        if not os.path.exists(input_sdf_train_path):
            raise ValueError(f"{input_sdf_train_path} does not exist")
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
            n_processes=n_processes,
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
            load_sdf_from_pickle=load_sdf_from_pickle,
        )
        state["sdf_classify"] = sdf_classify_path
        with open(state_file_path, "w+") as f:
            yaml.dump(state, f)
    else:
        logger.info("Skipping sdf_classify")
        sdf_classify_path = state["sdf_classify"]

    logger.info("Finished preprocessing")


if __name__ == "__main__":
    logger.info("Starting preprocessing")
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
        "--n_processes",
        type=int,
        help="Number of processes to use for parallel processing. If not specified, will use cfg.yaml parameter.s",
        required=False,
    )
    parser.add_argument("--load_sdf_from_pickle", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.n_processes:
        n_processes = args.n_processes
    else:
        n_processes = None
    main(
        input_fasta_file_path=args.input_fasta_file_path,
        input_full_profiles_dir_path=args.input_full_profiles_dir_path,
        input_sdf_train_path=args.input_sdf_train_path,
        data_dir_path=args.data_dir_path,
        n_processes=args.n_processes,
        load_sdf_from_pickle=args.load_sdf_from_pickle,
    )
