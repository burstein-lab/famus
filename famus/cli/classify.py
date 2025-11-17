import argparse
import os

from famus.classification import classify
from famus.cli.preprocess_classify import main as preprocess
import shutil


import yaml

from famus.logging import setup_logger
from .common import get_common_parser
from famus.config import get_default_config


def main():
    parser = argparse.ArgumentParser(
        parents=[get_common_parser()],
        description="Classify protein sequences using installed models.",
    )
    parser.add_argument(
        "input_fasta_file_path",
        type=str,
        help="Path to input fasta file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Models to use for classification separated by spaces",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Type of model(s) to use (comprehensive or light)",
    )
    parser.add_argument("--load-sdf-from-pickle", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    cfg_file_path = args.config
    if cfg_file_path:
        with open(cfg_file_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = get_default_config()
    no_log = args.no_log
    log_dir = args.log_dir or cfg["log_dir"]
    logger = setup_logger(enable_logging=not no_log, log_dir=log_dir)
    logger.info("Starting classification...")

    input_fasta_file_path = args.input_fasta_file_path
    output_dir = args.output_dir
    device = args.device or cfg["device"]
    chunksize = args.chunksize or cfg["chunksize"]
    models = args.models or cfg["models"]
    models_type = args.model_type or cfg["model_type"]
    n_processes = args.n_processes or cfg["n_processes"]
    models_dir = args.models_dir or cfg["models_dir"]
    model_paths = [os.path.join(models_dir, models_type, model) for model in models]
    if missing_models := [model for model in model_paths if not os.path.exists(model)]:
        raise FileNotFoundError(f"Missing models: {missing_models}")

    if args.load_sdf_from_pickle:
        for model in models:
            model_path = os.path.join(models_dir, models_type, model)
            sdf_train_path = os.path.join(model_path, "data_dir", "sdf_train.pkl")
            if not os.path.exists(sdf_train_path):
                raise FileNotFoundError(
                    f"--load-sdf-from-pickle was passed but missing sdf_train.pkl for {model}. Did you run famus-convert-sdf?"
                )

    os.makedirs(output_dir, exist_ok=True)
    for model in models:
        prediction_path = os.path.join(
            output_dir, f"{model}_classification_results.tsv"
        )
        if os.path.exists(prediction_path):
            logger.info(
                f"Classification results for {model} already exist. Skipping classification."
            )
            continue
        logger.info(f"Preprocessing data for {model}")
        model_path = os.path.join(models_dir, models_type, model)
        if args.load_sdf_from_pickle:
            sdf_train_path = os.path.join(model_path, "data_dir", "sdf_train.pkl")
        else:
            sdf_train_path = os.path.join(model_path, "data_dir", "sdf_train.json")
        input_full_profiles_dir_path = os.path.join(
            model_path, "data_dir", "subcluster_profiles/"
        )
        curr_tmp_path = os.path.join(output_dir, model)

        preprocess(
            input_fasta_file_path=input_fasta_file_path,
            input_full_profiles_dir_path=input_full_profiles_dir_path,
            input_sdf_train_path=sdf_train_path,
            data_dir_path=curr_tmp_path,
            n_processes=n_processes,
            load_sdf_from_pickle=args.load_sdf_from_pickle,
        )
        logger.info(f"Classifying data for {model}")
        threshold = open(os.path.join(model_path, "env")).read().strip()
        threshold = threshold.split("=")
        if not threshold[0] == "THRESHOLD":
            raise ValueError("Error in env file")
        threshold = float(threshold[1])
        sdf_classify_path = os.path.join(curr_tmp_path, "sdf_classify.pkl")
        train_embeddings_path = os.path.join(
            model_path, "data_dir", "train_embeddings.npy"
        )
        model_state_path = os.path.join(model_path, "state.pt")
        classify(
            sdf_train_path=sdf_train_path,
            sdf_classify_path=sdf_classify_path,
            model_path=model_state_path,
            train_embeddings_path=train_embeddings_path,
            classification_embeddings_path=None,
            output_path=prediction_path,
            device=device,
            chunksize=chunksize,
            threshold=threshold,
            n_processes=n_processes,
            load_sdf_from_pickle=args.load_sdf_from_pickle,
        )
        logger.info(f"Deleting {model} temporary files")
        shutil.rmtree(curr_tmp_path)
        logger.info(f"Finished classifying with {model}")
    logger.info("Finished classification successfully.")


if __name__ == "__main__":
    main()
