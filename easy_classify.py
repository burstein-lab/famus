import argparse
import os

from app import get_cfg
from models import models_root_path
from app.classification import classify
from preprocess_classify import main as preprocess
from app import logger
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify protein sequences using installed models."
    )
    parser.add_argument(
        "--input_fasta_file_path",
        type=str,
        required=True,
        help="[REQUIRED] input fasta file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="[REQUIRED] output directory",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        help="Number of processes to use for hmmsearch and cpu-based classification",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for classification (cpu or cuda)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Number of sequences to classify at once",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Models to use for classification separated by spaces",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model(s) to use (full or light)",
    )
    parser.add_argument("--load_sdf_from_pickle", action=argparse.BooleanOptionalAction)
    logger.info("Starting easy_classify.py...")
    args = parser.parse_args()
    input_fasta_file_path = args.input_fasta_file_path
    output_dir = args.output_dir
    cfg = get_cfg()
    device = cfg["user_device"] if args.device is None else args.device
    chunksize = cfg["chunksize"] if args.chunksize is None else args.chunksize
    models = cfg["models"] if args.models is None else args.models
    models_type = cfg["models_type"] if args.model_type is None else args.model_type
    n_processes = (
        args.n_processes if args.n_processes is not None else cfg["n_processes"]
    )
    model_paths = [
        os.path.join(models_root_path, models_type, model) for model in models
    ]
    if missing_models := [model for model in model_paths if not os.path.exists(model)]:
        raise FileNotFoundError(f"Missing models: {missing_models}")

    if args.load_sdf_from_pickle:
        for model in models:
            model_path = os.path.join(models_root_path, models_type, model)
            sdf_train_path = os.path.join(model_path, "data_dir", "sdf_train.pkl")
            if not os.path.exists(sdf_train_path):
                raise FileNotFoundError(
                    f"Missing sdf_train.pkl for {model}. Did you run convert_sdf.py?"
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
        model_path = os.path.join(models_root_path, models_type, model)
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
        logger.info(f"Finished classifying {model}")
    logger.info("Finished easy_classify.py.")
