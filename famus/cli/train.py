import argparse
import os

import yaml

from famus import get_cfg, logger
from famus.classification import calculate_threshold
from famus.train import train
from famus import MODELS_ROOT
from famus.cli.preprocess_train import main as preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data for training and train a model."
    )
    parser.add_argument(
        "--input_fasta_dir_path",
        type=str,
        required=True,
        help="""[REQUIRED] Path to directory containing input fasta files representing protein families.\n
        Must only include fasta files.""",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model to train. Options are 'full' or 'light'. The default value is in cfg.yaml in 'create_subclusters (True = full, False = light)'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Optional name for the model which will be used to request it during classification. The default value is the name of the input directory.",
    )
    parser.add_argument(
        "--unknown_sequences_fasta_path",
        type=str,
        help="Path to fasta file containing sequences not belonging to any given protein family.",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        help="Number of processes to use for parallel processing. If not specified, will use cfg.yaml parameter.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train the model. If not specified, will use cfg.yaml parameter.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training the model. If not specified, will use cfg.yaml parameter.",
    )
    parser.add_argument(
        "--stop_before_training",
        action="store_true",
        help="Stop right before training the model. Useful for running preprocess and train separately.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for training the model (cpu or cuda). If not specified, will use cfg.yaml parameter.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Number of sequences to process at once for threshold calculation. If not specified, will use cfg.yaml parameter.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        help="Number of batches after which to save a checkpoint. Default is 100,000.",
    )
    logger.info("Starting easy_train.py...")
    args = parser.parse_args()
    cfg = get_cfg()
    input_fasta_dir_path = args.input_fasta_dir_path
    n_processes = args.n_processes or cfg["n_processes"]
    num_epochs = args.num_epochs or cfg["num_epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    model_type = args.model_type
    device = args.device
    chunksize = args.chunksize or cfg["chunksize"]
    save_every = args.save_every or cfg["save_every"]
    model_name = args.model_name
    unknown_sequences_fasta_path = args.unknown_sequences_fasta_path
    if device:
        if device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device. Please choose from 'cpu' or 'cuda")
    else:
        device = cfg["user_device"]
        if device not in ["cpu", "cuda"]:
            raise ValueError(
                "Invalid user_device in cfg.yaml. Please choose from 'cpu' or 'cuda'."
            )

    if model_type:
        if model_type not in ["full", "light"]:
            raise ValueError(
                "Invalid model type. Please choose from 'full' or 'light'."
            )
        create_subclusters = model_type == "full"
    else:
        create_subclusters = cfg["create_subclusters"]
        if not isinstance(create_subclusters, bool):
            raise ValueError(
                "Invalid create_subclusters in cfg.yaml. Please set to True or False."
            )
        model_type = "full" if create_subclusters else "light"

    if not model_name:
        model_name = os.path.basename(input_fasta_dir_path.rstrip("/"))
    model_path = os.path.join(MODELS_ROOT, model_type, model_name)
    model_path = model_path + "/" if model_path[-1] != "/" else model_path
    data_dir_path = os.path.join(model_path, "data_dir/")

    if os.path.exists(os.path.join(model_path, "env")):
        logger.info(f"Model {model_name} already exists. Exiting.")
        return
    elif os.path.exists(model_path):
        previous_model_cfg = os.path.join(model_path, "cfg.yaml")
        if not os.path.exists(previous_model_cfg):
            raise ValueError(
                f"Model {model_name} already exists and did not finish training but no cfg.yaml file found. Exiting."
            )
        with open(previous_model_cfg, "r") as f:
            previous_cfg = yaml.safe_load(f)
            discordant_keys = []

            if input_fasta_dir_path != previous_cfg["input_fasta_dir_path"]:
                discordant_keys.append("input_fasta_dir_path")
            if (
                unknown_sequences_fasta_path
                != previous_cfg["unknown_sequences_fasta_path"]
            ):
                discordant_keys.append("unknown_sequences_fasta_path")
            if num_epochs != previous_cfg["num_epochs"]:
                discordant_keys.append("num_epochs")
            if batch_size != previous_cfg["batch_size"]:
                discordant_keys.append("batch_size")
            if model_type != previous_cfg["model_type"]:
                discordant_keys.append("model_type")
            if discordant_keys:
                raise ValueError(
                    (
                        f"Model {model_name} already exists and did not finish training. The following "
                        f"parameters are different from the previous run: {discordant_keys}. Please delete "
                        "the model directory or change the parameters to match the previous run: "
                        f"input_fasta_dir_path={previous_cfg['input_fasta_dir_path']}, "
                        f"unknown_sequences_fasta_path={previous_cfg['unknown_sequences_fasta_path']}, "
                        f"num_epochs={previous_cfg['num_epochs']}, batch_size={previous_cfg['batch_size']}, "
                        f"model_type={previous_cfg['model_type']}."
                    )
                )
    else:
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "cfg.yaml"), "w") as f:
            yaml.dump(
                {
                    "model_type": model_type,
                    "input_fasta_dir_path": input_fasta_dir_path,
                    "unknown_sequences_fasta_path": unknown_sequences_fasta_path,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                },
                f,
            )
        with open(os.path.join(model_path, "cfg.yaml"), "a") as f:
            f.write(
                "\n# This file was created by easy_train.py to store model preprocessing and training parameters."
            )
            f.write(
                "\n# Do not edit this file manually before the model training is finished unless you are sure of what you are doing."
            )

    preprocess(
        input_fasta_dir_path=input_fasta_dir_path,
        data_dir_path=data_dir_path,
        unknown_sequences_fasta_path=unknown_sequences_fasta_path,
        n_processes=n_processes,
        create_subclusters=create_subclusters,
    )
    if args.stop_before_training:
        logger.info("stop_before_training set to True. Stopping before training.")
        return
    sdfloader_path = os.path.join(data_dir_path, "sdfloader.pkl")
    checkpoints_dir_path = os.path.join(model_path, "checkpoints/")
    os.makedirs(checkpoints_dir_path, exist_ok=True)
    output_path = os.path.join(model_path, "state.pt")
    train(
        sdfloader_path=sdfloader_path,
        output_path=output_path,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_checkpoints=True,
        checkpoint_dir_path=checkpoints_dir_path,
        evaluation=True,
        save_every=save_every,
        lr=0.001,
        n_processes=n_processes,
    )

    if not os.path.exists(os.path.join(model_path, "env")):
        threshold = calculate_threshold(
            sdf_train_path=data_dir_path + "sdf_train.json",
            model_path=output_path,
            train_embeddings_path=data_dir_path + "train_embeddings.pkl",
            device=device,
            chunksize=chunksize,
            n_processes=n_processes,
        )
        with open(os.path.join(model_path, "env"), "w") as f:
            f.write(str(f"THRESHOLD={threshold}"))
    logger.info("Finished easy_train.py.")


if __name__ == "__main__":
    main()
