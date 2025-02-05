from preprocess_train import main as preprocess
from app.train import train
from app import get_cfg
import argparse
import os
from models import models_root_path
from app import logger
from app.classification import calculate_threshold


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
        action=argparse.BooleanOptionalAction,
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

    logger.info("Starting easy_train.py...")
    args = parser.parse_args()
    cfg = get_cfg()
    n_processes = args.n_processes or cfg["nthreads"]
    num_epochs = args.num_epochs or cfg["num_epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    model_type = args.model_type
    device = args.device
    chunksize = args.chunksize or cfg["chunksize"]

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
    else:
        create_subclusters = cfg["create_subclusters"]
        if not isinstance(create_subclusters, bool):
            raise ValueError(
                "Invalid create_subclusters in cfg.yaml. Please set to True or False."
            )
        model_type = "full" if create_subclusters else "light"

    input_fasta_dir_path = args.input_fasta_dir_path
    model_name = args.model_name
    if not model_name:
        model_name = os.path.basename(input_fasta_dir_path.rstrip("/"))
    existing_models = os.listdir(os.path.join(models_root_path, model_type))
    if model_name in existing_models:
        raise ValueError(
            f"Model for {model_name} already exists. Please change the input directory name or provide a different name with --model_name."
        )
    model_path = os.path.join(models_root_path, model_type, model_name)
    model_path = model_path + "/" if model_path[-1] != "/" else model_path
    data_dir_path = os.path.join(model_path, "data_dir/")
    unknown_sequences_fasta_path = args.unknown_sequences_fasta_path
    preprocess(
        input_fasta_dir_path=input_fasta_dir_path,
        data_dir_path=data_dir_path,
        unknown_sequences_fasta_path=unknown_sequences_fasta_path,
        nthreads=n_processes,
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
    )
    if not os.path.exists(os.path.join(model_path, "env")):
        threshold = calculate_threshold(
            sdf_train_path=data_dir_path + "sdf_train.json",
            model_path=output_path,
            train_embeddings_path=data_dir_path + "train_embeddings.pkl",
            device=device,
            chunksize=chunksize,
            nthreads=n_processes,
        )
        with open(os.path.join(model_path, "env"), "w") as f:
            f.write(str(f"THRESHOLD={threshold}"))
    logger.info("Finished easy_train.py.")


if __name__ == "__main__":
    main()
