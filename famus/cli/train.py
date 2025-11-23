import argparse
import os
import sys
import yaml

from famus.logging import setup_logger
from famus.classification import calculate_threshold
from famus.train import train
from famus.cli.preprocess_train import main as preprocess
from .common import get_common_parser
from .common_model_args import get_common_model_args_parser
from famus import config


def main():
    prog = os.path.basename(sys.argv[0])
    if prog.endswith(".py"):
        prog = "python -m famus.cli.train"
    parser = argparse.ArgumentParser(
        parents=[get_common_parser(), get_common_model_args_parser()],
        description="Train a FAMUS model",
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Example usage:

  # {prog} --unknown-sequences-fasta-path examples/unknowns.fasta --log-dir logs/ --n-processes 32 --models-dir models/ --save-every 100_000 --device cpu --num-epochs 20 --batch-size 32 --create-subclusters examples/example_orthologs/

  Full description of arguments can be found at https://github.com/burstein-lab/famus
        """,
    )
    parser.add_argument(
        "input_fasta_dir_path",
        type=str,
        help="""Path to directory containing input fasta files representing protein families.
        Must only include fasta files.""",
    )
    parser.add_argument(
        "--create-subclusters",
        action=argparse.BooleanOptionalAction,
        help=f"Whether to create subclusters within each protein family (--create-subclusters for comprehensive model, --no-create-subclusters for light model). [default: {config.DEFAULT_CREATE_SUBCLUSTERS}]",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Optional name for the model which will be used to request it during classification. The default value is the name of the input directory.",
    )
    parser.add_argument(
        "--unknown-sequences-fasta-path",
        type=str,
        help="Path to fasta file containing sequences not belonging to any given protein family.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help=f"Number of epochs to train the model. If not specified, will use cfg.yaml parameter. [{config.DEFAULT_NUM_EPOCHS}]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Batch size for training the model. If not specified, will use cfg.yaml parameter. [{config.DEFAULT_BATCH_SIZE}]",
    )
    parser.add_argument(
        "--stop-before-training",
        action="store_true",
        help=f"Stop right before training the model. Useful for running preprocess and train separately. [{config.DEFAULT_STOP_BEFORE_TRAINING}]",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help=f"Number of batches after which to save a checkpoint. [{config.DEFAULT_SAVE_EVERY_BATCHES}]",
    )
    parser.add_argument(
        "--mmseqs-n-processes",
        type=int,
        help=f"Number of processes to use for MMseqs2 during preprocessing. [{config.DEFAULT_MMSEQS_N_PROCESSES}]",
    )
    parser.add_argument(
        "--sampled-sequences-per-subcluster",
        help=f"Number of sequences to sample per subcluster for training during preprocessing. [{config.DEFAULT_SAMPLED_SEQUENCES_PER_SUBCLUSTER}]",
    )
    parser.add_argument(
        "--fraction-of-sampled-unknown-sequences",
        help=f"Fraction of unknown sequences to sample for training during preprocessing. [{config.DEFAULT_FRACTION_OF_SAMPLED_UNKNOWN_SEQUENCES}]",
    )
    parser.add_argument(
        "--samples-profiles-product-limit",
        type=int,
        help=f"Limit on the product of number of sampled sequences and number of profiles during preprocessing. [{config.DEFAULT_SAMPLES_PROFILES_PRODUCT_LIMIT}]",
    )
    parser.add_argument(
        "--sequences-max-len-product-limit",
        type=int,
        help=f"Limit on the product of number of sequences and their maximum length during preprocessing. [{config.DEFAULT_SEQUENCES_MAX_LEN_PRODUCT_LIMIT}]",
    )
    parser.add_argument(
        "--mmseqs-cluster-coverage",
        type=float,
        help=f"MMseqs2 cluster coverage parameter during preprocessing. [{config.DEFAULT_MMSEQS_CLUSTER_COVERAGE}]",
    )
    parser.add_argument(
        "--mmseqs-cluster-identity",
        type=float,
        help=f"MMseqs2 cluster identity parameter during preprocessing. [{config.DEFAULT_MMSEQS_CLUSTER_IDENTITY}]",
    )
    parser.add_argument(
        "--mmseqs-coverage-subclusters",
        type=float,
        help=f"MMseqs2 coverage for subclusters parameter during preprocessing. [{config.DEFAULT_MMSEQS_COVERAGE_SUBCLUSTERS}]",
    )
    parser.add_argument(
        "--log-to-wandb",
        action=argparse.BooleanOptionalAction,
        help=f"Whether to log training to Weights & Biases. [{config.DEFAULT_LOG_TO_WANDB}]",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help=f"Weights & Biases project name to use if logging to wandb. [{config.DEFAULT_WANDB_PROJECT}]",
    )
    parser.add_argument(
        "--wandb-api-key-path",
        type=str,
        help=f"Path to file containing Weights & Biases API key to use if logging to wandb. [{config.DEFAULT_WANDB_API_KEY_PATH}]",
    )

    args = parser.parse_args()
    cfg_file_path = args.config
    cfg = (
        config.load_cfg(cfg_file_path) if cfg_file_path else config.get_default_config()
    )
    no_log = args.no_log or cfg["no_log"]
    log_dir = args.log_dir or cfg["log_dir"]
    logger = setup_logger(enable_logging=not no_log, log_dir=log_dir)
    input_fasta_dir_path = args.input_fasta_dir_path
    n_processes = args.n_processes or cfg["n_processes"]
    num_epochs = args.num_epochs or cfg["num_epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    device = args.device or cfg["device"]
    chunksize = args.chunksize or cfg["chunksize"]
    save_every = args.save_every or cfg["save_every"]
    models_dir = args.models_dir or cfg["models_dir"]
    mmseqs_n_processes = args.mmseqs_n_processes or cfg["mmseqs_n_processes"]
    sampled_sequences_per_subcluster = (
        args.sampled_sequences_per_subcluster or cfg["sampled_sequences_per_subcluster"]
    )
    try:
        sampled_sequences_per_subcluster = int(sampled_sequences_per_subcluster)
    except ValueError:
        pass
    fraction_of_sampled_unknown_sequences = (
        args.fraction_of_sampled_unknown_sequences
        or cfg["fraction_of_sampled_unknown_sequences"]
    )
    try:
        fraction_of_sampled_unknown_sequences = float(
            fraction_of_sampled_unknown_sequences
        )
    except ValueError:
        pass
    samples_profiles_product_limit = (
        args.samples_profiles_product_limit or cfg["samples_profiles_product_limit"]
    )
    sequences_max_len_product_limit = (
        args.sequences_max_len_product_limit or cfg["sequences_max_len_product_limit"]
    )
    mmseqs_cluster_coverage = (
        args.mmseqs_cluster_coverage or cfg["mmseqs_cluster_coverage"]
    )
    mmseqs_cluster_identity = (
        args.mmseqs_cluster_identity or cfg["mmseqs_cluster_identity"]
    )
    mmseqs_coverage_subclusters = (
        args.mmseqs_coverage_subclusters or cfg["mmseqs_coverage_subclusters"]
    )

    model_name = args.model_name
    unknown_sequences_fasta_path = args.unknown_sequences_fasta_path
    if args.create_subclusters is None:
        create_subclusters = cfg["create_subclusters"]
    else:
        create_subclusters = args.create_subclusters
    if not isinstance(create_subclusters, bool):
        raise ValueError("create_subclusters must be True or False.")
    if device not in ["cpu", "cuda"]:
        raise ValueError("Invalid device. Please choose from 'cpu' or 'cuda")

    model_type = "comprehensive" if create_subclusters else "light"

    if not model_name:
        model_name = os.path.basename(input_fasta_dir_path.rstrip("/"))
    model_path = os.path.join(models_dir, model_type, model_name)
    model_path = model_path + "/" if model_path[-1] != "/" else model_path
    data_dir_path = os.path.join(model_path, "data_dir/")

    if os.path.exists(os.path.join(model_path, "env")):
        logger.info(f"Model {model_name} already exists with env file. Exiting.")
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
                },
                f,
            )
        with open(os.path.join(model_path, "cfg.yaml"), "a") as f:
            f.write(
                "\n# This file was created by train.py to store model preprocessing and training parameters."
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
        mmseqs_n_processes=mmseqs_n_processes,
        sampled_sequences_per_subcluster=sampled_sequences_per_subcluster,
        fraction_of_sampled_unknown_sequences=fraction_of_sampled_unknown_sequences,
        samples_profiles_product_limit=samples_profiles_product_limit,
        sequences_max_len_product_limit=sequences_max_len_product_limit,
        mmseqs_cluster_coverage=mmseqs_cluster_coverage,
        mmseqs_cluster_identity=mmseqs_cluster_identity,
        mmseqs_coverage_subclusters=mmseqs_coverage_subclusters,
    )
    stop_before_training = args.stop_before_training or cfg["stop_before_training"]
    if stop_before_training:
        logger.info("stop_before_training set to True. Stopping before training.")
        return
    sdfloader_path = os.path.join(data_dir_path, "sdfloader.pkl")
    checkpoints_dir_path = os.path.join(model_path, "checkpoints/")
    os.makedirs(checkpoints_dir_path, exist_ok=True)
    output_path = os.path.join(model_path, "state.pt")
    log_to_wandb = (
        args.log_to_wandb if args.log_to_wandb is not None else cfg["log_to_wandb"]
    )
    wandb_project = (
        args.wandb_project if args.wandb_project is not None else cfg["wandb_project"]
    )
    wandb_api_key_path = (
        args.wandb_api_key_path
        if args.wandb_api_key_path is not None
        else cfg["wandb_api_key_path"]
    )
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
        log_to_wandb=log_to_wandb,
        wandb_project=wandb_project,
        wandb_api_key_path=wandb_api_key_path,
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
    logger.info("Finished training successfully.")


if __name__ == "__main__":
    main()
