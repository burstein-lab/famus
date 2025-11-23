"""
Defines defaults for FAMUS configuration parameters.
"""

from pathlib import Path
import os
import yaml

FAMUS_HOME = Path.home() / ".famus"
FAMUS_HOME = Path(os.environ.get("FAMUS_HOME", FAMUS_HOME))

DEFAULT_LOG_DIR = FAMUS_HOME / "logs"
DEFAULT_NO_LOG = False
DEFAULT_N_PROCESSES = 4
DEFAULT_MODELS_DIR = FAMUS_HOME / "models"
DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHUNKSIZE = 20_000
DEFAULT_SAVE_EVERY_BATCHES = 100_000
DEFAULT_USER_DEVICE = "cuda"
DEFAULT_MODEL_TYPE = "comprehensive"
DEFAULT_CREATE_SUBCLUSTERS = True
DEFAULT_MMSEQS_N_PROCESSES = 4
DEFAULT_SAMPLED_SEQUENCES_PER_SUBCLUSTER = 60
DEFAULT_FRACTION_OF_SAMPLED_UNKNOWN_SEQUENCES = 1.0
DEFAULT_SAMPLES_PROFILES_PRODUCT_LIMIT = 150_000_000_000_000
DEFAULT_SEQUENCES_MAX_LEN_PRODUCT_LIMIT = 500_000_000
DEFAULT_MMSEQS_CLUSTER_COVERAGE = 0.8
DEFAULT_MMSEQS_CLUSTER_IDENTITY = 0.9
DEFAULT_MMSEQS_COVERAGE_SUBCLUSTERS = 0.5
DEFAULT_LOG_TO_WANDB = False
DEFAULT_WANDB_PROJECT = "famus"
DEFAULT_WANDB_API_KEY_PATH = "wandb_api_key.txt"
DEFAULT_STOP_BEFORE_TRAINING = False
DEFAULT_LOAD_SDF_FROM_PICKLE = False


def load_cfg(config_file, defaults_only=False):
    default_cfg = get_default_config()

    if defaults_only:
        return default_cfg

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in default_cfg.items():
        if key not in cfg:
            cfg[key] = value

    return cfg


def get_default_config():
    """
    Returns the default configuration as a dictionary.
    """
    return {
        "log_dir": str(DEFAULT_LOG_DIR),
        "no_log": DEFAULT_NO_LOG,
        "n_processes": DEFAULT_N_PROCESSES,
        "models_dir": str(DEFAULT_MODELS_DIR),
        "num_epochs": DEFAULT_NUM_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "save_every": DEFAULT_SAVE_EVERY_BATCHES,
        "chunksize": DEFAULT_CHUNKSIZE,
        "device": DEFAULT_USER_DEVICE,
        "model_type": DEFAULT_MODEL_TYPE,
        "create_subclusters": DEFAULT_CREATE_SUBCLUSTERS,
        "mmseqs_n_processes": DEFAULT_MMSEQS_N_PROCESSES,
        "sampled_sequences_per_subcluster": DEFAULT_SAMPLED_SEQUENCES_PER_SUBCLUSTER,
        "fraction_of_sampled_unknown_sequences": DEFAULT_FRACTION_OF_SAMPLED_UNKNOWN_SEQUENCES,
        "samples_profiles_product_limit": DEFAULT_SAMPLES_PROFILES_PRODUCT_LIMIT,
        "sequences_max_len_product_limit": DEFAULT_SEQUENCES_MAX_LEN_PRODUCT_LIMIT,
        "mmseqs_cluster_coverage": DEFAULT_MMSEQS_CLUSTER_COVERAGE,
        "mmseqs_cluster_identity": DEFAULT_MMSEQS_CLUSTER_IDENTITY,
        "mmseqs_coverage_subclusters": DEFAULT_MMSEQS_COVERAGE_SUBCLUSTERS,
        "log_to_wandb": DEFAULT_LOG_TO_WANDB,
        "wandb_project": DEFAULT_WANDB_PROJECT,
        "wandb_api_key_path": DEFAULT_WANDB_API_KEY_PATH,
        "stop_before_training": DEFAULT_STOP_BEFORE_TRAINING,
        "load_sdf_from_pickle": DEFAULT_LOAD_SDF_FROM_PICKLE,
    }


def main():
    print("Default FAMUS configuration:")
    cfg = get_default_config()
    for key, value in cfg.items():
        print(f"{key}: {value}")
    print("=" * 40)
    print("You can customize these parameters by creating a YAML config file.")
    print("See https://github.com/burstein-lab/famus for more information.")


if __name__ == "__main__":
    main()
