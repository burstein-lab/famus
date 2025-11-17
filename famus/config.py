"""
Defines defaults for FAMUS configuration parameters.
"""

from pathlib import Path
import os

FAMUS_HOME = Path.home() / ".famus"
FAMUS_HOME = Path(os.environ.get("FAMUS_HOME", FAMUS_HOME))

DEFAULT_LOG_DIR = FAMUS_HOME / "logs"
DEFAULT_N_PROCESSES = 4
DEFAULT_MODELS_DIR = FAMUS_HOME / "models"
DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHUNKSIZE = 20_000
DEFAULT_SAVE_EVERY_BATCHES = 100_000
DEFAULT_USER_DEVICE = "cuda"
DEFAULT_MODEL_TYPE = "comprehensive"
DEFAULT_CREATE_SUBCLUSTERS = True


def get_default_config():
    """
    Returns the default configuration as a dictionary.
    """
    return {
        "log_dir": str(DEFAULT_LOG_DIR),
        "n_processes": DEFAULT_N_PROCESSES,
        "models_dir": str(DEFAULT_MODELS_DIR),
        "num_epochs": DEFAULT_NUM_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "chunksize": DEFAULT_CHUNKSIZE,
        "device": DEFAULT_USER_DEVICE,
        "model_type": DEFAULT_MODEL_TYPE,
        "create_subclusters": DEFAULT_CREATE_SUBCLUSTERS,
    }
