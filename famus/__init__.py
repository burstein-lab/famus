from famus.__version__ import __version__
import logging
import os
import sys
from datetime import datetime
from functools import cache
from types import TracebackType
from importlib.resources import files
import yaml

TMP_DIR = files("famus") / "tmp"
MODELS_ROOT = files("famus") / "models"


class InactiveLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def now() -> str:
    """
    Get the current time as a string.
    :return: current time as a string.
    """
    return (
        str(datetime.now())
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


@cache
def get_cfg() -> dict:
    """
    Load configuration from cfg.yaml.
    First tries to load from current directory, then from package directory.
    """
    # Try current directory first (for development/running from repo)
    local_config = files("famus") / "cfg.yaml"
    with open(local_config, "r") as fp:
        cfg = yaml.full_load(fp)
    return cfg


if get_cfg()["logging"]:
    log_dir_path = files("famus") / "tmp" / "logs"
    os.makedirs(log_dir_path, exist_ok=True)
    log_path = log_dir_path / now()
    with open(log_path, "w") as f:
        pass
    logging.basicConfig(
        filename=log_path,
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stderr))

else:
    logger = InactiveLogger()


def log_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType,
) -> None:
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = log_exception
