import logging
import sys
from datetime import datetime
from functools import cache
from types import TracebackType

import yaml


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
    with open("cfg.yaml", "r") as fp:
        cfg = yaml.full_load(fp)
    return cfg


log_path = "tmp/logs/" + now()
with open(log_path, "w+") as f:
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


def log_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType,
) -> None:
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = log_exception
