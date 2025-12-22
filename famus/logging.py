import logging
import sys
from pathlib import Path
from datetime import datetime
from types import TracebackType

logger = logging.getLogger("famus")
_is_configured = False


def auto_configure():
    global _is_configured
    if _is_configured:
        return

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    _is_configured = True


def setup_logger(enable_logging=True, log_dir=None, verbose=False):
    global _is_configured

    logger.handlers.clear()

    if not enable_logging:
        logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        _is_configured = True
        return logger

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console_format = "%(levelname)s: %(message)s"
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"famus_{timestamp}.log"

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")

    def log_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = log_exception
    logger.propagate = False
    _is_configured = True
    return logger


auto_configure()
