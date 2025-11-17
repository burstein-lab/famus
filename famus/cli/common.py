import argparse
from famus import config


def get_common_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    parser.add_argument(
        "--log-dir",
        default=config.DEFAULT_LOG_DIR,
        type=str,
        help="Directory to save logs",
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--n-processes",
        default=config.DEFAULT_N_PROCESSES,
        type=int,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--models-dir",
        default=config.DEFAULT_MODELS_DIR,
        type=str,
        help="Directory to save or load models",
    )
    parser.add_argument(
        "--device",
        default=config.DEFAULT_USER_DEVICE,
        type=str,
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--chunksize",
        default=config.DEFAULT_CHUNKSIZE,
        type=int,
        help="Number of sequences to process at once for classification or threshold calculation",
    )

    return parser
