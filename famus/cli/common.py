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

    return parser
