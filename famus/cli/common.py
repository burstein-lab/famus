import argparse
from famus import config


def get_common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--no-log",
        action="store_true",
        help=f"Disable logging. [{config.DEFAULT_NO_LOG}]",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help=f"Directory to save logs. [{config.DEFAULT_LOG_DIR}]",
    )
    parser.add_argument("--config", type=str, help="Path to config file")

    return parser
