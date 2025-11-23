import argparse
from famus import config


def get_common_model_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--models-dir",
        type=str,
        help=f"Directory to save or load models. [{config.DEFAULT_MODELS_DIR}]",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        help=f"Number of processes to use. [{config.DEFAULT_N_PROCESSES}]",
    )
    parser.add_argument(
        "--device",
        type=str,
        help=f"Device to use (cpu or cuda). [{config.DEFAULT_DEVICE}]",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help=f"Number of sequences to process at once for classification or threshold calculation. [{config.DEFAULT_CHUNKSIZE}]",
    )

    return parser
