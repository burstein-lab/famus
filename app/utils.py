import shutil
from datetime import datetime
import subprocess

import torch
from tqdm import tqdm

from app import logger

GIGABYTE = 1024**3


def count_sequences(file: str) -> int:
    """
    Count the number of sequences in a fasta file.
    """
    with open(file) as f:
        return sum(1 for line in f if line.startswith(">"))


def sample_sequences(input_file: str, output_file: str, sample_size: int) -> None:
    """
    Use seqkit to sample a file of sequences.
    """
    subprocess.run(
        ["seqkit", "sample", "-n", str(sample_size), input_file, "-o", output_file]
    )


def _log_gpu_memory() -> None:
    """
    Log the current GPU memory usage.
    :return: None
    """
    current_bytes_in_gpu = torch.cuda.memory_allocated()
    current_gb_in_gpu = current_bytes_in_gpu // GIGABYTE
    logger.info("Current GPU memory: " + str(current_gb_in_gpu) + "GB")


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


def even_split(a: list, n: int) -> list:
    """
    Split a list into n roughly equal parts
    """
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def concatenate_files(files: list, output: str, track_progress=True) -> None:
    """
    Concatenate a list of files into a single file
    """
    with open(output, "wb+") as wfd:
        if track_progress:
            for file in tqdm(files, desc="concatenating files"):
                with open(file, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)
        else:
            for file in files:
                with open(file, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)
