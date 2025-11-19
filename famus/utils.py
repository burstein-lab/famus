import shutil
from datetime import datetime
import subprocess

from tqdm import tqdm


GIGABYTE = 1024**3


def count_sequences(filename):
    """
    Count the number of '>' characters in a file using subprocess.

    Args:
        filename: Path to the file to read

    Returns:
        int: Number of '>' characters found
    """
    try:
        # Use grep -o to find all occurrences of '>', then wc -l to count them
        result = subprocess.run(
            f"grep -o '>' {filename} | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        # grep returns exit code 1 if no matches found
        return 0


def sample_sequences(input_file: str, output_file: str, sample_size: int) -> None:
    """
    Use seqkit to sample a file of sequences.
    """
    subprocess.run(
        ["seqkit", "sample", "-n", str(sample_size), input_file, "-o", output_file]
    )


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
