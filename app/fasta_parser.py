import os
import random
import subprocess
from typing import TextIO


class FastaParser(object):
    """
    A class to parse fasta files.
    Can be used to get a list of all ids in the fasta file, or to export a list of ids to a given handle.
    """

    def __init__(self, fasta_file):
        with open(fasta_file, "r") as fasta_file:
            self.fasta_file_content = fasta_file.read()
            self.fasta_file_lines = self.fasta_file_content.split("\n")
            if self.fasta_file_lines[-1] == "":
                self.fasta_file_lines = self.fasta_file_lines[:-1]
            self.ids = [
                line.split(" ")[0][1:]
                for line in self.fasta_file_lines
                if line[0] == ">"
            ]

    def __len__(self) -> int:
        return len(self.get_ids())

    def __getitem__(self, index) -> str:
        return self.get_ids()[index]

    def __contains__(self, id) -> bool:
        return id in self.get_ids

    def get_ids(self):
        """
        Returns a list of all ids in the fasta file.
        """
        return self.ids

    def get_random_ids(self, n: int) -> list:
        """
        Returns a list of n random ids from the fasta file.
        :param n: number of ids to return.
        :return: list of n random ids from the fasta file.
        """
        if not isinstance(n, int) and n > 0:
            raise TypeError("n must be a positive integer")
        if not n <= len(self):
            raise ValueError(
                "n must be less or equal to the number of sequences in the fasta file"
            )
        ids_to_export = random.sample(self.get_ids(), n)
        return ids_to_export

    def export_sequences(self, ids_to_export: list, handle: TextIO) -> None:
        """
        Exports the sequences with the given ids to the given handle.
        :param ids_to_export: list of ids to export.
        :param handle: handle to export the sequences to.
        :return: None
        """
        pid = str(os.getpid())
        self.get_ids()  # in case it is not yet initialized
        pool = set(self.ids)
        if not all([id in pool for id in ids_to_export]):
            raise ValueError(
                "ids_to_export contains ids that are not in the fasta file"
            )
        with open(f"tmp/{pid}.allseq", "w+") as f:
            f.write(self.fasta_file_content)
        with open(f"tmp/{pid}.relseq", "w+") as f:
            f.write("\n".join(ids_to_export))
        cmd = f"seqtk subseq tmp/{pid}.allseq tmp/{pid}.relseq"
        handle.write(subprocess.check_output(cmd.split(" ")).decode())
        os.remove(f"tmp/{pid}.allseq")
        os.remove(f"tmp/{pid}.relseq")

    def export_random_sequences(self, n: int, handle: TextIO) -> None:
        """
        Exports n random sequences to the given handle.
        :param n: number of sequences to export.
        :param handle: handle to export the sequences to.
        """
        ids_to_export = self.get_random_ids(n)
        self.export_sequences(ids_to_export, handle)
