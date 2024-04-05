from app import fasta_parser
from app.utils import even_split
import os
from app import logger

kofam_scan_path = "/davidb/guyshur/kofam_scan/"
max_ids_per_file = 10000

base_cmd = "./exec_annotation -o {} --format=detail {}"


def kofam(fasta_path, tmp_path, output_path, n_processes=1):
    os.chdir(kofam_scan_path)
    parser = fasta_parser.FastaParser(fasta_path)
    ids = parser.get_ids()
    num_ids = len(ids)
    ids = even_split(ids, int(num_ids / max_ids_per_file) + 1)
    for i, curr_ids in enumerate(ids):
        logger.info("Writing file {}/{}".format(i + 1, len(ids)))
        tmp_fasta_path = os.path.join(tmp_path, "tmp_{}.fasta".format(i))
        with open(tmp_fasta_path, "w+") as h:
            parser.export_sequences(curr_ids, h)
    for i in range(len(ids)):
        logger.info("Running kofam_scan {}/{}".format(i + 1, len(ids)))
        os.system(
            base_cmd.format(
                os.path.join(tmp_path, "tmp_{}.out".format(i)),
                os.path.join(tmp_path, "tmp_{}.fasta".format(i)),
            )
        )
    logger.info("Concatenating results")
    os.system("cp {} {}".format(os.path.join(tmp_path, "tmp_0.out"), output_path))
    for i in range(1, len(ids)):
        # copy all but the first line
        os.system(
            "tail -n +3 {} >> {}".format(
                os.path.join(tmp_path, "tmp_{}.out".format(i)), output_path
            )
        )
