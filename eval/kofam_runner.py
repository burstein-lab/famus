import os
import pickle
import subprocess
import sys
import shutil
from subprocess import STDOUT, CalledProcessError, check_output
from uuid import uuid4

from Bio import SeqIO
from sklearn.metrics import f1_score
from app import logger
from app.utils import even_split

kofam_scan_path = "/davidb/guyshur/kofam_scan/"
max_ids_per_file = 10000
tmp_dir = "/davidb/guyshur/famus/tmp/kofam_tmp/"
base_cmd = "./exec_annotation -o {} --format=detail --tmp-dir={} --cpu={} {}"


def kofam(fasta_path, tmp_path, output_path, n_processes=1):
    if not os.path.exists(tmp_path):
        raise FileNotFoundError(f"{tmp_path} does not exist")
    curr_dir_full_path = os.path.dirname(os.path.realpath(__file__))
    # get full path of tmp_path
    os.chdir(kofam_scan_path)
    renamed_records = {}
    records = []
    seen_ids = set()

    for record in SeqIO.parse(open(fasta_path), "fasta"):
        if record.id not in seen_ids:
            records.append(record)
            seen_ids.add(record.id)
        else:
            old_id = record.id
            new_id = record.id + "_" + str(uuid4())
            renamed_records[new_id] = old_id
            record.id = new_id
            records.append(record)

    records = even_split(records, int(len(records) / max_ids_per_file) + 1)
    for i, curr_records in enumerate(records):
        logger.info("Writing file {}/{}".format(i + 1, len(records)))
        tmp_fasta_path = os.path.join(tmp_path, "tmp_{}.fasta".format(i))
        with open(tmp_fasta_path, "w+") as h:
            SeqIO.write(curr_records, h, "fasta")
    for i in range(len(records)):
        curr_tmp_dir = os.path.join(tmp_dir, str(uuid4()))
        os.makedirs(curr_tmp_dir)
        logger.info("Running kofam_scan {}/{}".format(i + 1, len(records)))
        cmd = base_cmd.format(
            os.path.join(tmp_path, "tmp_{}.out".format(i)),
            curr_tmp_dir,
            n_processes,
            os.path.join(tmp_path, "tmp_{}.fasta".format(i)),
        )
        logger.info(cmd)
        # os.system(
        #     base_cmd.format(
        #         os.path.join(tmp_path, "tmp_{}.out".format(i)),
        #         curr_tmp_dir,
        #         n_processes,
        #         os.path.join(tmp_path, "tmp_{}.fasta".format(i)),
        #     )
        # )
        try:
            check_output(
                base_cmd.format(
                    os.path.join(tmp_path, "tmp_{}.out".format(i)),
                    curr_tmp_dir,
                    n_processes,
                    os.path.join(tmp_path, "tmp_{}.fasta".format(i)),
                ).split(),
                stderr=STDOUT,
            )
        except CalledProcessError as exc:
            logger.error(f"Failed to run {cmd} - {exc.output}")
            exit(1)
        shutil.rmtree(curr_tmp_dir)
        if not os.path.exists(os.path.join(tmp_path, "tmp_{}.out".format(i))):
            logger.error(f"famus failed to run kofam_scan on {i}th file")
            exit(1)
    for i in range(len(records)):
        with open(os.path.join(tmp_path, "tmp_{}.out".format(i))) as h:
            result_text = h.read()
        for k, v in renamed_records.items():
            result_text = result_text.replace(k, v)
        with open(os.path.join(tmp_path, "tmp_{}.out".format(i)), "w") as h:
            h.write(result_text)
    logger.info("Concatenating results")

    os.system("cp {} {}".format(os.path.join(tmp_path, "tmp_0.out"), output_path))
    for i in range(1, len(records)):
        # copy all but the first line
        os.system(
            "tail -n +3 {} >> {}".format(
                os.path.join(tmp_path, "tmp_{}.out".format(i)), output_path
            )
        )
    # delete all tmp files
    for i in range(len(records)):
        os.remove(os.path.join(tmp_path, "tmp_{}.out".format(i)))
        os.remove(os.path.join(tmp_path, "tmp_{}.fasta".format(i)))
    os.chdir(curr_dir_full_path)


def shorten_spaces(string):
    return [substring for substring in string.split(" ") if substring]


def calc_kofam_metrics(input_fasta, results_path, ground_truth, output_path):
    gt = pickle.load(open(ground_truth, "rb"))
    gt = {k: v.split(";") for k, v in gt.items()}
    preds = {}
    top_scores = {}
    kofam_output = open(results_path, "r").readlines()
    kofam_output = [
        shorten_spaces(line) for line in kofam_output if line.startswith("*")
    ]
    for line in kofam_output:
        if top_scores.get(line[1], 0) < float(line[4]):
            top_scores[line[1]] = float(line[4])
            preds[line[1]] = line[2]
    y_true = []
    y_pred = []
    for record in SeqIO.parse(open(input_fasta), "fasta"):
        if record.id not in preds:
            preds[record.id] = "unknown"
        y_pred.append(preds[record.id])
        if len(gt[record.id]) > 1 and preds[record.id] in gt[record.id]:
            y_true.append(preds[record.id])
        else:
            y_true.append(gt[record.id][0])

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    logger.info(f"Kofam Weighted F1: {weighted_f1}, Micro F1: {micro_f1}")
    with open(output_path, "w+") as output_handle:
        output_handle.write(
            str(round(weighted_f1, 3)) + "\t" + str(round(micro_f1, 3)) + "\n"
        )
