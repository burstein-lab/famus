import os
import pickle
import shutil
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from sklearn.metrics import f1_score

from app import logger
from app.classification import classify, _calc_embeddings
from preprocess_classify import main as epc_main

from .kofam_runner import kofam, shorten_spaces

matplotlib.use("Agg")
warnings.simplefilter(action="ignore", category=FutureWarning)


n_sequences_from_each_family = 50

ground_truth_path = "/davidb/guyshur/kegg_data/2023/ground_truth.pkl"


color_scheme = [
    "#D55E00",  # kofam and kofam weighted
    "#0072B2",  # famus cpu and weighted
    "#56B4E9",  # famus CUDA and micro
    "#009E73",  # famus light cpu and weighted
    "#10CEA3",  # famus light CUDA and micro
    "#E69F00",  # kofam micro
]

tmp_dir = "tmp/specific_family_results/"


def sample_input_fasta():
    family_sequence_file = (
        "/davidb/guyshur/kegg_data/diff_23_21/labeled_only_23_in_21_kos.fasta"
    )
    gt = pickle.load(open(ground_truth_path, "rb"))
    family_to_sequences = {}
    from tqdm import tqdm

    for seq, families in tqdm(gt.items(), desc="Sampling input fasta"):
        family = families.split(";")[0]
        if family not in family_to_sequences:
            family_to_sequences[family] = []
        family_to_sequences[family].append(seq)
    import random

    sampled_sequence_ids = []
    for family, sequences in tqdm(
        family_to_sequences.items(), desc="Sampling sequences"
    ):
        sampled_sequence_ids += random.sample(
            sequences, min(n_sequences_from_each_family, len(sequences))
        )
    sampled_sequence_ids = set(sampled_sequence_ids)
    logger.info(
        f"Sampled {len(sampled_sequence_ids)} sequences from {len(family_to_sequences)} families"
    )
    from Bio import SeqIO

    records = SeqIO.parse(family_sequence_file, "fasta")
    records_to_write = []
    for record in tqdm(records, desc="Filtering records"):
        if record.id in sampled_sequence_ids:
            records_to_write.append(record)

    output_fasta_path = os.path.join(tmp_dir, "input.fasta")
    with open(output_fasta_path, "w") as output_handle:
        SeqIO.write(records_to_write, output_handle, "fasta")
    logger.info(f"Sampled input fasta written to {output_fasta_path}")


full_profiles_path = "models/full/kegg_2021_dedup/data_dir/subcluster_profiles/"
light_full_profiles_path = "data/kegg_2021_no_sc/subcluster_profiles/"
sdf_train_path = "models/full/kegg_2021_dedup/data_dir/sdf_train.pkl"
light_train_sdf_path = "data/kegg_2021_no_sc/sdf_train.pkl"
train_embeddings_path = "models/full/kegg_2021_reborn/data_dir/train_embeddings.npy"
light_train_embeddings_path = "data/kegg_2021_no_sc/train_embeddings.npy"  # 2021
model = "kegg_2021_reborn"
N_PROC = 100
run_kofam = True


def get_model_results():
    ground_truth = pickle.load(open(ground_truth_path, "rb"))
    if run_kofam:
        kofam_tmp_path = os.path.join(tmp_dir, "kofam_tmp")
        kofam_output_path = os.path.join(tmp_dir, "kofam.tsv")
        os.makedirs(kofam_tmp_path, exist_ok=True)
        kofam_metrics_path = os.path.join(tmp_dir, "kofam_metrics.tsv")
        if not os.path.exists(kofam_output_path):
            kofam(
                fasta_path=os.path.join(tmp_dir, "input.fasta"),
                tmp_path=kofam_tmp_path,
                output_path=kofam_output_path,
                n_processes=N_PROC,
            )
        if not os.path.exists(kofam_metrics_path):
            calc_kofam_metrics(
                ground_truth=ground_truth,
                results_path=kofam_output_path,
                output_path=kofam_metrics_path,
            )
    output_path = os.path.join(tmp_dir, "famus.tsv")
    famus_data_dir = os.path.join(tmp_dir, "famus_data_dir")
    logger.info("Running famus")
    epc_main(
        input_fasta_file_path=os.path.join(tmp_dir, "input.fasta"),
        input_full_profiles_dir_path=full_profiles_path,
        input_sdf_train_path=sdf_train_path,
        data_dir_path=famus_data_dir,
        n_processes=N_PROC,
        load_sdf_from_pickle=True,
    )
    model_path = os.path.join("models", "full", model, "state.pt")
    classify(
        sdf_train_path=sdf_train_path,
        sdf_classify_path=os.path.join(famus_data_dir, "sdf_classify.pkl"),
        model_path=model_path,
        train_embeddings_path=train_embeddings_path,
        output_path=output_path,
        device="cpu",
        threshold=open(os.path.join("models", "full", model, "env")).read().strip(),
        n_processes=N_PROC,
        load_sdf_from_pickle=True,
    )


#     metrics_path = os.path.join(
#         results_dir, f"perc_{perc}_repeat_{repeat}_famus_light_metrics.tsv"
#     )
#     data_dir_path = os.path.join(curr_repeat_tmp_dir, "famus_light_data_dir")
#     if not os.path.exists(metrics_path):
#         predictions_path = os.path.join(data_dir_path, "preds.tsv")
#         sdf_classify_path = os.path.join(data_dir_path, "sdf_classify.pkl")
#         if not os.path.exists(sdf_classify_path):
#             epc_main(
#                 input_fasta_file_path=input_fasta_path,
#                 input_full_profiles_dir_path=light_full_profiles_path,
#                 input_sdf_train_path=light_train_sdf_path,
#                 data_dir_path=data_dir_path,
#                 n_processes=N_PROC,
#                 load_sdf_from_pickle=True,
#             )
#         else:
#             logger.info(f"perc: {perc}, repeat: {repeat} sdf_classify already exists")
#         if not os.path.exists(predictions_path):
#             classify(
#                 sdf_train_path=light_train_sdf_path,
#                 sdf_classify_path=sdf_classify_path,
#                 model_path=LIGHT_MODEL_PATH,
#                 train_embeddings_path=light_train_embeddings_path,
#                 classification_embeddings_path="",
#                 output_path=predictions_path,
#                 device="cpu",
#                 threshold=LIGHT_MODEL_THRESHOLD,
#                 n_processes=N_PROC,
#                 load_sdf_from_pickle=True,
#             )
#         else:
#             logger.info(f"perc: {perc}, repeat: {repeat} predictions already exists")
#         famus_calc_metrics(
#             results_path=predictions_path,
#             output_path=metrics_path,
#         )
#     if os.path.exists(data_dir_path):
#         shutil.rmtree(data_dir_path)

#     metrics_path = os.path.join(
#         results_dir, f"perc_{perc}_repeat_{repeat}_famus_metrics.tsv"
#     )

#     if not os.path.exists(metrics_path):
#         input_fasta_path = os.path.join(repeat_path, "input.fasta")
#         predictions_path = os.path.join(repeat_path, "preds.tsv")
#         n_labeled = int(n_sequences * (1 - perc))
#         n_unlabeled = int(n_sequences * perc)
#         data_dir_path = os.path.join(repeat_path, "data_dir")
#         sdf_classify_path = os.path.join(data_dir_path, "sdf_classify.pkl")

#         if not os.path.exists(input_fasta_path):
#             sample_fasta(
#                 n_labeled=n_labeled,
#                 n_unlabeled=n_unlabeled,
#                 output_path=input_fasta_path,
#                 tmp_path=curr_repeat_tmp_dir,
#             )
#         else:
#             logger.info(f"perc: {perc}, repeat: {repeat} fasta input already exists")
#         if not os.path.exists(sdf_classify_path):
#             logger.info("preprocessing")
#             epc_main(
#                 input_fasta_file_path=input_fasta_path,
#                 input_full_profiles_dir_path=full_profiles_path,
#                 input_sdf_train_path=sdf_train_path,
#                 data_dir_path=data_dir_path,
#                 n_processes=N_PROC,
#                 load_sdf_from_pickle=True,
#             )
#         else:
#             logger.info(f"perc: {perc}, repeat: {repeat} sdf_classify already exists")
#         if not os.path.exists(predictions_path):
#             logger.info("running classification")
#             classify(
#                 sdf_train_path=sdf_train_path,
#                 sdf_classify_path=sdf_classify_path,
#                 model_path=MODEL_PATH,
#                 train_embeddings_path=train_embeddings_path,
#                 classification_embeddings_path="",
#                 output_path=predictions_path,
#                 device="cpu",
#                 threshold=FULL_MODEL_THRESHOLD,
#                 n_processes=N_PROC,
#                 load_sdf_from_pickle=True,
#             )
#         else:
#             logger.info(f"perc: {perc}, repeat: {repeat} predictions already exists")
#         logger.info("calculating metrics")
#         famus_calc_metrics(
#             results_path=predictions_path,
#             output_path=metrics_path,
#         )

#     else:
#         logger.info(f"perc: {perc}, repeat: {repeat} already exists")

#     if not os.path.exists(results_dir + f"perc_{perc}_repeat_{repeat}_kofam.tsv"):
#         kofam(
#             fasta_path=tmp_dir + f"perc_{perc}/repeat_{repeat}/input.fasta",
#             tmp_path=tmp_dir,
#             output_path=results_dir + f"perc_{perc}_repeat_{repeat}_kofam.tsv",
#             n_processes=N_PROC,
#         )
#         os.chdir("/davidb/guyshur/famus/")
#     if not os.path.exists(
#         results_dir + f"perc_{perc}_repeat_{repeat}_kofam_metrics.tsv"
#     ):
#         calc_kofam_metrics(
#             input_fasta=tmp_dir + f"perc_{perc}/repeat_{repeat}/input.fasta",
#             results_path=results_dir + f"perc_{perc}_repeat_{repeat}_kofam.tsv",
#             ground_truth=ground_truth_path,
#             output_path=results_dir + f"perc_{perc}_repeat_{repeat}_kofam_metrics.tsv",
#         )


# # plot_results(results_dir)


def calc_metrics(y_true, y_pred, output_path):
    """
    Calculate F1 score for each family and save the results to a file.
    """
    unique_families = set(y_true)
    family_f1_scores = {}
    for family in unique_families:
        mask = y_true == family
        curr_y_pred, curr_y_true = y_pred[mask], y_true[mask]
        family_f1_scores[family] = f1_score(curr_y_true, curr_y_pred, zero_division=0)
    return family_f1_scores


def calc_kofam_metrics(ground_truth, results_path, output_path):
    # gt = pickle.load(open(ground_truth, "rb"))
    # gt = {k: v.split(";") for k, v in gt.items()}
    from Bio import SeqIO

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
    for record in SeqIO.parse(open(), "fasta"):
        if record.id not in preds:
            preds[record.id] = "unknown"
        y_pred.append(preds[record.id])
        if (
            len(ground_truth[record.id]) > 1
            and preds[record.id] in ground_truth[record.id]
        ):
            y_true.append(preds[record.id])
        else:
            y_true.append(ground_truth[record.id][0])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return calc_metrics(y_true, y_pred, output_path)


def famus_calc_metrics(ground_truth, results_path: str, output_path: str):
    results = open(results_path, "r").readlines()
    results = [line.strip().split("\t") for line in results if "sample" not in line]
    seqs, y_pred = zip(*results)
    y_pred = [pred.split(";") for pred in y_pred]
    y_true = [ground_truth[seq] for seq in seqs]
    for i, preds in enumerate(y_pred):
        curr_ground_truth = ground_truth[seqs[i]]
        curr_ground_truth = curr_ground_truth.split(";")

        if (len(curr_ground_truth) > 1 or len(preds) > 1) and len(
            set(preds) & set(curr_ground_truth)
        ) > 0:
            y_pred[i] = ground_truth[seqs[i]]
        else:
            y_pred[i] = preds[0]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return calc_metrics(y_true, y_pred, output_path)


if __name__ == "__main__":
    sample_input_fasta()
