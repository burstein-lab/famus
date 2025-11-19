import argparse
import os

import yaml

from famus import data_preprocessing as dp
from famus.logging import logger
from famus.sdfloader import prepare_sdfloader


def main(
    input_fasta_dir_path,
    data_dir_path,
    unknown_sequences_fasta_path,
    n_processes,
    create_subclusters,
    mmseqs_n_processes,
    number_of_sampled_sequences_per_subcluster,
    fraction_of_sampled_unknown_sequences,
    samples_profiles_product_limit,
    mmseqs_cluster_coverage,
    mmseqs_cluster_identity,
    mmseqs_coverage_subclusters,
) -> None:
    logger.info("Starting preprocessing")

    if not os.path.exists(input_fasta_dir_path):
        raise ValueError("Input fasta directory does not exist")

    state_file_path = os.path.join(data_dir_path, ".training_preprocessing_state")

    if not (
        isinstance(mmseqs_cluster_coverage, float) and 0 <= mmseqs_cluster_coverage <= 1
    ):
        raise ValueError(
            f"mmseqs_cluster_coverage must be a float between 0 and 1. got {mmseqs_cluster_coverage}"
        )
    if not (
        isinstance(mmseqs_cluster_identity, float) and 0 <= mmseqs_cluster_identity <= 1
    ):
        raise ValueError(
            f"mmseqs_cluster_identity must be a float between 0 and 1. got {mmseqs_cluster_identity}"
        )
    if not (
        isinstance(mmseqs_coverage_subclusters, float)
        and 0 <= mmseqs_coverage_subclusters <= 1
    ):
        raise ValueError(
            f"mmseqs_coverage_subclusters must be a float between 0 and 1. got {mmseqs_coverage_subclusters}"
        )
    if not (
        number_of_sampled_sequences_per_subcluster == "use_all"
        or (
            isinstance(number_of_sampled_sequences_per_subcluster, int)
            and number_of_sampled_sequences_per_subcluster > 0
        )
    ):
        raise ValueError(
            f"number_of_sampled_sequences_per_subcluster must be 'use_all' or a positive integer. got {number_of_sampled_sequences_per_subcluster}"
        )

    if not (
        fraction_of_sampled_unknown_sequences in ["use_all", "do_not_use"]
        or (
            isinstance(fraction_of_sampled_unknown_sequences, float)
            and fraction_of_sampled_unknown_sequences > 0
        )
    ):
        raise ValueError(
            f"fraction_of_sampled_unknown_sequences must be 'use_all', 'do_not_use', or a positive float. got {fraction_of_sampled_unknown_sequences}"
        )
    if (
        unknown_sequences_fasta_path
        and fraction_of_sampled_unknown_sequences == "do_not_use"
    ):
        raise ValueError(
            "unknown_sequences_fasta_path was provided but fraction_of_sampled_unknown_sequences is set to 'do_not_use'"
        )
    if (
        not unknown_sequences_fasta_path
        and fraction_of_sampled_unknown_sequences != "do_not_use"
    ):
        raise ValueError(
            "unknown_sequences_fasta_path was not provided but fraction_of_sampled_unknown_sequences is not set to 'do_not_use'"
        )

    if (
        not isinstance(samples_profiles_product_limit, int)
        or samples_profiles_product_limit < 1
    ):
        raise ValueError(
            f"samples_profiles_product_limit must be a positive integer. got {samples_profiles_product_limit}"
        )
    if not isinstance(create_subclusters, bool):
        raise ValueError(
            f"create_subclusters must be True or False. got {create_subclusters}"
        )
    if (
        not isinstance(samples_profiles_product_limit, int)
        or samples_profiles_product_limit < 1
    ):
        raise ValueError(
            f"samples_profiles_product_limit must be a positive integer. got {samples_profiles_product_limit}"
        )

    if not data_dir_path.endswith("/"):
        data_dir_path += "/"

    logger.info("Input fasta directory: {}".format(input_fasta_dir_path))
    logger.info("Data directory: {}".format(data_dir_path))
    logger.info("Number of processes: {}".format(n_processes))

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        state = {
            "subcluster_dir": None,
            "rep_seq_dir": None,
            "augmented_subcluster_dir": None,
            "full_profile_dir": None,
            "full_input_fasta": None,
            "hmmsearch_tmp_path": None,
            "full_hmmsearch_results": None,
            "subcluster_split_fastas_dir": None,
            "subcluster_split_scoring_dir": None,
            "subcluster_split_profiles_dir": None,
            "split_subcluster_md": None,
            "split_hmmsearch_results": None,
            "ground_truth": None,
            "sdf_train": None,
            "sdfloader": None,
        }
        yaml.dump(
            state,
            open(state_file_path, "w+"),
        )
        skip = False
    else:
        if not os.path.exists(state_file_path):
            raise ValueError(
                "Data directory already exists but does not contain .training_preprocessing_state."
            )
        state = yaml.full_load(open(state_file_path, "r"))
        skip = True

    # create tmp directory
    tmp_dir_path = os.path.join(data_dir_path, "tmp/")
    os.makedirs(tmp_dir_path, exist_ok=True)
    # Create initial subclusters
    if not state["subcluster_dir"] or not state["rep_seq_dir"] or not skip:
        skip = False
        subcluster_dir = os.path.join(data_dir_path, "subclusters/")
        rep_seq_dir = os.path.join(data_dir_path, "representative_sequences/")
        logger.info("Creating initial subclusters")
        dp.orthologs_to_subclusters(
            ortholog_fasta_dir=input_fasta_dir_path,
            output_subcluster_dir=subcluster_dir,
            output_rep_seq_dir=rep_seq_dir,
            tmp_dir_path=tmp_dir_path,
            n_processes=n_processes,
            create_subclusters=create_subclusters,
            mmseqs_cluster_coverage=mmseqs_cluster_coverage,
            mmseqs_cluster_identity=mmseqs_cluster_identity,
            mmseqs_coverage_subclusters=mmseqs_coverage_subclusters,
            samples_profiles_product_limit=samples_profiles_product_limit,
            mmseqs_n_processes=mmseqs_n_processes,
        )
        state["subcluster_dir"] = subcluster_dir
        state["rep_seq_dir"] = rep_seq_dir
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing subclusters")
        subcluster_dir = state["subcluster_dir"]
        rep_seq_dir = state["rep_seq_dir"]

    # Augment subclusters
    if not state["augmented_subcluster_dir"] or not skip:
        skip = False
        logger.info("Augmenting subclusters")
        augmented_subcluster_dir = os.path.join(data_dir_path, "augmented_subclusters/")
        dp.augment_small_subclusters(
            subclusters_dir=subcluster_dir,
            output_dir=augmented_subcluster_dir,
            tmp_dir=tmp_dir_path,
            min_sequences_per_subcluster=6,
            n_processes=n_processes,
        )
        state["augmented_subcluster_dir"] = augmented_subcluster_dir
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing augmented subclusters")
        augmented_subcluster_dir = state["augmented_subcluster_dir"]

    # Create full profile database
    if not state["full_profile_dir"] or not skip:
        skip = False
        full_profile_dir = os.path.join(data_dir_path, "subcluster_profiles/")
        logger.info("Creating full profiles")
        dp.create_subcluster_profiles(
            subcluster_dir=augmented_subcluster_dir,
            profile_dir=full_profile_dir,
            tmp_dir_path=tmp_dir_path,
            n_processes=n_processes,
        )
        state["full_profile_dir"] = full_profile_dir
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing full profile directory")
        full_profile_dir = state["full_profile_dir"]
    if not state["full_input_fasta"] or not skip:
        skip = False
        logger.info("Creating full input fasta")
        full_hmmsearch_input = os.path.join(data_dir_path, "full_hmmsearch_input.fasta")
        dp.prepare_full_hmmsearch_input(
            data_dir_path=data_dir_path,
            augmented_subcluster_dir=augmented_subcluster_dir,
            tmp_dir_path=tmp_dir_path,
            input_unknown_sequences_fasta_path=unknown_sequences_fasta_path,
            output_full_hmmsearch_input_path=full_hmmsearch_input,
            number_of_sampled_sequences_per_subcluster=number_of_sampled_sequences_per_subcluster,
            fraction_of_sampled_unknown_sequences=fraction_of_sampled_unknown_sequences,
            samples_profiles_product_limit=samples_profiles_product_limit,
        )

        state["full_input_fasta"] = full_hmmsearch_input
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing full input fasta")
        full_hmmsearch_input = state["full_input_fasta"]

    if not state["full_hmmsearch_results"] or not skip:
        logger.info("Running full hmmsearch")
        full_hmmsearch_results = os.path.join(
            data_dir_path, "full_hmmsearch_results.txt"
        )
        if not (skip and state["hmmsearch_tmp_path"]):
            hmmsearch_tmp_path = os.path.join(data_dir_path, "hmmsearch/")
            os.makedirs(hmmsearch_tmp_path, exist_ok=True)
            state["hmmsearch_tmp_path"] = hmmsearch_tmp_path
            yaml.dump(state, open(state_file_path, "w+"))
        else:
            hmmsearch_tmp_path = state["hmmsearch_tmp_path"]

        dp.full_hmmsearch(
            input_full_profiles_dir_path=full_profile_dir,
            input_fasta_path=full_hmmsearch_input,
            output_full_hmmsearch_results_path=full_hmmsearch_results,
            tmp_dir_path=hmmsearch_tmp_path,
            n_processes=n_processes,
            no_filter=False,
        )
        state["full_hmmsearch_results"] = full_hmmsearch_results
        yaml.dump(state, open(state_file_path, "w+"))
        skip = False
    else:
        logger.info("Using existing full hmmsearch results")
        full_hmmsearch_results = state["full_hmmsearch_results"]

    if (
        not (
            state["subcluster_split_fastas_dir"]
            and state["subcluster_split_scoring_dir"]
            and state["split_subcluster_md"]
        )
        or not skip
    ):
        skip = False
        logger.info("Splitting subclusters")
        subcluster_split_fastas_dir = os.path.join(
            data_dir_path, "subcluster_split_fastas/"
        )
        subcluster_split_scoring_dir = os.path.join(
            data_dir_path, "subcluster_split_scoring/"
        )
        subcluster_split_md_path = os.path.join(
            data_dir_path, "split_subcluster_md.pkl"
        )
        dp.randomly_split_subclusters(
            input_subclusters_fasta_dir_path=augmented_subcluster_dir,
            output_fasta_dir_for_profiles_path=subcluster_split_fastas_dir,
            output_fasta_dir_for_scoring_path=subcluster_split_scoring_dir,
            output_path_md=subcluster_split_md_path,
        )

        state["subcluster_split_fastas_dir"] = subcluster_split_fastas_dir
        state["subcluster_split_scoring_dir"] = subcluster_split_scoring_dir
        state["split_subcluster_md"] = subcluster_split_md_path
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing split subclusters")
        subcluster_split_fastas_dir = state["subcluster_split_fastas_dir"]
        subcluster_split_scoring_dir = state["subcluster_split_scoring_dir"]
        subcluster_split_md_path = state["split_subcluster_md"]

    if not state["subcluster_split_profiles_dir"] or not skip:
        skip = False
        logger.info("Creating split subcluster profiles")
        subcluster_split_profiles_dir = os.path.join(
            data_dir_path, "subcluster_split_profiles/"
        )
        dp.create_subcluster_profiles(
            subcluster_dir=subcluster_split_fastas_dir,
            profile_dir=subcluster_split_profiles_dir,
            tmp_dir_path=tmp_dir_path,
            n_processes=n_processes,
        )
        state["subcluster_split_profiles_dir"] = subcluster_split_profiles_dir
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing split subcluster profiles")
        subcluster_split_profiles_dir = state["subcluster_split_profiles_dir"]

    if not state["split_hmmsearch_results"] or not skip:
        split_hmmsearch_tmp_dir_path = os.path.join(data_dir_path, "split_hmmsearch/")
        os.makedirs(split_hmmsearch_tmp_dir_path, exist_ok=True)
        logger.info("Running split hmmsearch")
        skip = False
        split_hmmsearch_results_path = os.path.join(
            data_dir_path, "split_hmmsearch_results.txt"
        )
        dp.split_profiles_hmmsearch(
            input_split_profiles_dir_path=subcluster_split_profiles_dir,
            input_fasta_dir_for_scoring_path=subcluster_split_scoring_dir,
            output_split_hmmsearch_results_path=split_hmmsearch_results_path,
            tmp_path=split_hmmsearch_tmp_dir_path,
            n_processes=n_processes,
        )
        state["split_hmmsearch_results"] = split_hmmsearch_results_path
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing split hmmsearch results")
        split_hmmsearch_results_path = state["split_hmmsearch_results"]

    if not state["ground_truth"] or not skip:
        logger.info("Creating ground truth")
        ground_truth_path = os.path.join(data_dir_path, "ground_truth.pkl")
        skip = False
        dp.generate_ground_truth(
            input_augmented_subclusters_fastas_dir_path=augmented_subcluster_dir,
            input_unannotated_sequences_fasta_path=unknown_sequences_fasta_path,
            output_ground_truth_path=ground_truth_path,
        )
        state["ground_truth"] = ground_truth_path
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        ground_truth_path = state["ground_truth"]
        logger.info("Using existing ground truth")

    if not state["sdf_train"] or not skip:
        logger.info("Creating sparse dataframe for training")

        sdf_train_path = os.path.join(data_dir_path, "sdf_train.json")
        skip = False
        dp.hmmsearch_results_to_train_sdf(
            input_split_hmmsearch_results_path=split_hmmsearch_results_path,
            input_split_subcluster_md_path=subcluster_split_md_path,
            input_full_hmmsearch_results_path=full_hmmsearch_results,
            input_full_profiles_dir_path=full_profile_dir,
            input_all_sequences_fasta_path=full_hmmsearch_input,
            input_ground_truth_path=ground_truth_path,
            output_sdf_path=sdf_train_path,
        )
        state["sdf_train"] = sdf_train_path
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing sparse dataframe for training")
        sdf_train_path = state["sdf_train"]

    if not state["sdfloader"] or not skip:
        logger.info("Creating sdfloader")
        skip = False
        sdfloader_path = os.path.join(data_dir_path, "sdfloader.pkl")
        prepare_sdfloader(
            sdf_train_path=sdf_train_path,
            n_processes=n_processes,
            triplets_per_class=3000,
            output_path=sdfloader_path,
            load_stack_size=100000,
        )
        state["sdfloader"] = sdfloader_path
        yaml.dump(state, open(state_file_path, "w+"))
    else:
        logger.info("Using existing sdfloader")
        sdfloader_path = state["sdfloader"]

    logger.info("Finished preprocessing")
