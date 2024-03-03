import os
import subprocess
import multiprocessing as mp

job_id = int(os.environ["HMMSEARCH_JOB_ID"])


def _hmmsearch_single_profile(
    fasta_file_path: str,
    profiles_db_path: str,
    output_path: str,
    threads: int = 1,
) -> None:
    # run hmmsearch
    cmd = "hmmsearch -o /dev/null --tblout {} --cpu {} {} {}".format(
        output_path,
        str(threads),
        profiles_db_path,
        fasta_file_path,
    )

    subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)


def hmmsearch():
    profile_files = os.listdir("data/kegg_data_0.9/subcluster_profiles")
    if job_id == 1:
        profile_files = [f for f in profile_files if f[1] == "0" and int(f[2]) < 9]
    elif job_id == 2:
        profile_files = [
            f
            for f in profile_files
            if (f[1] == "0" and int(f[2]) >= 9) or f[1] == "1" and int(f[2]) < 8
        ]
    elif job_id == 3:
        profile_files = [
            f for f in profile_files if (f[1] == "1" and int(f[2]) >= 8) or f[1] == "2"
        ]
    else:
        raise ValueError("Invalid job_id")
    profile_files = [
        os.path.join("data/kegg_data_0.9/subcluster_profiles", f) for f in profile_files
    ]
    pool = mp.Pool(64)
    # fasta_files = [
    #     "data/kegg_data_0.9/augmented_subclusters/"
    #     + os.path.basename(f).removesuffix(".hmm")
    #     + ".fasta"
    #     for f in profile_files
    # ]
    fasta_files = ["data/kegg_data_0.9/full_hmmsearch_input.fasta"] * len(profile_files)
    output_path = "data/kegg_data_0.9/tmp/hmmsearch/"
    output_paths = [
        os.path.join(
            output_path, os.path.basename(f).removesuffix(".hmm") + ".hmmsearch"
        )
        for f in profile_files
    ]
    pool.starmap(
        _hmmsearch_single_profile,
        zip(
            fasta_files,
            profile_files,
            output_paths,
        ),
    )
    pool.close()
    pool.join()


if __name__ == "__main__":
    hmmsearch()
