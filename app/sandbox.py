import pickle
from tqdm import tqdm
import os
import re
import subprocess
from app.data_preprocessing import get_fasta_ids

SPLIT_SUBCLUSTER_FASTA_SUFFIX_PATTERN = r"\.sub_cluster\.cluster\.d+\.d+\.fasta"

md_output_path = "new_md.pkl"
subcluster_splits_dir = "data/kegg_data_0.9/subcluster_split_fastas"
for fasta_path in tqdm(
    [os.path.join(subcluster_splits_dir, f) for f in os.listdir(subcluster_splits_dir)],
    desc="Processing subclusters",
):
    fasta_ids = get_fasta_ids(fasta_path)
    file_basename = os.path.basename(fasta_path)
    ortholog = re.sub(SPLIT_SUBCLUSTER_FASTA_SUFFIX_PATTERN, "", file_basename)
    parts = file_basename.split(".")
    subcluster = 