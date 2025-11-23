# Content: Configuration file for the training and classification pipelines
# See README.md for more information about each parameter

n_processes: 32 # positive integer
user_device: 'cuda' # 'cpu' or 'cuda'
logging: True # True or False

# the following parameters are only relevant for classification

models_type: "light" # 'light' or 'full'

chunksize: 20000 # positive integer
batch_size: 32
num_epochs: 10

# the following parameters are only relevant for the training preprocessing pipeline

processes_per_mmseqs_job: 4 # positive integer
sampled_sequences_per_subcluster: 60 # 'use_all' or positive integer
fraction_of_sampled_unknown_sequences: 1.0 # 'use_all', 'do_not_use', or 0 <= float <= 1.0
samples_profiles_product_limit: 150_000_000_000_000 # positive integer
create_subclusters: True # True (creates full model) or False (creates light model)
mmseqs_cluster_coverage: 0.8 # 0 <= float <= 1.0
mmseqs_cluster_identity: 0.9 # 0 <= float <= 1.0
mmseqs_cluster_coverage_subclusters: 0.5 # 0 <= float <= 1.0
max_fasta_n_sequences_times_longest_sequence: 500_000_000 # positive integer
save_every: 100000 # positive integer
log_to_wandb: False # True or False
wandb_project: 'famus'
wandb_api_key_path: "wandb_api_key.txt"


default_config = [
    ["tmp_dir", "~/.famus/tmp", "Temporary directory for famus"],
    ["logging", True, "Enable logging"],
    ["n_processes", 4, "Number of processes to use for parallel tasks"],
    ["models_type", "light", "Type of models to train or classify with: 'light' or 'full'"],
    ["user_device", "cpu", "Device to use: 'cpu' or 'cuda'"],
    ["chunksize", 20000, "Number of sequences to process in each chunk during classification"],
    ["batch_size", 32, "Training batch size"],