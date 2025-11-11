# FAMUS: Functional Annotation Method Using Siamese neural networks

FAMUS has a web interface available at https://famus-6e94e.web.app/.  

FAMUS is a Siamese Neural Network (SNN) based framework that annotates protein sequences with function. Input sequences are transformed to numeric vectors with pre-trained neural networks tailored to individual protein family databases, and then compared to sequences of those databases to find the closest match.

This repository can also be used to train a model for any protein database by using one fasta file for each protein family, and preferrably a large number of negative examples (sequences not belonging to any family). 

We provide one main module for training (easy_train) and one for classification (easy_classify), which automatically take care of all relevant steps of training and/or inference. If interrupted, using these modules again with the same parameters will attempt to resume from where the program has stopped. For this reason do not rename/remove files from the directories the modules are using if you intend to restart an interrupted pipeline using the same data directory, as the names of some files are hardcoded into the program.

Check out famus.ipynb for a more detailed explanation of the modules and their usage.

## Installation

### Installing dependencies

To install with conda, run the following commands:

`conda env create -f environment.yml -n famus`

Activate the environment with `conda activate famus` and install the correct pytorch version for your environment from the pytorch website via pip: https://pytorch.org/get-started/locally/

### Downloading pre-trained models

If you plan on using the pre-trained models, you will need to download them from Zenodo (https://zenodo.org/uploads/14941373).

To download all model data from Zenodo, run the following command: `bash download_model_data.bash`.   
 - Alternatively you can manually download only the TAR files for the models you are interested in and place them in the top level of the repository. Verify that the TAR files are named correctly (e.g. full_kegg.tar, as they appear in Zenodo).

To extract the pre-trained models, run the following command: `bash extract_model_data.bash`

After running these commands, it is recommended to convert some of the data that was downloaded as JSON to pickle format for faster loading. This can be done by running the following command: `python -m convert_sdf`

## Classifying sequences

To classify sequences, you will need a fasta format file of sequences to classify.

The main module for classification is `easy_classify.py`. It can be run with the following command: `python -m easy_classify --input_fasta_file_path /path/to/file --output_dir /path/to/output_dir` 

Command line arguments for easy_classify (unused arguments will be read from cfg.yaml):
- input_fasta_file_path - the path of the sequeces for classification. (required)
- output_dir - the directory to save the results to. (required)
- n_processes - number of cpu cores to use.
- device - cpu/cuda - in HPC environments with multiple CPU cores, there isn't a real difference.
- chunksize - how many sequences to classify per iteration. Decrease if GPU RAM becomes an issue (default is 20,000).
- models - space-separated list of model names to use. 
- models_type - full/light - type of model to use (light is slightly less accurate but significantly faster).
- load_sdf_from_pickle - loads training data from pickle instead of json. Only usable after running `python -m convert_sdf`.

## Training a model

Training a model on a large database can take a long time and be computationally expensive. It is recommended to be faniliar with the options in the configuration file before starting training.

The main module for training is `easy_train.py`. It can be run with the following command: `python -m easy_train --input_fasta_dir_path /path/to/directory`. **Note:** every file name in the input directory **must** end in .fasta, and files must not be named unknown.fasta (since unknown is reserved for unknown sequences).

Command line arguments for easy_train (unused arguments will be read from cfg.yaml):
- input_fasta_dir_path - the path of the directory holding fasta files where each file defines a protein family (required).
- model_type - full/light. The type of model to create - full models take longer to train and classify but are slightly more accurate.
- model_name - optional name for the model that will be used to refer to it in easy_classify. If not specified, the input directory base name will be used.
- unknown_sequences_fasta_path - fasta file with sequences of unknown function as negative examples for the model. Optional but recommended.
- n_processes - number of CPU cores to use.
- num_epochs - number of epochs to train the model for.
- batch_size - training batch size.
- stop_before_training - calling easy_train with --stop_before_training will exit before starting to train the model (useful for things like preprocessing in a high-CPU environment and them training the model in a different environment with CUDA).
- device - cpu/cuda.
- chunksize - reduce if GPU RAM becomes an issue when calculating threshold using GPU.
- save_every - save a checkpoint of the model's state every \<save_every> steps. Will load the last checkpoint automatically if the script is restarted.

## Configuration

cfg.yaml contains the configuration for the training and classification modules. Optional command line arguments not specified when running the modules will be taken from the configuration file.

The following parameters can be set in the configuration file:

- n_processes: number of processes to use for parallelization during preprocessing and cpu-based training and classification
- user_device: 'cpu' or 'cuda' - the device to use for training and classification
- logging: True or False - whether to create a log file in tmp/logs/ with information about the training or classification process
- models_type: 'light' or 'full' - whether to classify using full models or light models. Light models do not cluster protein families into sub-families.
- models: a list of protein family databases to use for training or classification. The available pretrained are: kegg, interpro, orthodb, eggnog. Classification will use all models specified here.
- chunksize: positive integer - the number of sequences to process in each batch during classification - decrease if running out of memory.
- threshold: 'bootstrap' or positive float - the distance threshold for classifying sequences - not relevant when using easy_train or easy_classification, mainly used for development purposes. When using easy_train, the threshold will be automatically calculated and saved, and automatically used in easy_classify.
- batch_size: positive integer - the batch size to use for training.
- num_epochs: positive integer - the number of epochs to train for.
- processes_per_mmseqs_job: positive integer - the number of processes to use for each mmseqs job during preprocessing.
- number_of_sampled_sequences_per_subcluster: 'use_all' or positive integer - the number of sequences to sample from each subcluster during preprocessing. If 'use_all', all sequences will be used. These sequences will be used to train the model and as positive examples during classification.
- fraction_of_sampled_unknown_sequences: 'use_all', 'do_not_use', or 0 <= float <= 1.0 - the fraction of unknown sequences to sample relative to the number of labeled seuqneces that were sampled. If 'use_all', all unknown sequences will be used. These sequences will be used as negative examples during training and classification.
- samples_profiles_product_limit: positive integer - if the number of protein families / sub-families times the number of sampled sequences per subcluster exceeds this limit, the number of sampled sequences per subcluster will be reduced to stay below the limit. This is to avoid extremely long training times.
- create_subclusters: True or False - whether to create a full (comprehensive) model or a light model. The only difference is that the light model does not cluster protein families into sub-families.
- mmseqs_cluster_coverage: 0 <= float <= 1.0 - the coverage threshold for deduplication of protein families.
- mmseqs_cluster_identity: 0 <= float <= 1.0 - the identity threshold for deduplication of protein families.
create_subclusters: True # True (creates full model) or False (creates light model)
- mmseqs_cluster_coverage_subclusters: 0 <= float <= 1.0 - the coverage threshold for clustering protein families into sub-families.
- max_fasta_n_sequences_times_longest_sequence: positive integer - if create_subclusters is set to false, but the number of sequences times the length of the longest sequence for a particular protein family exceeds this limit, the protein family will be split into subclusters anyway. This is to avoid doing multiple sequence alignments on very large protein families.
- save_every: positive integer - how often (in number of steps) to create a checkpoint during training.




