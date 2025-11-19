# FAMUS: Functional Annotation Method Using Siamese neural networks

FAMUS has a web interface available at https://famus-6e94e.web.app/.  

FAMUS is a Siamese Neural Network (SNN) based framework that annotates protein sequences with function. Input sequences are transformed to numeric vectors with pre-trained neural networks tailored to individual protein family databases, and then compared to sequences of those databases to find the closest match.

This repository (or the famus conda package) can also be used to train a model for any protein database by using one fasta file for each protein family, and preferrably a large number of negative examples (sequences not belonging to any family in the given database). 

We provide one main module for training and one for classification, which automatically take care of all relevant steps of training and/or inference. If interrupted, using these modules again with the same parameters will attempt to resume from where the program has stopped. For this reason do not rename/remove files from the directories the modules are using if you intend to restart an interrupted pipeline using the same data directory, as the names of some files are hardcoded into the program.

Check out famus.ipynb for a more detailed explanation of the modules and their usage.

Currently FAMUS only supports linux and macOS operating systems.

## Installation

### With Conda

To install with conda, first create a new conda environment:

`conda create -n famus`

Activate the environment with `conda activate famus` and install the correct pytorch version for your environment from the pytorch website via pip: https://pytorch.org/get-started/locally/

Finally, install famus with
`conda insall -c conda-forge famus`

Using famus tools requires that the conda environment is activated. You can check if the installation was successful by running `famus-train -h`.

### From source

Alternatively, you can download the famus source code. First, clone the repository:

`git clone https://github.com/burstein-lab/famus.git`

Without conda, FAMUS has five dependencies (other than python and pip) that need to be installed separately:

- PyTorch - follow the instructions at https://pytorch.org/get-started/locally/
- mmseqs2
- seqkit
- hmmer
- mafft

Make sure that the executables for `mmseqs2`, `seqkit`, `hmmsearch` from hmmer, and `mafft` are all in your PATH variable.

Create and activate a new conda or pip virtual environment, then install the required python packages with:
`pip install -r requirements.txt`


### Downloading pre-trained models

If you plan on using the pre-trained models, you will need to download them from Zenodo (https://zenodo.org/uploads/14941373). The available pre-trained models are:  

```
kegg_comprehensive
kegg_light
interpro_comprehensive
interpro_light
orthodb_comprehensive
orthodb_light
eggnog_comprehensive
eggnog_light
```

To easily download pre-trained models, we provide a command line tool called `famus-install` (for conda). source code users will use tbe module famus.cli.install_models. This will download a large number of profile HMMs, so make sure you have enough disk space (several GBs depending on the models you download).

If installed with conda, run `famus-install --models <space-separated list of model names> --models-dir <path to models directory>`. For example, to download the comprehensive KEGG and light InterPro models to famus_models, run:
`famus-install --models kegg_comprehensive interpro_light --models-dir famus_models`. If using the source code, run `python -m famus.cli.install_models` from the root directory instead of `famus-install`. See details below for a comprehensive list of command line arguments.

Some python data is downloaded as JSON for security reasons. After running this command, it is recommended (but optional) to convert the data that was downloaded as JSON to pickle format for faster data loading. This can be done by running the following command: `python -m convert_sdf`

## Configuration and priority of parameters

Most FAMUS tools expect parameters as either command line arguments or in a given (optional) configuration file. An example for a configuration file can be found in `example_cfg.yaml` of this repository's root directory. The order of priority is as follows:
1. Command line arguments
2. Configuration file parameters (if provided as a command line argument and the relevant parameter is specified there)
3. Default parameters (running `famus-defaults` (conda) or `python -m famus.config` (source code) will print the default parameters to the console)

## Classifying sequences

To classify sequences, you will need a fasta format file of sequences to classify.

The main tool for classification is `famus-classify` for conda and `famus.cli.classify` for source code users.
Usage:
 - conda: `famus-classify [options] <input_fasta_file_path> <output_dir>`
 - source code: `python -m famus.cli.classify [options] <input_fasta_file_path> <output_dir>`

Main command line arguments for `famus-classify` (unused arguments will be read from config or set to default values):
- input_fasta_file_path - the path of the sequeces for classification. (required)
- output_dir - the directory to save the results to. (required)
- --config - path to configuration file.
- --n-processes - number of cpu cores to use.
- --device - cpu/cuda
- --models - space-separated list of model names to use. 
- --models-dir - directory where the models are installed.
- --models-type - comprehensive/light - type of model to use (light may be slightly less accurate but significantly faster).
- --load-sdf-from-pickle - loads training data from pickle instead of json. Only usable after running `python -m convert_sdf`.
- --no-log - do not create a log file.
- --log-dir - directory to save the log file to.

## Training a model

Training a model on a large database can take a long time and be computationally expensive. It is recommended to be faniliar with the options in the configuration file before starting training.

The main tools for training are `famus-train` for conda, and `famus.cli.train` for source code users.

Usage:
 - conda: `famus-train [options] <input_fasta_dir_path>`
 - source code: `python -m famus.cli.train [options] <input_fasta_dir_path>`

**Important notes:**
 - every file name in the input directory **must** end in .fasta, and files must not be named unknown.fasta (since unknown is reserved for unknown sequences).
 - It is recommended to provide a fasta file of unknown sequences (sequences not belonging to any family in the database) as negative examples for training. This will reduce false positives during classification.

Main command line arguments for `famus-train` (unused arguments will be read from config or set to default values):
- input_fasta_dir_path - the path of the directory holding fasta files where each file defines a protein family (required).
- --config - path to configuration file.
- --create-subclusters / --no-create-subclusters - whether to create a comprehensive (True) or light (False) model. Comprehensive models cluster protein families into sub-families, which increases accuracy but also training and classification time.
- --model-name - optional name for the model. If not specified, the input directory base name will be used.
- --unknown-sequences-fasta-path - fasta file with sequences of unknown function as negative examples for the model. Optional but recommended.
- --n-processes - number of CPU cores to use.
- --num-epochs - number of epochs to train the model for.
- --batch-size - training batch size.
- --stop-before-training - calling easy_train with --stop_before_training will exit before starting to train the model (useful for things like preprocessing in a high-CPU environment and them training the model in a different environment with CUDA).
- --device - cpu/cuda.
- --chunksize - reduce if GPU RAM becomes an issue when calculating threshold using GPU.
- --save-every - save a checkpoint of the model's state every \<save_every> steps. Will load the last checkpoint automatically if the script is restarted.

## Comprehensive list of configuration parameters

Training and classification parameters:
- --n-processes: number of processes to use for parallelization during preprocessing and cpu-based training and classification
- --user-device: 'cpu' or 'cuda' - the device to use for training and classification. Classification with GPU is only marginally faster within HPC environments.
- --no-log: do not create a log file.
- --log-dir: directory to save the log file to.
- --models-dir: directory where the models are installed.
- --load-sdf-from-pickle: whether to load training data from pickle files instead of json files. Only usable after running `python -m convert_sdf`.

Classification-specific parameters:

- --models-type: 'comprehensive' or 'light' - type of model to use for classification (light may be slightly less accurate but significantly faster).
- --models: a space-separated list of protein family databases to use for training or classification. The available pretrained are: kegg, interpro, orthodb, eggnog for both comprehensive and light models. Classification will use all models specified here for the specified model type.
- --chunksize: positive integer - the number of sequences to process (load to GPU) in each batch during classification - decrease if running out of GPU RAM.
- --batch-size: positive integer - the batch size to use for training.
- --num-epochs: positive integer - the number of epochs to train for.

Training-specific parameters:
- --create-sublucsters/--no-create-subclusters: whether to create a comprehensive or light model. Comprehensive models cluster protein families into sub-families, which increases accuracy but also training and classification time.
- --processes-per-mmseqs-job: positive integer - the number of processes to use for each mmseqs job during preprocessing. Higher values will work faster for fewer but bigger protein families, lower values will work faster for many small protein families.
- --number-of-sampled-sequences-per-subcluster: 'use_all' or positive integer - the number of sequences to sample from each subcluster during preprocessing. If 'use_all', all sequences will be used. These sequences will be used to train the model and as positive examples during classification. decrease this value to reduce preprocessing time and space usage, increase to improve training data variety.
- --fraction-of-sampled-unknown-sequences: 'use_all', 'do_not_use', or 0 <= float <= 1.0 - the fraction of unknown sequences to sample relative to the number of labeled seuqneces that were sampled (e.g, 1.0 will sample up to the same number of unknown sequences as total labeled sequences). If 'use_all', all unknown sequences will be used (not recommended if the number of unknowns is much higher). These sequences will be used as negative examples during training and classification.
- --samples-profiles-product-limit: positive integer - if the number of protein families (in light models) or sub-families (in comprehensive models) times the number of sampled sequences per subcluster exceeds this limit, the number of sampled sequences per subcluster will be reduced to stay below the limit. This is to avoid extremely long processing times.
- --mmseqs-cluster-coverage: float between 0 and 1 - mmseqs clustering coverage parameter during deduplication of protein families. Higher values will de-duplicate less aggressively.
- --mmseqs-cluster-identity: float between 0 and 1 - mmseqs clustering identity parameter during deduplication of protein families. Higher values will de-duplicate less aggressively.
- --mmseqs-coverage-subclusters: float between 0 and 1 - mmseqs coverage parameter during creation of subclusters within protein families. Higher values will create more and smaller subclusters.



