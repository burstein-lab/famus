# FAMUS: Functional Annotation Method Using Siamese neural networks

FAMUS is a Siamese Neural Network (SNN) based framework that annotates protein sequences with function. Input sequences are transformed to numeric vectors with pre-trained neural networks tailored to individual protein family databases, and then compared to sequences of those databases to find the closest match.

This repository can also be used to train a model for any protein database by using one fasta file for each protein family, and preferrably a large number of negative examples (sequences not belonging to any family). 

We provide one main module for training (easy_train) and one for classification (easy_classify), which automatically take care of all relevant steps of training and/or inference. If interrupted, using these modules again with the same parameters will attempt to resume from where the program has stopped. For this reason do not rename/remove files from the directories the modules are using if you intend to restart an interrupted pipeline using the same data directory, as the names of some files are hardcoded into the program.

Check out famus.ipynb for a more detailed explanation of the modules and their usage.

## Installation

### Manual installation

To install with conda, run the following commands:

`conda env create -f environment.yml -n famus`

Activate the environment with `conda activate famus` and install the correct pytorch version for your environment from the pytorch website via pip: https://pytorch.org/get-started/locally/

### Docker installation

Alternatively, you can use the provided Dockerfile to build a Docker image. To do so, run the following command: ` docker build -t famus .`

### Apptainer/Singularity installation

If you are using Apptainer or Singularity, you can use the provided Singularity recipe file to build a Singularity image. To do so, run the following command: `<singularity/apptainer> build famus.sif famus.def`

## Classifying sequences

To classify sequences, you will need a fasta format file of sequences to classify.

## Training a model

Please read the entirety of the following section before starting!

If you want to train your own model with a collection of fasta files representing protein families, you will need:
 - A directory containing all and only the fasta files (one for each protein family)
 - A fasta file with diverse fasta sequences that do not belong to any of your protein families (optional, but highly recommended unless you are certain all the sequences you wish to classify belong to one of your input protein families)
 
Due to how the scripts are written, please adhere to the following:
- The input fasta file directory must contain all (and only) the relevant amino acid sequence fasta files.
- All fasta file names must end in .fasta.
- Each fasta file should be named in accordance with the label you want to represent its sequences, e.g. sequences matched to the protein family ABC_TRANSPORTER.fasta will be labeled as 'ABC_TRANSPORTER'.
- Avoid naming any file 'unknown.fasta' - 'unknown' is the label used for an uncertain protein family.

### Categories of sequences

This tool uses the input fasta files to create HMM profiles of each protein family, but also to acquire bit-scores of the files' sequences on each profile. Depending on the size of the training set, this bit-score acquisition part (hmmsearch) of the preprocessing phase can be a significant bottleneck in terms of runtime. To partially remedy this, it is possible to downsample the sequences that will be used to generate bit-scores. In this case we distinguish between three types of sequences when sampling fasta files to use for training a model:

- subcluster sequences: representative sequences of each protein family that end up in subclusters.
- non-subcluster known sequences: sequences that belong to a protein family but filtered out of participation in a subcluster.
- unknown / unannotated sequences: sequences that do not belong to a portein family to be incorporated in the model.

To control the level of sampling of each category, we use the following parameters in the cfg.yaml file:

- number_of_sampled_sequences_per_subcluster: 'use_all' will disable downsampling and use all of this category's sequences. A positive integer will make the tool sample up to that many sequences from each subcluster (or less if the subcluster is smaller than the integer). 
- number_of_sampled_non_subcluster_sequences: 'use_all' will disable downsampling and use all of this category's sequences. A positive integer will make the tool sample up to this many total sequences randomly.
- number_of_sampled_unknown_sequences: 'use_all' will disable downsampling and use all of this category's sequences. A positive integer will make the tool sample up to this many total sequences randomly. 'do_not_use' will disable sampling of this category altogether. 

**Please note that it is highly recommended to provide a significant number of negative samples for the model,** preferrably at around a 1:1 ratio with the rest of the sampled sequences, for two reasons:
- Unannotated sequences are used as possible nearest neighbors in the classification step - meaning that if their embedding is closest to an input sequence for classification, it will be marked as unknown.
- Unannotated sequences are used as false positives to estimate the maximal distance from a sample that can be considered a nearest neighbor. If unannotated sequences are not provided, this threshold will be estimated using cross-label near samples, which is less accurate.

### How to train a model

To preprocess your data and create the model's input, use the following script:

```
python -m easy_preprocess_train --input_fasta_dir_path data/example/example_orthologs/ --data_dir_path data/example/example_training/ [--unknown_sequences_fasta_path data/example/example_unannotated_sequences.fasta] [--nthreads NTHREADS]
```

After the preprocessing phase has finished, you can train the model:

```
python -m easy_train --sdfloader_path data/example/example_training/sdfloader.pkl --output_path data/example/example_training/model.pt [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--nthreads NTHREADS] [--device DEVICE]
```
See the section about cfg.yaml for details about the device parameter.
