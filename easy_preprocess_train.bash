#!/bin/bash
#PBS -q dudulight
#PBS -N kegg2021_0.8
#PBS -l select=1:ncpus=64:mem=250gb
#PBS -oe
#PBS -V


if [ -z "$input_fasta_dir_path" ] || [ -z "$data_dir_path" ]; then
    echo "Usage: qsub -v input_fasta_dir_path=...,data_dir_path=...[,unknown_sequences_fasta_path=...] easy_preprocess_train.bash"
fi

if [ -z "$unknown_sequences_fasta_path" ]; then
    unknown_sequences_fasta_path=None
fi

cd /davidb/guyshur/famus/
export PATH=/davidb/guyshur/anaconda3/bin:$PATH
source activate famus
echo "Running easy_preprocess_train with input_fasta_dir_path=$input_fasta_dir_path, data_dir_path=$data_dir_path, unknown_sequences_fasta_path=$unknown_sequences_fasta_path" >> error.log
/davidb/guyshur/anaconda3/envs/famus/bin/python -m easy_preprocess_train --input_fasta_dir_path $input_fasta_dir_path --data_dir_path $data_dir_path --unknown_sequences_fasta_path $unknown_sequences_fasta_path
