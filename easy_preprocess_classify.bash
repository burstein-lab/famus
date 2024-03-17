#!/bin/bash
#PBS -q duduheavy
#PBS -N easy_preprocess_classify
#PBS -l select=1:ncpus=64:mem=250gb
#PBS -oe
#PBS -V


if [ -z "$input_fasta_file_path" ] || 
    [ -z "$input_full_profiles_dir_path" ] || 
    [ -z "$input_sdf_train_path" ] || 
    [ -z "$data_dir_path" ]; then
    echo "Usage: qsub -v input_fasta_file_path=...,input_full_profiles_dir_path=...,input_sdf_train_path=...,data_dir_path=...[,nthreads=...] easy_preprocess_classify.bash"
fi

if [ -z "$nthreads" ]; then
    nthreads=64
fi

cd /davidb/guyshur/famus/
module load singularity/singularity-3.2.1
singularity exec --bind .:/app --pwd /app appsing python \
-m easy_preprocess_classify \
--input_fasta_file_path $input_fasta_file_path \
--input_full_profiles_dir_path $input_full_profiles_dir_path \
--input_sdf_train_path $input_sdf_train_path \
--data_dir_path $data_dir_path \
--nthreads $nthreads

