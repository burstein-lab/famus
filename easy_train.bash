#!/bin/bash
#PBS -q duduheavy
#PBS -N easy_train_kegg2023_512x3
# -l select=1:ncpus=4:mem=250gb:ngpus=1:host=compute-0-419
#PBS -l select=1:ncpus=64:mem=250gb
#PBS -oe
#PBS -V

export CUDA_VISIBLE_DEVICES=1

if [ -z "$sdfloader_path" ] || [ -z "$output_path" ]; then
    echo "Usage: qsub -v sdfloader_path=...,output_dir...[,num_epochs=...,batch_size=...] easy_train.bash"
fi

if [ -z "$num_epochs" ]; then
    num_epochs=10
fi

if [ -z "$batch_size" ]; then
    batch_size=32
fi

cd /davidb/guyshur/famus/
export PATH=/davidb/guyshur/anaconda3/bin:$PATH
source activate famus
/davidb/guyshur/anaconda3/envs/famus/bin/python -m easy_train --sdfloader_path $sdfloader_path --output_dir $output_dir --num_epochs $num_epochs --batch_size $batch_size
