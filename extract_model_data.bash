#!/bin/bash

set -e

cleanup() {
  echo "Script interrupted, exiting cleanly..."
  exit 1
}

trap cleanup SIGINT SIGTERM

types=("full" "light")
models=("kegg" "orthodb" "interpro" "eggnog")

for type in "${types[@]}"; do
  for model in "${models[@]}"; do
    echo "Processing ${type}_${model}..."
    
    mkdir -p models/${type}
    
    if [ -f "${type}_${model}.tar" ]; then
      cp "${type}_${model}.tar" models/${type}/
    else
      echo "Warning: ${type}_${model}.tar not found"
      continue
    fi
    
    cd models/${type}/
    
    tar -xf "${type}_${model}.tar"
    
    mkdir -p ${model}/data_dir
    
    cd ${model}
    
    if [ "$type" == "light" ]; then
      SDF_FILE="${model}_sdf_light.json.gz"
    else
      SDF_FILE="${model}_sdf.json.gz"
    fi
    
    if [ -f "$SDF_FILE" ]; then
      gunzip -c "$SDF_FILE" > "data_dir/sdf_train.json"
      rm "$SDF_FILE"
    fi
    
    if [ -f "train_embeddings.npy.gz" ]; then
      gunzip -c train_embeddings.npy.gz > data_dir/train_embeddings.npy
      rm train_embeddings.npy.gz
    fi
    
    if [ -f "state.pt.gz" ]; then
      gunzip -c state.pt.gz > state.pt
      rm state.pt.gz
    fi
    
    if [ -f "subcluster_profiles.tar.gz" ]; then
      mv subcluster_profiles.tar.gz data_dir/
      cd data_dir
      tar -xzf subcluster_profiles.tar.gz
      rm subcluster_profiles.tar.gz
      cd ..
    fi
    
    cd ../../..
    
    if [ $? -eq 0 ]; then
      echo "Processing completed successfully, removing original tar file"
      rm "${type}_${model}.tar"
      rm "models/${type}/${type}_${model}.tar"
    else
      echo "Warning: Processing may not have completed successfully for ${type}_${model}"
    fi
    
    echo "Finished processing ${type}_${model}"
  done
done

echo "All processing complete!"