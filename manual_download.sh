#!/bin/bash

# Create the directory
mkdir -p /data/remote_cache/patrick/discrete_diffusion/openwebtext-train/2025.10.24/153455/subsets/

# Change to that directory
cd /data/remote_cache/patrick/discrete_diffusion/openwebtext-train/2025.10.24/153455/subsets/

# Download all 21 tar files from HuggingFace
for i in {0..20}; do
    filename=$(printf "urlsf_subset%02d.tar" $i)
    echo "Downloading $filename..."
    wget "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/$filename"
done
