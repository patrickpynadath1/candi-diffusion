#!/bin/bash

steps=$2
temp=$3

CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
        model=small \
        model.length=1024 \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/openwebtext-train/text8-candi2-1024/checkpoints/last.ckpt \
        algo.temp=$temp \
        sampling.steps=$steps;
