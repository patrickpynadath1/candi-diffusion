#!/bin/bash
steps=$2
temp=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=mdlm \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/openwebtext-train/text8-mdlm-1024/checkpoints/last.ckpt \
        model=small \
        model.length=1024 \
        algo.temp=$temp \
        sampling.steps=$steps;