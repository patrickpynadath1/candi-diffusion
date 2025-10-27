#!/bin/bash
steps=$2
temp=$3
# CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
#         eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.10.15/150356/checkpoints/best.ckpt \
#         algo.is_embed=True \
#         algo.mixed_coeff=0 algo.min_percentile=.05 algo.max_percentile=.35\
#         algo.temp=$temp \
#         algo.pure_continuous=true sampling.steps=$steps;

CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/openwebtext-train/text8-pure-cont-embed-1024/checkpoints/last.ckpt \
        algo.is_embed=True \
        algo.mixed_coeff=0 \
        model=small \
        model.length=1024 \
        algo.min_percentile=.05 \
        algo.max_percentile=.4\
        algo.temp=$temp \
        algo.pure_continuous=true sampling.steps=$steps;

