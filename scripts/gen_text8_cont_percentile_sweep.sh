#!/bin/bash
steps=$2
temp=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/openwebtext-train/text8-pure-cont-percentile-1024/checkpoints/last.ckpt \
        algo.mixed_coeff=0 algo.min_percentile=.1 algo.max_percentile=.4\
        model=small \
        model.length=1024 \
        algo.pure_continuous=true;

