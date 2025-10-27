#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py --config-name=config_text8 mode=sample_eval algo=candi \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/openwebtext-train/text8-pure-cont-embed-1024/checkpoints/last.ckpt \
        algo.is_embed=True \
        algo.mixed_coeff=0 \
        model=small \
        model.length=1024 \
        algo.min_percentile=.05 \
        algo.max_percentile=.4\
        algo.pure_continuous=true;

