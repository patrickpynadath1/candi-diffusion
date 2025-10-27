#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py --config-name=config_text8 mode=sample_eval algo=mdlm \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/openwebtext-train/text8-mdlm-1024/checkpoints/last.ckpt \
        model=small \
        model.length=1024;