#!/bin/bash
#SBATCH -J sample_mdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption


export HYDRA_FULL_ERROR=1




CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model=small \
  algo=candi \
  sampling.num_sample_batches=16 \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/puure-cont-embed/checkpoints/last.ckpt \
  model.length=1024 \
  algo.step_size=1 \
  algo.mixed_coeff=0 \
  algo.is_embed=True \
  algo.pure_continuous=True \
  +wandb.offline=true;