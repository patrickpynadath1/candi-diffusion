#!/bin/bash
#SBATCH -J text8-pure-cont-embed
#SBATCH -p gpu                 # H100 partition  :contentReference[oaicite:0]{index=0}
#SBATCH --nodes=1              # keep all GPUs on the same node
#SBATCH --gres=gpu:8           # all 8 H100
#SBATCH -A ruqiz
#SBATCH --cpus-per-task=112    # give the single task every core
#SBATCH -t 24:00:00              # wall-time (edit as needed)
#SBATCH -o logs/%x-%j.out      # capture stdout/stderr

module purge
module load conda
conda activate cont_diff

export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 main.py scratch_dir=$SCRATCH_DIR \
        data=text8 \
        wandb.name=pure-cont-embed-text8 \
        optim.lr=.0003 \
        lr_scheduler=constant_warmup \
        lr_scheduler.num_warmup_steps=250 \
        wandb.project=candi-owt \
        algo=candi \
        model=small \
        model.length=1024 \
        trainer.max_steps=1000 \
        trainer.accumulate_grad_batches=4 \
        batch_size=512 \
        trainer.val_check_interval=500 \
        algo.mixed_coeff=0 \
        algo.min_percentile=.1 \
        algo.max_percentile=.4 \
        algo.pure_continuous=True \
        algo.is_embed=True;