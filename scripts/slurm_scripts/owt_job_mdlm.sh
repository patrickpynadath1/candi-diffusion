#!/bin/bash
#SBATCH -J owt-candi
#SBATCH -p gpu                 # H100 partition  :contentReference[oaicite:0]{index=0}
#SBATCH --nodes=1              # keep all GPUs on the same node
#SBATCH --gres=gpu:8           # all 8 H100
#SBATCH -A ruqiz
#SBATCH --cpus-per-task=112    # give the single task every core
#SBATCH -t 96:00:00              # wall-time (edit as needed)
#SBATCH -o logs/%x-%j.out      # capture stdout/stderr

module purge
module load conda
conda activate cont_diff

export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 main.py scratch_dir=$SCRATCH_DIR \
        data=openwebtext \
        wandb.name=mdlm-orig \
        optim.lr=.0003 \
        lr_scheduler=constant_warmup \
        wandb.project=candi-owt \
        model=small \
        algo=mdlm \
        model.length=1024 \
        trainer.max_steps=50000 \
        trainer.accumulate_grad_batches=4 \
        batch_size=512 \
        trainer.val_check_interval=5000;