#!/bin/bash
#
#SBATCH --job-name=continue
#SBATCH --partition=common
#SBATCH --qos=2gpu8h
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output='jobs/%x/%J.stdout'

echo "Job $SLURM_JOB_ID started at $(date +'%F %R') on $(hostname), $(nvidia-smi -L)"
export PYTHONUNBUFFERED=1

python -m vit_shapley.modules.timm_wrapper fit \
    --config checkpoints/explainer/timm_wrapper/lightning_logs/version_17655/config.yaml \
    --ckpt_path checkpoints/explainer/timm_wrapper/lightning_logs/version_17655/checkpoints/epoch=0-step=1563.ckpt
