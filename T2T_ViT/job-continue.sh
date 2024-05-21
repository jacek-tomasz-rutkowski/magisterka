#!/bin/bash
#
#SBATCH --job-name=continue
#SBATCH --partition=common
#SBATCH --qos=2gpu8h
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output='checkpoints/%x/%J/stdout'

echo "Job $SLURM_JOB_ID started at $(date +'%F %R') on $(hostname), $(nvidia-smi -L)"
export PYTHONUNBUFFERED=1

python -m vit_shapley.modules.timm_wrapper fit \
    --config checkpoints/transfer/17655/config.yaml \
    --ckpt_path 'checkpoints/transfer/17655/checkpoints/epoch=0.ckpt'
