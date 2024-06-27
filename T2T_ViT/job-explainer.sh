#!/bin/bash
#
#SBATCH --job-name=explainer
#SBATCH --partition=common
#SBATCH --qos=2gpu8h
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output='checkpoints/%x/s%J/stdout'
#SBATCH --signal=SIGINT@90
# Signal is sent 90 seconds before time limit for clean shutdown.
# Replace SIGINT with SIGUSR1 to enable auto-requeueing
# (save a temporary checkpoint, requeue job, load checkpoint).
# See: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html#enable-auto-wall-time-resubmissions

echo "Job $SLURM_JOB_ID started at $(date +'%F %R') on $(hostname), $(nvidia-smi -L)"
export PYTHONUNBUFFERED=1

srun python -m lightning_modules.explainer fit --config lightning_configs/explainer.yaml
