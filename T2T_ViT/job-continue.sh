#!/bin/bash
#
#SBATCH --job-name=continue
#SBATCH --partition=common
#SBATCH --qos=2gpu8h
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output='checkpoints/%x/s%J/stdout'
#SBATCH --signal=SIGINT@90
# Signal is sent 90 seconds before time limit for clean shutdown.
# (Note that Lightning's auto-requeue doesn't work with continue,
#  it restarts from the old checkpoint instead of the new temporary one).

echo "Job $SLURM_JOB_ID started at $(date +'%F %R') on $(hostname), $(nvidia-smi -L)"
export PYTHONUNBUFFERED=1

srun python -m lightning_modules.surrogate fit \
    --config checkpoints/surrogate/s17905/config.yaml \
    --ckpt_path 'checkpoints/surrogate/s17905/checkpoints/epoch=170_val-acc=0.905.ckpt'
