#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 5b (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- same as config 5
# Augment: same as config 5 but rotation_p=0.8 (corrected from 5's 0.3 -- see script docstring)
# Mask:    circular, square_size=5/radius 2 -- identical to config 5
# Warm-start: models/new_unet/unet-final-k5.ckpt, same source as config 5 (not config 5's own checkpoint)
# ==============================================================================
#SBATCH --job-name=unet-online-k5-5b
#SBATCH --partition=gpu-m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:geforce_rtx_4090:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

cd "$SLURM_SUBMIT_DIR"

module load uv
module load cuda/12.9

uv sync

uv run wings/modeling/training/augmented_unet_5b.py
