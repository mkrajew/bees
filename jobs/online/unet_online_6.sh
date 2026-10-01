#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 6 (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- unchanged, matches the DeepWings paper's own weighting too
# Augment: Aug B, rotation_p=1.0 (full rotation training again -- see script docstring for why this may now be safe)
# Mask:    circular, square_size=9/radius 4 -- matches the DeepWings paper's own validated optimal landmark radius
# Warm-start: models/new_unet/unet-final-k5.ckpt (fresh start, new mask target size)
# ==============================================================================
#SBATCH --job-name=unet-online-k5-6
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

uv run wings/modeling/training/augmented_unet_6.py
