#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 7b (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- unchanged from config 7
# Augment: Aug B, rotation_p=1.0, triangle_max_size=11 -- identical to config 7
# Mask:    circular, square_size=9/radius 4 -- unchanged from config 6/7
# Change:  early_stop_patience 25 -> 45 -- config 7 early-stopped on a plateaued
#          val_mean_error_px while val_wrong_spot_count_pct was still improving
# Warm-start: config 7's own checkpoint (unet-400-online-augmentation-k5-7/last.ckpt)
# ==============================================================================
#SBATCH --job-name=unet-online-k5-7b
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

uv run wings/modeling/training/augmented_unet_7b.py
