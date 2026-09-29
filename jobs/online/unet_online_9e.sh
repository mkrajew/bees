#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 9e (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=100, dice_weight=0.7, bce_weight=0.3) -- combines 9d's pos_weight + 9a's dice
# Augment: Aug B, rotation_p=1.0, triangles 60-300, size 2-13 -- unchanged from config 8d
# Mask:    circular, square_size=7/radius 3 -- matches config 8d
# Warm-start: config 8d's own checkpoint, UNet WEIGHTS ONLY (pos_weight gotcha -- see script docstring)
# ==============================================================================
#SBATCH --job-name=unet-online-k5-9e
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

uv run wings/modeling/training/augmented_unet_9e.py
