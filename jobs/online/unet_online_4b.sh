#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 4b (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=25, dice_weight=0.5, bce_weight=0.5)
# Augment: same as config 4 -- rotation +/-90 deg, triangle noise 80% (40-240 triangles), color jitter 100% (0.5-1.5x)
# Warm-start: config 4's own trained checkpoint (models/online/last.ckpt), not unet-final-k5.ckpt
# ==============================================================================
#SBATCH --job-name=unet-online-k5-4b
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

uv run wings/modeling/training/augmented_unet_4b.py
