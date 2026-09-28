#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 3 (see jobs/online/README.md)
# Loss:    WeightedDiceLoss(landmark_weight=100)
# Augment: rotation +/-90 deg, triangle noise 80% (40-240 triangles), color jitter 100% (0.5-1.5x)
# ==============================================================================
#SBATCH --job-name=unet-online-k5-3
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

uv run wings/modeling/training/augmented_unet_3.py
