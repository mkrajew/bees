#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 8d (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- unchanged from config 8
# Augment: Aug B, rotation_p=1.0, triangles 60-300, size 2-13 -- unchanged from config 8
# Mask:    circular, square_size=7/radius 3 -- down from config 8's radius 4
# Warm-start: config 8's own checkpoint (unet-400-online-augmentation-k5-8/last.ckpt)
# ==============================================================================
#SBATCH --job-name=unet-online-k5-8d
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

uv run wings/modeling/training/augmented_unet_8d.py
