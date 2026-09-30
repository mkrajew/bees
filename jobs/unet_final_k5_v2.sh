#!/bin/bash
# ==============================================================================
# Redo of unet-final-k5.ckpt's own recipe, warm-started from that same
# checkpoint, with the wings/dataset.py coordinate-rounding fix (commit
# 8ef4e61) applied. See wings/modeling/training/unet_final_k5_v2.py's
# docstring for the recipe (BCEDiceLoss(pos_weight=50, dice_weight=0.8,
# bce_weight=0.2), kernel_size=5, sigmoid=False, mask square_size=3) and
# why it's believed to match the original run.
#
# Rebuilds the static square_size=3 mask datasets first
# (rebuild_baseline_k5_datasets.py -- NOT wings/dataset.py's own __main__,
# which has since moved to square_size=5 and *_sq5.pth-suffixed files for a
# different, later need) so training actually picks up the fix -- this
# offline pipeline reads pre-built .pth files, unlike the online-
# augmentation series which builds masks fresh every access.
# ==============================================================================
#SBATCH --job-name=unet-final-k5-v2
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

uv run wings/modeling/training/rebuild_baseline_k5_datasets.py

uv run wings/modeling/training/unet_final_k5_v2.py
