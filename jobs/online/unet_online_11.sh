#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 11 (see jobs/online/README.md)
# Not a new hyperparameter: re-runs config 10a/9a's exact recipe now that
# wings/dataset.py's coordinate-rounding bug is fixed (see
# augmented_unet_11.py's docstring, and commit 8ef4e61).
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3) -- unchanged from 9a/10a
# Augment: Aug B, rotation_p=1.0, triangles 60-300, size 2-13 -- unchanged from 9a/10a
# Mask:    circular, square_size=7/radius 3 -- unchanged from 9a/10a
# Warm-start: config 9a's own checkpoint (unet-400-online-augmentation-k5-9a/last.ckpt),
#             not 10a's -- 10a's own run only reached epoch 8/150 before stopping.
# ==============================================================================
#SBATCH --job-name=unet-online-k5-11
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

# wings/deepwings_eval.py (called automatically at the end of training, see
# wings/modeling/train.py) needs DeepWings' test/ + test_masks/ folders --
# copy them here first (see jobs/online/README.md's transfer commands).
export DEEPWINGS_DIR="$HOME/deepwings"

uv sync

uv run wings/modeling/training/augmented_unet_11.py
