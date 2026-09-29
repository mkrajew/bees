#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 10b (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3) -- unchanged from 9a
# Augment: Aug B, rotation_p=1.0, triangles 80-360, size 2-16 -- stronger debris noise than 9a
# Mask:    circular, square_size=7/radius 3 -- unchanged from 9a
# Warm-start: config 9a's own checkpoint (unet-400-online-augmentation-k5-9a/last.ckpt)
# New:     smoothed-metric checkpoint selection (see wings/modeling/litnet.py /
#          train.py) -- see augmented_unet_10b.py's docstring.
# ==============================================================================
#SBATCH --job-name=unet-online-k5-10b
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

uv run wings/modeling/training/augmented_unet_10b.py
