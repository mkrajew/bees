#!/bin/bash
# ==============================================================================
# Online augmentation training -- config 4d (see jobs/online/README.md)
# Loss:    BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- same as config 4
# Augment: same as config 4 -- rotation +/-90 deg, triangle noise 80% (40-240 triangles), color jitter 100% (0.5-1.5x)
# Mask:    circular landmark mask (vs. square everywhere else), same square_size=5 diameter knob
# Warm-start: models/new_unet/unet-final-k5.ckpt, same source as configs 1-4/4c
# ==============================================================================
#SBATCH --job-name=unet-online-k5-4d
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

# UV_CACHE_DIR: uv's default cache lives on node-local /local/ssd/cache/uv,
# but .venv (on the shared NFS home dir) stores symlinks INTO that cache --
# so a .venv populated on one node (e.g. glasser) has dangling symlinks on
# any other node (h32/h86/gpu-s), since each node has its own separate
# local disk. Pointing the cache at the shared home dir instead makes it
# resolve identically everywhere. Fixes: "ImportError: cannot import name
# 'logger' from 'loguru'" (and similar) when a job lands on a node whose
# local cache never got populated -- see jobs/online/README.md.
export UV_CACHE_DIR="$HOME/.cache/uv-shared"
module load cuda/12.9

uv sync

uv run wings/modeling/training/augmented_unet_4d.py
