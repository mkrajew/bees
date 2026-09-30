#!/bin/bash
# ==============================================================================
# Backfill checkpoint_eval.json for config 10b's already-finished run.
# See jobs/online/backfill_10a.sh for why this is needed.
# ==============================================================================
#SBATCH --job-name=backfill-10b
#SBATCH --partition=gpu-m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
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

export DEEPWINGS_DIR="$HOME/deepwings"

uv sync

uv run python wings/modeling/backfill_checkpoint_eval.py \
    wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-10b
