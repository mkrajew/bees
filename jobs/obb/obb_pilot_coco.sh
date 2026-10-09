#!/bin/bash
# ==============================================================================
# OBB wing detector, pilot run 1 of 3 (see jobs/obb/README.md): start from the base YOLO26n (COCO).
# Config: configs/obb/pilot-coco.yaml (the three pilot runs differ only in their starting weights).
# Submit from the repository root (~/bees):   sbatch jobs/obb/obb_pilot_coco.sh
# Continue an interrupted run:                 RESUME=1 sbatch jobs/obb/obb_pilot_coco.sh
# ==============================================================================
#SBATCH --job-name=obb-pilot-coco
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

# Plain `uv sync`, never `--reinstall`: concurrent jobs share one .venv (see jobs/online/README.md).
uv sync

# The data pipeline limits the speed, so the workers use all CPUs of the task (SLURM_CPUS_PER_TASK).
uv run python -m wings.detection.train_obb train configs/obb/pilot-coco.yaml ${RESUME:+--resume}
