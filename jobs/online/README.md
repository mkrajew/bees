# Online augmentation training configs

Four configs comparing 2 loss functions x 2 augmentation-strength presets, all
warm-started from the same checkpoint so only the loss/augmentation axes vary.

| Config | Loss | Augmentation |
|---|---|---|
| 1 | A | A |
| 2 | B | A |
| 3 | A | B |
| 4 | B | B |

Each config `N` has a training script (`wings/modeling/training/augmented_unet_N.py`)
and a matching SLURM job (`jobs/online/unet_online_N.sh`).

## Loss options

| | Loss A | Loss B |
|---|---|---|
| Class | `WeightedDiceLoss` | `BCEDiceLoss` |
| Params | `landmark_weight=100` | `pos_weight=50, dice_weight=0.5, bce_weight=0.5` |
| Note | Current baseline loss | Same loss class `unet-final-k5.ckpt` was originally trained with |

## Augmentation options

`TrainAugmentConfig(...)` (`wings/transforms.py`) fields not listed below are left
at their dataclass defaults (`horizontal_flip_p=0.5`, `triangle_min_size=2`,
`triangle_max_size=6`) for both presets -- only these fields differ:

| | Aug A | Aug B |
|---|---|---|
| `rotation_degrees` | (-90.0, 90.0) | (-90.0, 90.0) |
| `triangle_noise_p` | 0.5 | 0.8 |
| `n_triangles_range` | (80, 120) | (40, 240) |
| `color_jitter_p` | 1.0 | 1.0 |
| `color_jitter_brightness_range` | (0.5, 1.5) | (0.5, 1.5) |
| `color_jitter_contrast_range` | (0.5, 1.5) | (0.5, 1.5) |

Both presets are substantially stronger than `TrainAugmentConfig()`'s plain
defaults (`triangle_noise_p=0.4`, `n_triangles_range=(10, 60)`,
`color_jitter_p=0.4`, `color_jitter_brightness/contrast_range=(0.7, 1.3)`); B is
the stronger of the two (higher noise probability and a much wider triangle-count
range).

## Held constant across all 4 configs

- Model: `UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)`
- Warm-start checkpoint: `models/new_unet/unet-final-k5.ckpt`, loaded with
  `strict=False` (its saved criterion state doesn't necessarily match every
  config's criterion here -- e.g. `BCEDiceLoss`'s `pos_weight` buffer isn't
  present in `WeightedDiceLoss` -- so that mismatched key is ignored while the
  actual UNet weights still load exactly; verified for both loss classes)
- Mask square size: 5
- `num_epochs=100`, `batch_size=12`, `num_workers=8`
- `early_stop_patience=25`, `early_stop_min_delta=0.01` (monitor: `val_mean_error_px`)
- Data source: `data/processed/cropped/` (plain YOLO-cropped, no offline augmentation)
- Wandb project: `wingai-online-augmentation` (separate from the other training
  scripts' shared `wingai` project), logs saved under
  `wings/modeling/training/online/`
- SLURM resources: partition `gpu-m`, 1 node, 8 CPUs, 24G mem, 1x
  `geforce_rtx_4090` (pinned specifically -- see note below), 12h walltime.
  `gpu-m` ("medium jobs, classic neural networks") is the appropriate tier for
  this ~20M-param UNet -- it already trains fine on a 12GB laptop GPU, so
  `gpu-l` (reserved for large/VRAM-hungry models) would be needlessly hogging a
  bigger card. Each config runs as its own single-GPU job rather than
  distributing one config across multiple GPUs: the 4 configs are independent
  experiments, so running them as 4 separate single-GPU jobs is embarrassingly
  parallel with no cross-GPU communication overhead -- cheaper and simpler
  than DDP-ing each one across multiple cards. If a job hits the
  12h limit before finishing, `save_last=True` checkpointing (see `train.py`)
  means it can be resumed from its last checkpoint rather than restarting.

**Why `geforce_rtx_4090` is pinned instead of a generic `--gres=gpu:1`:** an
earlier run landed two configs on `h32` (`rtx_pro_6000_blackwell_max_1g`, MIG
slices of a Blackwell card) and both crashed with `CUDA error: no kernel image
is available for execution on the device` -- the cluster's installed
`torch==2.14.0+cu126` only has compiled kernels up to compute capability 9.0,
and Blackwell is 12.0. Only `glasser`'s RTX 4090s (Ada, compute capability
8.9) are within the supported range, so jobs are pinned there until the
cluster's torch/CUDA setup is upgraded to a build with Blackwell kernels
(cu130+, per the error message's own suggestion). This means only 2 configs
(matching `glasser`'s 2 physical cards) can actually run in parallel right
now; the rest queue.

**Why the job scripts use `uv sync` instead of `uv sync --reinstall`:**
running multiple jobs concurrently against the same shared `.venv` in
`~/bees`, each doing a full uninstall+reinstall of all 199 packages, caused a
real race: one job's training script hit `ImportError: cannot import name
'logger' from 'loguru'` because another job's concurrent `--reinstall` was
mid-swap of that exact package. Plain `uv sync` is a fast no-op when the
environment already matches `pyproject.toml`/`uv.lock`, so it doesn't
destructively touch already-correct packages.

## Naming per config `N`

- Run name: `online-augmentation-k5-N`
- Model/checkpoint dir name: `unet-400-online-augmentation-k5-N`
- Checkpoint files land in
  `wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-N/`,
  named `unet-400-online-augmentation-k5-N-{epoch:02d}-{val_mean_error_px:.4f}-online-augmentation-k5-N.ckpt`

## Adding more configs later

Copy the highest-numbered `augmented_unet_N.py` and `unet_online_N.sh` to
`N+1`, change `run_num`, `model_name`, `PARAMETERS["criterion"]` and/or
`TRAIN_AUGMENT_CONFIG` as needed, and add a row to this file. Keep whatever
you're not deliberately testing identical to an existing config so the
comparison stays interpretable.
