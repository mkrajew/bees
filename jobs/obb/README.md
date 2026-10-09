# OBB wing detector: training runs (stage 2)

Training of the oriented-box detector that replaces the axis-aligned `models/yolo26n/best.pt` (design and data:
`docs/superpowers/specs/2026-10-07-obb-dataset-augmentation-design.md`). One run = one YAML file in `configs/obb/` (all settings, see
`ObbRunConfig` in `wings/detection/train_obb.py`) + one SLURM job in `jobs/obb/`. Nothing but the YAML defines a run.

- **Model:** `yolo26n-obb` (2.65 M parameters; `n` because the detector runs in the browser), image size 640, one class.
- **Training data:** `WingOBBDataset` makes every sample online from the raw images (rotation, scale, flip, compositions of 2-4 wings, photometric
  steps); one epoch = every training image once as the first wing. Ultralytics' own augmentations are switched off.
- **Validation:** the frozen val set (`data/processed/detection-obb/val`, 2,697 images) after every epoch; the best epoch is chosen by the Ultralytics
  fitness (0.1 mAP50 + 0.9 mAP50-95, rotated IoU). The frozen **test** set is for the final model only (`eval --split test`).
- **Last epochs:** `close_mosaic` epochs at the end are trained without compositions (`p_multi = 0`, single wings only, the production case).
- **Where things land:** runs in `wings/detection/runs/obb/<name>/` (`weights/best.pt`, `last.pt`, `results.csv`, `obb_config.yaml`, plots),
  weights to start from in `models/` (`models/obb/` for OBB checkpoints), W&B project `wings-detection-obb` (entity `furkot-team`).

## One-time setup on the cluster

Run on the **laptop**, in the repository root (`<login>` = how you `ssh` to the cluster; the repository lives in `~/bees` there):

```bash
rm -f data/processed/detection-obb/val/labels.cache data/processed/detection-obb/test/labels.cache   # Ultralytics' caches of this laptop; the cluster builds its own
ssh <login> "mkdir -p bees/data/processed/detection-obb bees/models/obb bees/logs"
scp -r data/processed/detection-obb/labels.csv data/processed/detection-obb/dataset.yaml data/processed/detection-obb/val data/processed/detection-obb/test <login>:bees/data/processed/detection-obb/
scp models/obb/yolo26n-obb.pt <login>:bees/models/obb/
```

- The frozen val and test sets are copied (not regenerated) so that they are byte for byte the ones inspected locally. `data/raw` is already on the
  cluster; the old `data/processed/detection/` folders are not needed. The image lists (`train.txt` ...) hold absolute paths, so every job rebuilds them
  for the cluster when it starts (and stops with the first missing file name if `data/raw` is incomplete); nothing to copy.
- The other two starting weights, `models/yolo26n.pt` and `models/yolo26n/best.pt`, come from the old detector; copy them with `scp` if they are not there.
- `git pull` in `~/bees` (this branch, `yolo-obb-training`, or `main` once it is merged), then `uv sync` once from the login node.
- `wandb login` once. If the compute nodes have no internet, add `export WANDB_MODE=offline` to the job and upload later with `wandb sync`.

## The pilot: which starting weights?

Three runs, **identical except for the starting weights** (40 epochs, batch 32, MuSGD lr0 0.01, last 5 epochs without compositions, seed 42, W&B group `pilot`).
40 epochs because the pilot only has to rank the starts cheaply; `patience` equals the number of epochs, so there is no early stopping and the three runs
have exactly the same schedule. (100 epochs for the pilot were considered and dropped: they cost 2.5 times as much, and the longer comparison is the next round.)

| Run | Starts from | Tensors transferred (of 792) | Job |
|---|---|---|---|
| `pilot-coco` | `models/yolo26n.pt` (base YOLO26n, COCO) | 606 (the head starts random) | `jobs/obb/obb_pilot_coco.sh` |
| `pilot-wings` | `models/yolo26n/best.pt` (our wing detector) | 708 (only the angle branch starts random) | `jobs/obb/obb_pilot_wings.sh` |
| `pilot-dota` | `models/obb/yolo26n-obb.pt` (Ultralytics OBB, DOTAv1, 1024 px, rotations +-180 deg, release v8.4.0 of `ultralytics/assets`) | 780 (only the class outputs start random) | `jobs/obb/obb_pilot_dota.sh` |

```bash
sbatch jobs/obb/obb_pilot_coco.sh
sbatch jobs/obb/obb_pilot_wings.sh
sbatch jobs/obb/obb_pilot_dota.sh
```

Submit from `~/bees`. Only two jobs can run at once (node `glasser` has 2 RTX 4090 and 16 CPUs in total, every job takes 8), the third waits.
The winner is the run with the best validation fitness in W&B (group `pilot`, metric `metrics/mAP50-95(B)`); the first epochs also tell the time per
epoch, which decides the epoch budget of the later rounds. If a job hits its time limit: `RESUME=1 sbatch jobs/obb/obb_pilot_dota.sh` continues it from
`last.pt` (same W&B run).

Cost: 3 x 40 epochs, so with E minutes per epoch the pilot takes 2 x E GPU-hours (E = 4: 8 h, E = 8: 16 h of the roughly 45-50 h budget). Submit
`pilot-dota` first, read E from its first epochs (40 x E / 60 hours against the 12 h limit of the job), and only then submit the other two.

## Planned next rounds (one variable at a time, like `jobs/online/`)

1. With the winning start: `lr0` 0.01 vs 0.003, then `p_multi` 0.2 / 0.4 / 0.6, then the brightness range of the photometric step (`augment.brightness_range`,
   the default 0.5-1.5 saturates bright backgrounds to white), 100 epochs each (including a 100-epoch baseline with the default settings, because the pilot
   ran only 40).
2. Final run: the best configuration, 150-200 epochs (optionally 2-3 seeds), then `eval --split test` once.

Each new config is a copy of the previous YAML and job script with the changed key(s) and a new `name`; keep everything else identical so the comparison
stays interpretable, and add a row here.

## Smoke test on the laptop (before the cluster)

```bash
uv run python -m wings.detection.train_obb train configs/obb/pilot-dota.yaml --set name=smoke --set epochs=2 --set fraction=0.02 --set wandb=false --set close_mosaic=1 --set workers=2
```

`--set key=value` overrides any key of the YAML (`augment.p_multi=0.2` reaches into the augmentation settings); `fraction` uses only the first part of
the training images; `workers=2` because on Windows every worker is spawned (the default would be one per CPU). The same run with W&B in offline mode:
`WANDB_MODE=offline` in front and `wandb=true` instead of `wandb=false`. Evaluate a checkpoint:
`uv run python -m wings.detection.train_obb eval wings/detection/runs/obb/smoke/weights/best.pt --workers 2` (add `--split test` only for the final model).
A smoke run leaves `wings/detection/runs/obb/smoke`; delete it before running the command again (a finished run is never overwritten).

## Things to know

- `optimizer: auto` **ignores `lr0` and `momentum`** (it picks them itself), so every config names the optimizer (`MuSGD`, which is what `auto` picks here).
- The speed is limited by the data pipeline (tens of milliseconds of CPU per sample; in a profile roughly half of it is `apply_photometric`, the rest
  PNG decoding, warping and compositions), not by the GPU. The real time per epoch on the cluster is known only from the first epochs of the pilot
  (`time` column of `results.csv`, the progress bar in `logs/`). If it is too slow, the photometric step is the first thing to speed up (and a
  speed-up that changes its output needs `freeze` re-run). Asking for 16 CPUs would fill the whole `glasser` node and block the second GPU, so two jobs
  with 8 CPUs each give the same total throughput. `workers` follows `SLURM_CPUS_PER_TASK`.
- Jobs are pinned to `geforce_rtx_4090`: a generic `--gres=gpu:1` can land on the Blackwell node `h32`, where the installed torch has no kernels. The time
  limit of the partition is 24 h; the pilot jobs ask for 12 h, like the older ones: 40 epochs need 40 x E / 60 hours at E minutes per epoch, and E is known
  only after the first epochs. A job that is killed at the limit is continued with `RESUME=1`.
- On Windows every DataLoader worker is spawned (about a minute for eight, and again when the compositions are switched off); on the cluster (Linux,
  fork) they start at once. Timings from a laptop, especially one on battery, say little about the cluster.
- `uv sync`, never `--reinstall`, in jobs: concurrent jobs share one `.venv`.
- At the start of a run Ultralytics' AMP check downloads `yolo26n.pt` (5 MB) into the working directory (`~/bees/yolo26n.pt`, git-ignored); without
  internet it only warns and trains on.
- `wandb: true/false` in the config decides about W&B logging, whatever the per-user Ultralytics setting `wandb` says on the machine (the stock
  callback would otherwise start a run of its own and crash on the project name). Set only for the running process, the settings file is not touched.
- At the very end of a run W&B prints about ten warnings `Tried to log to step N that is less than the current step N+1`: Ultralytics logs its final
  plots (PR and F1 curves, confusion matrices, validation batches) to the step of the last epoch, which is already closed. Harmless, the per-epoch
  metrics and the best checkpoint are logged; the plots are in the run folder.
- A finished run is never overwritten: use another `name` or delete the folder. A run that was started (even if it crashed early) can be continued
  with `--resume` once `weights/last.pt` exists.
- The frozen val and test sets are about 81% single wings and 19% compositions (val: 2,197 / 257 / 120 / 123 images with 1 / 2 / 3 / 4 wings), whereas
  the training samples are 60% single wings and 40% compositions (and 100% single wings in the last `close_mosaic` epochs). The validation fitness
  therefore weighs compositions less than training does; it is still the number to compare runs by.
