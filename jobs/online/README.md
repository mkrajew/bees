# Online augmentation training configs

Four configs comparing 2 loss functions x 2 augmentation-strength presets, all
warm-started from the same checkpoint so only the loss/augmentation axes vary.
Configs 4b, 4c and 4d are follow-ups on top of config 4's own result (see
below) rather than further points in this grid.

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
| Model `sigmoid` | `True` | `False` |
| Note | Current baseline loss | Same loss class `unet-final-k5.ckpt` was originally trained with |

**Important:** `WeightedDiceLoss` (Loss A, configs 1/3) does not apply `sigmoid`
internally -- it expects `y_pred` already in [0,1] -- while `BCEDiceLoss`
(Loss B, configs 2/4) does (`probs = torch.sigmoid(logits)` plus
`BCEWithLogitsLoss` for the BCE term, which itself requires raw logits). So
`UNet(..., sigmoid=...)` must be built differently per config: `sigmoid=True`
for configs 1/3, `sigmoid=False` for configs 2/4. Getting this wrong produces
a mathematically invalid (out-of-\[0,1\]) Dice ratio -- a clear tell is a
**negative** `train_loss`/`val_loss`, which is impossible for a correctly
computed Dice loss and immediately signals this mismatch. (`DiceLoss` and
`IoULoss` in `wings/modeling/loss.py` have the same no-internal-sigmoid
behavior as `WeightedDiceLoss`, in case either is used in a future config.)

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

## Follow-up: config 4b

Config 4 (Loss B, Aug B) performed best of the 4 but plateaued rather than
still improving at the early-stop point. Config 4b tries a lower `pos_weight`
on top of that result, instead of restarting the loss/augmentation grid:

| | Config 4 | Config 4b |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | `BCEDiceLoss(pos_weight=25, dice_weight=0.5, bce_weight=0.5)` |
| Augmentation | Aug B | Aug B (unchanged) |
| Warm-start | `models/new_unet/unet-final-k5.ckpt` | Config 4's own checkpoint, `wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-4/last.ckpt` |

`pos_weight` trades off precision vs. recall in `BCEWithLogitsLoss` (higher
values push harder for recall at the cost of precision on the very sparse
landmark pixels); config 4's run showed `val_precision=0.75` / `val_recall=0.91`
-- halving `pos_weight` is a first, cheap step to see if that imbalance
narrows without giving up too much recall.

**Warm-start gotcha -- only the UNet weights are loaded, not the full
checkpoint:** configs 1-4 warm-start via `train(..., path=checkpoint,
strict=False)`, which loads the *entire* saved `LitNet` state, including the
criterion's own buffers. That's fine there because `strict=False` only papers
over missing/unexpected keys (e.g. `WeightedDiceLoss` has no `.bce`
submodule at all). Config 4b reuses the exact same criterion *class*
(`BCEDiceLoss`) with a different `pos_weight`, so the key
`criterion.bce.pos_weight` exists on **both** sides -- and `strict` does not
guard against a matching key's value being overwritten by the checkpoint's
stored one. Verified directly: loading config 4's `last.ckpt` the normal way
into a freshly built `BCEDiceLoss(pos_weight=25)` leaves it holding
`pos_weight=50` straight after loading, silently discarding the whole point
of config 4b. `wings/modeling/training/augmented_unet_4b.py` avoids this by
extracting only the `model.`-prefixed keys from that checkpoint's state
dict, loading those into a plain `UNet` directly, and calling
`train(..., path=None)` so `LitNet` is built fresh around the (untouched)
config 4b criterion.

If a future config needs to warm-start from another config's own checkpoint
*and* change something inside a shared submodule (not just the top-level
loss class), check for this same gotcha first.

## Follow-up: config 4c

Config 4b's `pos_weight` change barely moved `val_mean_error_px` (~2.3px for
config 4 vs ~2.4px for 4b), even after a `ReduceLROnPlateau` cut -- but a
direct re-measurement of `unet-final-k5.ckpt` (the pre-online-augmentation
checkpoint every config here warm-starts from) using today's exact
evaluation code, on the same val split, told a bigger story than either:

| | `unet-final-k5.ckpt` (old) | Config 4 | Config 4b |
|---|---|---|---|
| `val_dice` | 0.576 | 0.824 | 0.837 |
| `val_wrong_spot_count_pct` | 0.99% | 9.46% | 5.89% |
| `val_mean_error_px` | 1.383 | 2.297 | 2.375 |
| `val_median_error_px` | 1.312 | 2.200 | 2.291 |

The old model has *much lower* Dice but *much better* landmark position
accuracy and point-count reliability -- the opposite of what you'd expect if
Dice tracked landmark quality. The explanation traces to a variable nobody
had touched: **`unet-final-k5.ckpt` was originally trained with mask
`square_size=3`** (confirmed by inspecting `notebooks/04_save_datasets.ipynb`
and the commit that produced it), while every online-augmentation config so
far (1-4, 4b) uses `square_size=5`. A larger mask target lets the model get
away with larger, more diffuse predicted blobs: still decent Dice (rewards
covering more of a bigger target region), but a less precisely centered
`cv2.moments` centroid, and more likely to bleed into a neighboring
landmark's blob (hence 4/4b's much higher `wrong_spot_count_pct`). Dice
between `square_size=3` and `square_size=5` targets isn't directly
comparable at all -- the smaller target is inherently more sensitive to a
1px boundary error, so a *sharper, more accurate* model can legitimately
score *lower* on it. `val_mean_error_px` itself doesn't depend on
`square_size` (it compares the predicted blob centroid straight to the raw
ground-truth coordinate, never touching the mask target), so the 1.38 vs.
2.3-2.4px gap is not an artifact of that mismatch -- it's the real,
comparable number, and it's what config 4c tests.

Config 4c changes **only** `square_size` (5 -> 3) relative to config 4 --
same loss, same `pos_weight=50`, same augmentation, same warm-start source
(`unet-final-k5.ckpt`, not config 4b's checkpoint, since 4b already
introduced pos_weight as a second variable and unet-final-k5.ckpt is itself
square_size=3-native) -- to isolate this one variable instead of conflating
it with 4b's `pos_weight` change:

| | Config 4 | Config 4c |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same |
| Augmentation | Aug B | same |
| Mask `square_size` | 5 | **3** |
| Warm-start | `models/new_unet/unet-final-k5.ckpt` | same |

No warm-start gotcha here (unlike 4b): `unet-final-k5.ckpt`'s own saved
criterion is `BCEDiceLoss(pos_weight=50, ...)` -- confirmed directly from its
state dict -- an exact match to config 4c's, so the standard
`train(..., path=checkpoint, strict=False)` pattern (same as configs 1-4) is
safe as-is.

## Follow-up: config 4d

A second, complementary test of the same underlying idea as 4c: instead of
changing the mask target's *size* (square_size 5 -> 3), config 4d changes its
*shape* -- circular instead of square, via the new
`generate_circular_landmark_mask` (`wings/dataset.py`), holding `square_size`
at 5 (so the circle's radius is `square_size // 2 = 2`, same as 4/4b's
square). A circle has no corners -- the pixels farthest from the true
landmark center, and the cheapest ones for a model to "cover" for Dice credit
without actually sharpening its localization -- and measurably less area at
the same nominal size (13 vs. 25 px/landmark, verified directly). If config
4/4b's diffuse blobs are partly the model exploiting those corners rather
than the raw target area, a circular target should push toward smaller,
better-centered blobs somewhat like 4c's smaller square, without touching the
"size" knob at all. Running 4c and 4d side by side tells apart whether it's
target *area*, target *shape* (corners), or both, that drove the gap in the
4c table above.

| | Config 4 | Config 4c | Config 4d |
|---|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same | same |
| Augmentation | Aug B | same | same |
| Mask shape | square | square | **circle** |
| Mask `square_size` | 5 | **3** | 5 (radius 2) |
| Warm-start | `models/new_unet/unet-final-k5.ckpt` | same | same |

`generate_circular_landmark_mask` is a drop-in for `generate_landmark_mask`
(same `(image, labels, square_size)` signature); `TransformedMaskDataset` and
`build_mask_datasets` both take a `mask_fn` parameter selecting between them
(default stays `generate_landmark_mask`, so configs 1-4/4b/4c are unaffected).
No evaluation-side changes needed: `val_mean_error_px`/`wrong_spot_count_pct`
are computed from the model's *predicted* mask's contours regardless of what
shape the *training target* used, so whichever shape the model learns to
predict is picked up automatically.

## Held constant across all 4 configs

- Model: `UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)`
- Warm-start checkpoint: `models/new_unet/unet-final-k5.ckpt`, loaded with
  `strict=False` (its saved criterion state doesn't necessarily match every
  config's criterion here -- e.g. `BCEDiceLoss`'s `pos_weight` buffer isn't
  present in `WeightedDiceLoss` -- so that mismatched key is ignored while the
  actual UNet weights still load exactly; verified for both loss classes)
- Mask shape/size: square, `square_size=5` (config 4c changes size to 3,
  config 4d changes shape to circle -- see above)
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

Same pattern for configs 4b/4c/4d (`N` = `"4b"`/`"4c"`/`"4d"`, e.g.
`unet-400-online-augmentation-k5-4d`).

## Adding more configs later

Copy the highest-numbered `augmented_unet_N.py` and `unet_online_N.sh` to
`N+1`, change `run_num`, `model_name`, `PARAMETERS["criterion"]` and/or
`TRAIN_AUGMENT_CONFIG` as needed, and add a row to this file. Keep whatever
you're not deliberately testing identical to an existing config so the
comparison stays interpretable.

For a follow-up on top of a specific config's own result (like 4b), instead
copy that config's files, point the warm-start at its checkpoint instead of
`unet-final-k5.ckpt`, and check the warm-start gotcha above before assuming
`train(..., path=checkpoint, strict=False)` is still safe -- it only is when
nothing you're changing lives inside a submodule (e.g. the criterion) whose
key names are unchanged from the checkpoint you're loading.
