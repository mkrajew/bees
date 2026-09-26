# Online augmentation training configs

Four configs comparing 2 loss functions x 2 augmentation-strength presets, all
warm-started from the same checkpoint so only the loss/augmentation axes vary.
Configs 4b, 4c and 4d are follow-ups on top of config 4's own result (see
below) rather than further points in this grid. Config 5 is a fresh config
combining the best-evidenced fixes found across that whole 4-series
investigation (see its own section below) -- not a follow-up on any single
one of them. Config 5b corrects a `rotation_p` miscommunication in config 5.
Configs 6a-6d are loss-tuning follow-ups on top of config 5b's own result.

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

**`rotation_p`** (default `1.0`, added for config 5 -- see below): unlike
`horizontal_flip_p`/`triangle_noise_p`/`color_jitter_p`, rotation previously
had no apply-probability at all -- every training access got rotated by a
random angle drawn from `rotation_degrees`, unconditionally. At `rotation_p=1.0`
(every config through 4d) that's fine as a concept but has a real consequence:
`rotation_degrees` is a wide *continuous* range, so the chance of a random draw
landing anywhere near 0 degrees (what validation always uses -- val/test never
rotate) is essentially zero. Training supplies virtually no examples resembling
what val is scored on. `rotation_p < 1` gives that fraction of training accesses
the identity (no rotation) instead, via `v2.RandomApply` wrapping the existing
`v2.RandomRotation`.

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

## Config 5 -- not a config-4 follow-up, a fresh synthesis

Pulling the wandb history for configs 4/4b/4c/4d (full per-epoch curves, not
just best-vs-last) showed something bigger than any single config's own
result: **all four `val_mean_error_px` curves *monotonically worsen* after an
early peak** (epoch 0-2), with no sign of turning back around before
early-stopping fires -- not "hasn't recovered yet," but a steady, ongoing
drift in the wrong direction (config 4c: 2.133 -> 2.235 -> ... -> 2.415 over
25 epochs; 4d and 4/4b show the same shape, shallower). More training epochs
at any of these settings would likely make things worse, not better.

The mechanism: `unet-final-k5.ckpt` was already a fully converged, precise
model on *plain, unrotated* wing crops. Fine-tuning it with `rotation_p=1.0`
(every config through 4d) means virtually 100% of training gradient comes
from randomly rotated -- and therefore resampling-blurred, for any angle away
from the axes -- images, while val/test never rotate at all. Training
supplies essentially no examples resembling what val is scored on, so
continued training pulls the model further toward "good under rotation/blur"
at the direct expense of "precise on the crisp images validation measures" --
independent of `pos_weight` or mask shape/size, which is exactly why 4b/4c/4d
didn't fix it either.

Config 5 combines the two best-evidenced fixes instead of testing one more
variable in isolation on top of config 4:

| | Config 4 | Config 5 |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same |
| Augmentation | Aug B (`rotation_p=1.0`, implicit) | Aug B, **`rotation_p=0.3`** |
| Mask | square, `square_size=5` | **circular**, `square_size=5` (radius 2) |
| Warm-start | `models/new_unet/unet-final-k5.ckpt` | same |

Why the circular mask (config 4d's approach) and not 4c's `square_size=3`:
`square_size=3` was tried first -- it initially looked more promising than
4d (better best-epoch median error on clean validation, and a *fully
converged* track record from `unet-final-k5.ckpt` itself at 1.38px, which the
circular mask didn't have). But re-running config 4c's actual checkpoint
through notebook 25's rotation-robustness sweep showed it performing
noticeably worse than `square_size=5` specifically on rotated input: a 3x3
target is a much less forgiving detection problem once the input is degraded
by rotation's resampling blur, with far less margin for error than a 5x5 one.
The circular mask keeps the *same radius* as config 4/4b/4d's square
(`square_size=5` -> radius 2) -- it removes the corners without shrinking the
target's overall reach -- so it shouldn't cost the rotation robustness this
whole online-augmentation effort exists to build in the first place, unlike
an outright smaller target.

Why `pos_weight=50` and not 4b's 25: 4b's full curve was *worse* than
config 4's at essentially every epoch measured (e.g. best epoch: 2.340 vs
2.281) -- no evidence so far that lowering pos_weight helps at all.

Why `rotation_p=0.3` specifically: see the "Augmentation options" section
above for the full reasoning; 0.3 keeps most of the rotation-robustness
training signal (notebook 25 confirmed that robustness is real and worth
keeping) while giving 70% of training accesses a chance to look like what val
actually measures. Worth revisiting after seeing results -- lower if the
val-error drift is still present, higher if rotation-robustness visibly
suffers (re-check with notebook 25).

**Result:** finished at 26 epochs, best epoch 0 (`val_mean_error_px=1.83`,
`val_wrong_spot_count_pct=2.49%` -- by far the best single number, and by far
the lowest wrong_pct, seen anywhere in this online-augmentation series).
Climbs to ~2.1-2.2px over the next few epochs, same direction as 4/4b/4c/4d,
but unlike their clean monotonic climb to the end, this one plateaus/
oscillates in that band instead of continuing to worsen; wrong_pct stays in a
stable 3-4% band throughout (vs. 4/4b/4c/4d's eventual 5-9%+), ending at 2.92%
-- the best *final* wrong_pct of any config so far -- and `val_median_error_px`
keeps slowly falling all the way to the last epoch (2.003). Reads as both
structural fixes genuinely working on `val_mean_error_px`.

**But:** re-running config 5's checkpoint through notebook 25's rotation-
robustness sweep showed it performing *worse* than config 4b specifically on
rotated input -- config 4b comes out clearly best there. This traces to a
miscommunication, not a flaw in the rotation_p idea itself: `rotation_p=0.3`
was meant to keep *most* rotation training while giving the model *some*
unrotated exposure, but 0.3 actually means only 30% of training accesses get
rotated (70% don't) -- the opposite emphasis from what was intended
(~70% rotated). `val_mean_error_px` structurally can't see this cost (val/test
are never rotated), which is exactly why it only showed up once someone
manually re-checked notebook 25 -- see config 5b below for the fix, and take
`val_mean_error_px` alone as an incomplete picture of "better" for any config
that touches `rotation_p` going forward.

## Config 5b -- corrects config 5's `rotation_p`

Same as config 5 (circular mask, `square_size=5`, `pos_weight=50`, Aug B) but
`rotation_p=0.8` instead of 0.3 -- correcting the inversion (config 5 rotated
only 30% of training accesses instead of the intended ~70%+), and leaning
further toward preserving rotation robustness than the initially-corrected
0.7 (80% of training accesses rotated, 20% not).
Warm-started from `unet-final-k5.ckpt` again (not config 5's own checkpoint,
since those weights are downstream of the rotation_p this file corrects).
Configs 6a-6d (below) are retargeted to warm-start from config 5b's
checkpoint instead of config 5's, once this run finishes -- there is no
reason to tune loss hyperparameters on top of a base whose augmentation
recipe is already known to need fixing.

## Config 6a-6d -- loss tuning on top of config 5b

(Retargeted from config 5 to config 5b once the `rotation_p` miscommunication
above was caught -- these were originally built and briefly documented
against config 5's checkpoint, before 5b existed.)

Config 5(b) fixes the structural issues (mask corners, rotation/val
mismatch) but on config 5 still plateaued around `val_mean_error_px`
~2.1-2.2px, held back by a persistent ~3-4% `val_wrong_spot_count_pct` tail
even as `val_median_error_px` kept slowly improving -- reads as a genuine
subset of hard-to-detect landmarks, not general stagnation. These four
configs tune the loss function itself on top of config 5b's result
(assuming the same pattern shows up there too), varying two different axes:

| | Config 5b | 6a | 6b | 6c | 6d |
|---|---|---|---|---|---|
| `pos_weight` | 50 | **75** | **100** | 50 | **75** |
| `dice_weight` / `bce_weight` | 0.5 / 0.5 | same | same | **0.8 / 0.2** | **0.8 / 0.2** |
| Augmentation | Aug B, `rotation_p=0.8` | same | same | same | same |
| Mask | circular, `square_size=5` | same | same | same | same |
| Warm-start | `unet-final-k5.ckpt` | config 5b's own checkpoint (weights only) | same | config 5b's own checkpoint (`strict=False`) | config 5b's own checkpoint (weights only) |

**`pos_weight` axis (6a, 6b):** targets the wrong_pct tail directly --
pushing harder for recall should reduce completely-missed landmarks (which
is what forces `handle_coordinates`' costly missing-point extrapolation),
at some cost to precision that the extra-point handling already copes with
reasonably well. 6a (75) and 6b (100) map two points above config 5's 50, to
see the shape of the response rather than a single guess -- config 4b
already showed 25 is worse than 50, so this round only explores upward.

**`dice_weight`/`bce_weight` axis (6c, 6d):** a ratio never varied anywhere
in this series before -- every config 1-6b kept it at 0.5/0.5. 0.8/0.2 is a
deliberate echo of `unet-final-k5.ckpt`'s own original training recipe
(`unet_kernel_5x5.py`, per git history) -- the model whose landmark
precision (1.38px, fully converged) nothing in this series has matched yet.
Dice rewards overall region overlap more forgivingly than BCE's harder
per-pixel penalty, which may reduce the missed-landmark tail from a
different angle than `pos_weight`.

**6d combines both** (6a's milder pos_weight=75 + 6c's ratio) rather than
waiting for 6a/6c's individual results first: all four run in parallel on
the cluster regardless, so testing the combination now saves a full
round-trip later if both individually help.

**Warm-start gotcha, again:** 6a/6b/6d change `pos_weight` relative to
config 5b's own saved criterion (50), so they need the same weights-only
loading as config 4b (see its section above) -- `strict=False` alone would
silently reload `pos_weight=50` from the checkpoint and discard whatever
these files set. 6c only changes `dice_weight`/`bce_weight`, which are plain
Python floats on `BCEDiceLoss` (never registered as buffers), so they never
appear in the checkpoint's state dict at all -- nothing to overwrite, and
its `pos_weight=50` matches config 5b's exactly anyway, so the standard
`train(..., checkpoint_path, strict=False)` pattern is safe for 6c alone.

## Held constant across all 4 configs

- Model: `UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)`
- Warm-start checkpoint: `models/new_unet/unet-final-k5.ckpt`, loaded with
  `strict=False` (its saved criterion state doesn't necessarily match every
  config's criterion here -- e.g. `BCEDiceLoss`'s `pos_weight` buffer isn't
  present in `WeightedDiceLoss` -- so that mismatched key is ignored while the
  actual UNet weights still load exactly; verified for both loss classes)
- Mask shape/size: square, `square_size=5` (config 4c changes size to 3;
  configs 4d, 5, 5b and 6a-6d change shape to circle, keeping
  `square_size=5`/radius 2 -- see above)
- `rotation_p=1.0` (configs 5 at 0.3 and 5b/6a-6d at 0.8 are the exceptions
  -- see above)
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

Same pattern for configs 4b/4c/4d/5b/6a/6b/6c/6d (`N` = `"4b"`/`"4c"`/`"4d"`/
`"5b"`/`"6a"`/`"6b"`/`"6c"`/`"6d"`, e.g. `unet-400-online-augmentation-k5-6a`).

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
