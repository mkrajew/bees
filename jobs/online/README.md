# Online augmentation training configs

Four configs comparing 2 loss functions x 2 augmentation-strength presets, all
warm-started from the same checkpoint so only the loss/augmentation axes vary.
Configs 4b, 4c and 4d are follow-ups on top of config 4's own result (see
below) rather than further points in this grid. Config 5 is a fresh config
combining the best-evidenced fixes found across that whole 4-series
investigation (see its own section below) -- not a follow-up on any single
one of them. Config 5b corrects a `rotation_p` miscommunication in config 5.
Configs 6a-6d are loss-tuning follow-ups on top of config 5b's own result.
Config 6 is a further fresh config (not a 6a-6d follow-up despite the
shared "6" -- see its own section) informed by reading the actual DeepWings
paper.

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

## GPA fix: `pca_prealign` (`wings/gpa.py`)

Re-checking config 5b's checkpoint against notebook 25's rotation sweep (see
config 5b's section above) surfaced a GPA-vs-nearest-neighbor gap again at
large rotation angles, similar in *symptom* to the original flip/rotation
multistart problem but only partly explained by it: at 90 degrees, several
samples showed GPA-ordered error far above nearest-neighbor error even with
`FULL_ROTATION_MULTISTART_ANGLES` already active (e.g. one sample: GPA=15.4px
vs. NN=7.9px).

Reading the actual DeepWings paper (see config 6 below) surfaced the fix:
their landmark-sorting step never solves rotation and identity-assignment
together the way `recover_order`'s Hungarian-matching does. Instead they (i)
run PCA on the *unordered* detected point cloud to find its dominant axis,
(ii) rotate the whole mask so that axis is horizontal, and only then (iii)
sort left-to-right. PCA on a point cloud doesn't care about landmark
identity or order at all -- it's a cheap, robust, per-sample-adaptive
estimate of orientation, immune to the same "bad initial correspondence
guess" failure mode that a fixed angle grid (or the raw unrotated guess) can
fall into.

`recover_order`/`handle_coordinates` gained a `pca_prealign` parameter that
does the equivalent for our own pipeline: compute the PCA angle of
`mean_coords` once and of the shape being ordered, take the (up to two,
since a principal axis is ambiguous by 180 degrees) candidate rotations that
would align them, and add those to whatever `multistart_angles` already
provides -- cheap enough (one 2x2 eigendecomposition per candidate) to apply
directly inside the `itertools.combinations` search too, not just the final
call, unlike `multistart_angles`'s fixed grid. Verified on config 5b's
checkpoint at 90 degrees: every sample that previously showed a large
GPA-vs-NN gap (e.g. the 15.4px-vs-7.9px case above) came back within ~1-3px
of its NN baseline, and the aggregate GPA mean dropped from 9.28px to
8.61px -- with `FULL_ROTATION_MULTISTART_ANGLES` combined with
`pca_prealign` giving identical results to `pca_prealign` alone in this
test, at essentially the same wall time as before. Wired into
`wings/app/images.py`'s production inference call alongside the existing
`multistart_angles=FULL_ROTATION_MULTISTART_ANGLES`. Not needed in
`litnet.py`'s training-time validation for the same reason the original
multistart fix wasn't: real val/test images are never rotated.

## Config 6 -- larger mask radius (matching the DeepWings paper) + full rotation

Prompted by actually reading the DeepWings paper (Rodrigues et al. 2022,
*Big Data Cogn. Comput.* 6(3):70 -- the source of the 0.943
positional-precision benchmark and the 19-landmark scheme this whole project
is built on). Two findings from it directly challenge choices made earlier
in this series:

**Their optimal landmark radius is *larger* than anything tried here, not
smaller.** Every mask-size experiment so far (4c's `square_size=3`, every
circular-mask config's radius 2) moved toward a *smaller* target, on the
theory that smaller forces sharper, more precisely centered blobs. The
paper's own published ablation (their Table 2) found the opposite within
their tested range: radius <3px was "virtually ignored by the network", and
accuracy kept improving from radius 3 to 4 (88.2% -> 91.8% exact-19-landmark
detection), with radius >4 the point where blobs start merging into each
other. Their radius-4 circle has ~50px of area -- larger than even our
original `square_size=5` square (25px), and ~4x our own circular radius-2
(13px). This reframes config 5b's rotation-robustness regression (see its
section above): the drop was visible directly in nearest-neighbor-matched
distances, which don't depend on landmark ordering at all, so it was a real
detection-quality cost, not (only) an ordering problem -- consistent with a
too-small target being less robust to rotation's resampling blur, the same
mechanism that made config 4c's `square_size=3` fail rotation robustness
earlier. Config 6 uses `square_size=9` (radius 4, ~49px measured) to match
the paper's own validated value.

**Their `pos_weight` (effectively 50, "background weight 1, landmark class
weight +50") and kernel size (5x5) already match ours exactly** -- so
neither of those was ever the differentiator.

**Their landmark-sorting method is unrelated to GPA/Procrustes entirely**
(PCA-align, then sort left-to-right -- see the `pca_prealign` section
above), which is a fix to our *ordering algorithm*, orthogonal to this
config's mask-size change.

Given both a properly-sized mask target and a more robust ordering
algorithm are now in place, config 6 also reverts `rotation_p` to `1.0`
(full rotation training, like every config before 5/5b/6a-6d): the
diagnosis that motivated lowering it (config 5's section above) was made
entirely on `square_size=5`/circular-radius-2 checkpoints, i.e. potentially
confounded by the same too-small-target issue. It's an open question
whether a properly-sized target removes the need to ration rotation
exposure at all -- config 6 tests exactly that.

| | Config 5b | Config 6 |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same |
| Augmentation | Aug B, `rotation_p=0.8` | Aug B, **`rotation_p=1.0`** |
| Mask | circular, `square_size=5` (radius 2) | circular, **`square_size=9`** (radius 4) |
| Warm-start | `models/new_unet/unet-final-k5.ckpt` | same |

Warm-started fresh from `unet-final-k5.ckpt` (not any config 4/5-family
checkpoint): this changes the mask target size, which nothing else in this
series has ever trained against, so there's no more-relevant checkpoint to
build on than the original converged baseline. No warm-start gotcha:
`pos_weight=50` matches `unet-final-k5.ckpt`'s own criterion exactly, so the
standard `strict=False` pattern is safe.

## Follow-up: config 6

Notebook 25 (in-domain test set) on config 6's checkpoint: positional
precision ~0.992-0.994, essentially flat across the entire 0-90 degree
rotation sweep (see "Metrics standardization" below for how this is now
reported) -- the best rotation-robustness result of this whole series, and
achieved even *without* `pca_prealign` (multistart alone was enough this
time). Contrast with config 5b, which still showed GPA=8.61px at 90 degrees
even *with* `pca_prealign` -- confirming that config 5b's problem really was
(partly) a detection-quality cost from too-small a mask, not only an
ordering-algorithm gap, exactly as config 6's own docstring predicted.

Notebook 26 (external DeepWings test set, 3510 image/mask pairs -- true
out-of-domain generalization) on the same checkpoint: mean positional
precision 0.9127 over all evaluated wings (paper: 0.943), 0.9525 restricted
to the 77.9% of wings with a plausible (16-22) predicted point count --
actually *exceeding* the paper's own number on that subset. The entire gap
to 0.943 traces to outright detection failures on the noisier ~22% of
DeepWings photos, not to positional/shape accuracy where detection succeeds
at all -- motivating config 7 below.

## Metrics standardization: positional precision as the single headline number

Notebooks 25/26 used to report four separate pixel-distance numbers (GPA
mean/median, nearest-neighbor mean/median), with no absolute reference point
-- reasonable for tracking one config's own rotation robustness in isolation,
but not for "are we doing well," since pixel error isn't comparable across
images of different scale/resolution, let alone against the DeepWings
paper's own headline metric. Standardized on **mean `wing_positional_precision`**
(DeepWings' own Procrustes-based shape-only metric, bounded [0, 1], their
paper reports 0.943) as *the* number to lead with and compare across
configs/against the paper, computed over **all** evaluated samples with no
filtering -- a "restricted to reliable point counts" version would reward a
model that fails outright more often, since each failure would just drop out
of the average instead of counting against it (see notebook 26's own
restricted-vs-unfiltered gap above for why this matters in practice). Pixel
distances (GPA-ordered and nearest-neighbor mean/median) are kept in both
notebooks as secondary/diagnostic detail, not the headline.

Extracted the metric implementation (previously duplicated ad hoc between
notebooks 20 and 26) into `wings/metrics.py`, shared by notebooks 25 and 26:
`gpa_ordered_metrics_single` (ground truth already ordered -- notebook 25's
in-domain case, only the prediction needs GPA ordering) and
`gpa_ordered_metrics_both` (neither side ordered -- notebook 26's
external-dataset case) both return `(pixel_distances, precision)` from a
single GPA pass, alongside the shared `wing_positional_precision` and
`nn_matched_distances`.

## Config 7 -- larger triangle noise, targeting DeepWings' dirtier photos

A follow-up on config 6's checkpoint (see above), not a new hypothesis:
config 6's remaining gap to the DeepWings paper's 0.943 traces almost
entirely to outright detection failures on the ~22% of DeepWings photos
outside the reliable 16-22 predicted-point range, not to positional
accuracy -- and DeepWings' own photos are frequently noted as considerably
dirtier/more damaged than this project's own collection. `TrainAugmentConfig`
has always varied triangle noise *count* (`n_triangles_range`) but never
*size* (`triangle_min_size`/`triangle_max_size` left at their defaults, 2/6,
in every config so far -- specks only ~2-9px on a 400x400 crop, comparable to
or smaller than a single landmark).

**Visual check before training on it (`notebooks/24_online_augmentation.ipynb`,
"Config 7" section):** an initial proposal of `triangle_max_size=20` (roughly
3x, leaving `n_triangles_range` unchanged) rendered at fixed count/size
combinations to compare medium against large draws directly. Two things fell
out of that check: (1) at `max_size=20` combined with the existing
240-triangle ceiling, wing venation was substantially obscured across a large
fraction of the crop -- too aggressive to train on as-is; (2) since
`TriangleNoise` draws each triangle's size *uniformly* between `min_size` and
`max_size`, raising the max shifts every triangle's expected size up, not
just a rare worst case (half of all draws land above the new range's
midpoint on *every* image) -- so the "medium" case was already more cluttered
than any config before it. A maxed-count/medium-size draw and a
medium-count/maxed-size draw looked comparably cluttered, i.e. count and size
contribute similar amounts of total occlusion, with neither dominating.
Settled on `triangle_max_size=11` (moderated down from 20) with
`n_triangles_range` left unchanged at `(40, 240)`: legible at the 240-triangle
ceiling, still meaningfully bigger than config 6's specks, and with size
itself now moderated there's no longer a clear case for also cutting count.

| | Config 6 | Config 7 |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same |
| Augmentation | Aug B, `rotation_p=1.0`, `triangle_max_size=6` (default) | Aug B, `rotation_p=1.0`, **`triangle_max_size=11`** |
| Mask | circular, `square_size=9` (radius 4) | same |
| Warm-start | `models/new_unet/unet-final-k5.ckpt` | **config 6's own checkpoint** |

Warm-started from config 6's own trained checkpoint
(`unet-400-online-augmentation-k5-6/last.ckpt`), not `unet-final-k5.ckpt`:
nothing about the mask size or architecture changes this time, so config 6's
checkpoint is the more relevant starting point. No warm-start gotcha:
`pos_weight=50` matches config 6's own criterion exactly, so the standard
`strict=False` pattern is safe.

## Follow-up: config 7 -- and config 7b, continuing past an early stop

Also along the way: pulled the full online-augmentation project's
`test_mean_error_px` history from wandb to check where this series actually
stands against the pre-online-augmentation baseline (`unet-final-k5.ckpt`'s
own lineage, several independent runs all landing around 1.15-1.18px on the
old `load_datasets`-loaded split -- not confirmed to be the exact same test
images as `WingsRawDataset.split(seed=42)`, so treat as a rough reference,
not an exact one). Every online-augmentation config pays a real, consistent
cost on this specific metric relative to that baseline: even the series'
best result (configs 5/5b, 1.77-1.78px) is roughly 1.5x worse, and config 6
sits at 2.12px, roughly 1.8x worse.

**What's actually driving that cost -- mask radius or `rotation_p`?**
Config 6 changed both relative to 5b at once. Re-running notebook 25's
rotation-sweep logic against config 4d's checkpoint (radius 2,
`rotation_p=1.0` -- i.e. "half of config 6's change") isolates them:

| | radius | rotation_p | test_mean_error_px | GPA @ 90 deg (multistart) |
|---|---|---|---|---|
| 4d | 2 | 1.0 | 2.21px | 5.99px (NN alone: 4.87px -- a real detection-quality drop, not just ordering) |
| 5 / 5b | 2 | 0.3 / 0.8 | 1.77 / 1.78px | (5b, with pca_prealign) 8.61px |
| 6 | 4 | 1.0 | 2.12px | 0.84px |

At `rotation_p=1.0`, radius barely moves clean-image accuracy (4d's 2.21px
vs. 6's 2.12px -- comparable, if anything favoring the bigger radius) but
massively changes rotation robustness (4d's NN=4.87px vs. 6's NN=0.83px at
90 degrees -- the smaller mask target is fragile under rotation's resampling
blur regardless of how much rotated data it's trained on). Meanwhile,
holding radius fixed at 2, `rotation_p` alone tracks the clean-accuracy cost
directly (1.0 -> 2.21px, 0.3-0.8 -> 1.77-1.78px). These look like two
**independent** knobs -- radius for rotation robustness, `rotation_p` for
clean-image accuracy -- not one shared tradeoff, which opens up an untested
combination: **radius 4 + `rotation_p=0.8`** (a "config 8", not yet built --
parked until config 7/7b's results are in, to combine this with whatever
config 7's bigger triangle noise contributes rather than testing changes in
parallel).

**Config 7 itself early-stopped at epoch 36/100.** `EarlyStopping` and
`ReduceLROnPlateau` (`wings/modeling/litnet.py`'s `configure_optimizers`)
both monitor only `val_mean_error_px`, which plateaued almost immediately
(noisy, flat ~2.25-2.37px from epoch 0 onward) and triggered the
`patience=25` stop. But `val_wrong_spot_count_pct` -- the metric config 7's
whole premise targets (detection reliability, not positional precision) --
kept trending down over those same 36 epochs (8.08% -> mostly 7.0-7.4% by
the end, best single epoch 6.79%). This shows up directly in the final
numbers: config 7 (36 epochs) already beats config 6 (full 100 epochs) on
`test_wrong_spot_count_pct` (7.69% vs. 7.97%), for essentially the same
`test_mean_error_px` (2.14 vs. 2.12px) -- the run was stopped by a metric
that had already saturated, before the metric it actually targets finished
improving.

**Config 7b** continues from config 7's own checkpoint with
`early_stop_patience` raised (25 -> 45) -- otherwise identical (same loss,
mask, augmentation). Warm-starting resets `configure_optimizers`'s AdamW/
`ReduceLROnPlateau` to `lr=1e-5` from scratch, which also matters here:
config 7's own LR had likely already decayed several times by epoch 36
(`ReduceLROnPlateau`'s `patience=8` is much shorter than `EarlyStopping`'s),
so this isn't just a longer patience window on an already-shrunk LR, it's a
genuinely fresh step size to keep improving `val_wrong_spot_count_pct` with.

## Follow-up: config 7b's DeepWings result, and a notebook 26 bug

Notebook 26 on config 7b's checkpoint: mean positional precision **0.9373**
over all wings, unfiltered (paper: 0.943 -- only 0.0057 away), and **0.9542**
restricted to wings with a plausible (16-22) predicted point count --
already *above* the paper's average. First read of the reliable-point-count
percentage said 74.2%, which looked like a large, broad problem; that number
turned out to be stale (the notebook's `reliable` bounds had been changed to
18-22 temporarily to check something else, then changed back, without
re-running the cell, so the file's *source* said `[16, 22]` while its saved
*output* still reflected the older 18-22 run -- a plain Jupyter
edited-but-not-rerun state, not a code bug). Re-running the notebook against
the current, correct 16-22 bounds gives **90.5%** -- close to the paper's own
91.8% exact-19 rate (not quite apples-to-apples: we measure a wider 16-22
window, so our true exact-19 rate is somewhat lower than 90.5%, but the gap
is nowhere near what 74.2% suggested).

This reframes the remaining gap: it's small (0.0057 on the headline number)
and concentrated in the ~9.5% hard tail, not a broad problem across the
dataset. Inspecting that tail directly (notebook 26's worst-offenders
section, already computed) shows a consistent pattern across every example
checked: dense fields of small, dark, triangular debris covering the wing
*and* the surrounding background, causing gross **over-detection** (20-35
predicted points against 19 ground truth) -- the model treating debris
speckles as landmarks -- not the under-detection/merging this series started
out targeting. The debris density in these real photos visibly exceeds what
config 7b's augmentation (40-240 triangles, size 2-11) produces.

## Config 8 -- another gentle triangle-noise increase, informed by real DeepWings failures

A follow-up on config 7b's checkpoint. Given the gap above is now small and
narrow (one hard tail, not a broad problem), this is a deliberately gentle
step, same reasoning as config 7's own choice of 11 over an initially
considered 20 (see notebook 24): raises `n_triangles_range` (40-240 -> 60-300)
and `triangle_max_size` (11 -> 13) moderately rather than jumping straight to
match the worst-offender images' apparent density, which would risk
re-introducing the occlusion/legibility problem notebook 24 found at more
aggressive values.

| | Config 7b | Config 8 |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same |
| Augmentation | Aug B, `n_triangles_range=(40,240)`, `triangle_max_size=11` | Aug B, **`n_triangles_range=(60,300)`**, **`triangle_max_size=13`** |
| Mask | circular, `square_size=9` (radius 4) | same |
| Warm-start | config 7's own checkpoint | **config 7b's own checkpoint** |

Warm-started from config 7b's own trained checkpoint. No warm-start gotcha:
`pos_weight=50` matches config 7b's own criterion exactly, so the standard
`strict=False` pattern is safe.

## Config 8a-8c -- loss family: WeightedDiceLoss, never tried in this series

Three follow-ups on top of config 8's checkpoint, testing `WeightedDiceLoss`
(a Dice-based loss with per-pixel class weighting, `wings/modeling/loss.py`)
in place of `BCEDiceLoss` -- used in every config so far (1-8) but never
itself tried in the online-augmentation series.

**A sigmoid-pairing correction made while building these**: `BCEDiceLoss`
and `WeightedDiceLoss` need *opposite* `UNet(sigmoid=...)` settings.
`BCEDiceLoss.forward(logits, targets)` applies sigmoid itself
(`nn.BCEWithLogitsLoss` for its BCE term, an explicit `torch.sigmoid(logits)`
for its Dice term) -- it needs raw logits in, `sigmoid=False` (every config
1-8 uses this, correctly). `WeightedDiceLoss.forward(y_pred, y_true)` has no
sigmoid anywhere in its own code and uses `y_pred` directly -- it
mathematically expects actual [0, 1] probabilities, `sigmoid=True`. It first
looked like the original pre-online-augmentation baseline
(`wings/modeling/training/bced_unet.py`, ~1.15-1.18px test error) had already
validated `WeightedDiceLoss` paired with `sigmoid=False` (a mismatched but
apparently-proven combination worth reproducing faithfully) -- but
`git log --all -- wings/modeling/training/bced_unet.py` shows that file's
`WeightedDiceLoss(landmark_weight=100)` line was introduced in a commit dated
2026-06-27, over a month after the wandb runs that actually achieved
1.15-1.18px (created 2026-05-11 to 05-13). Those runs ran the version of the
file live at the time -- `BCEDiceLoss(pos_weight=50, dice_weight=0.8,
bce_weight=0.2)`, correctly paired with `sigmoid=False` -- and there's no
evidence the later `WeightedDiceLoss` line was ever actually trained to
completion. So there was no proven `WeightedDiceLoss`+`sigmoid=False` recipe
to preserve; configs 8a-8c use the mathematically correct pairing,
`sigmoid=True`.

| | 8a | 8b | 8c |
|---|---|---|---|
| Loss | `WeightedDiceLoss(landmark_weight=50)` | `WeightedDiceLoss(landmark_weight=75)` | `WeightedDiceLoss(landmark_weight=100)` |
| Everything else | Aug B (config 8's `n_triangles_range=(60,300)`, `triangle_max_size=13`), circular mask radius 4, `sigmoid=True` | same | same |
| Warm-start | config 8's own checkpoint | config 8's own checkpoint | config 8's own checkpoint |

All three are independent siblings warm-started from config 8 directly (not
chained to each other). `background_weight=1.0` (the class default) held
constant across all three -- only `landmark_weight` varies, mirroring how
configs 6a/6b mapped out `BCEDiceLoss`'s `pos_weight` response. `WeightedDiceLoss`
registers no torch buffers (`landmark_weight`/`background_weight` are plain
Python floats, not `nn.Module` buffers) -- unlike `BCEDiceLoss`'s `pos_weight`
(see `augmented_unet_4b.py`'s docstring for that gotcha) -- so there's no
overlapping criterion state to accidentally inherit from config 8's
checkpoint; `strict=False` is safe here for a different reason than usual (no
shared keys at all, not matching values at a shared key).

## Config 8d -- one step back down in mask radius (4 -> 3)

A follow-up on config 8's checkpoint: identical in every respect except mask
radius (`square_size` 9 -> 7, radius 4 -> 3). Motivated by the DeepWings
paper's own ablation (radius 3: 88.2% exact-19 accuracy, radius 4: 91.8% --
a real but not huge gap) together with config 8's own remaining DeepWings
worst-offenders, which include landmark blobs merging/overlapping under the
densest real debris (the same failure mode `mask_to_coords`'s watershed
split targets). A smaller radius leaves more gap between adjacent landmarks'
circles in the first place (roughly 2x the gap for the same landmark
spacing), at some cost to the rotation-robustness/reliability gains radius 4
brought over the project's earlier radius-2 configs (5/5b) -- this tests
where radius 3 actually lands now that full rotation_p, denser triangle
noise, and the watershed decoder fix are all already in place, rather than
re-deriving the radius choice in isolation the way config 6 originally did.

| | Config 8 | Config 8d |
|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | same |
| Augmentation | Aug B, `n_triangles_range=(60,300)`, `triangle_max_size=13` | same |
| Mask | circular, `square_size=9` (radius 4) | circular, **`square_size=7`** (radius 3) |
| Warm-start | config 7b's own checkpoint | **config 8's own checkpoint** |

Warm-started from config 8's own trained checkpoint. Mask radius only
affects target *data* generation, not the model architecture or the
criterion's own state -- unlike a `pos_weight` change, there's no buffer/key
overlap to worry about at all, so `strict=False` carries no gotcha in either
direction here.

## Follow-up: config 8a-8c -- a real bug, and a negative result

Two things came out of these three runs, one a bug in shared code, one a
genuine (negative) finding.

**Bug**: `wings/modeling/litnet.py`'s `compute_statistics`/`binary_stats`
both default to `output_is_logits=True`, and `validation_step`/`test_step`
called both without ever overriding it -- harmless for every config before
these three (1-8/8d, all `sigmoid=False`, so applying sigmoid once inside
these functions is correct), but 8a-8c are the first in this entire series
to use `sigmoid=True` (`WeightedDiceLoss` needs actual probabilities, see
above). Their model output is already a probability, so applying sigmoid
again pushes *every* pixel's value to >= 0.5 (sigmoid of anything in [0, 1]
is at least sigmoid(0) = 0.5, confirmed directly:
`torch.sigmoid(torch.tensor([0.0001, 0.9999]))` = `[0.500025, 0.73097]`,
both > 0.5) -- the entire predicted mask reads as positive regardless of
actual confidence. `val_wrong_spot_count_pct` got stuck near 100% for both
runs before this was caught, and -- worse -- `val_mean_error_px` (what both
`EarlyStopping` and `ReduceLROnPlateau` monitor) was equally meaningless, so
their early-stopping/LR-schedule decisions were corrupted from epoch 0.
Fixed by passing `output_is_logits=not self.model.sigmoid` explicitly at all
four call sites; for every `sigmoid=False` config this is a no-op
(`not False == True`, identical to the previous hardcoded default), so
nothing about configs 1-8/8d changes. All three runs were cancelled and
restarted fresh from config 8's checkpoint after the fix.

**Result, post-fix**: `WeightedDiceLoss` does not beat `BCEDiceLoss` here,
at any tested `landmark_weight`, and gets monotonically worse as
`landmark_weight` increases:

| | `val_mean_error_px` | `val_wrong_spot_count_pct` |
|---|---|---|
| config 8 (`BCEDiceLoss`, baseline) | 2.170 | 2.95% |
| 8a (`landmark_weight=50`) | 2.581 | 5.99% |
| 8b (`landmark_weight=75`) | 2.727 | 6.79% |
| 8c (`landmark_weight=100`) | 2.900 | 8.10% |

(`val_loss` itself is far lower for 8a-8c than for config 8, but that's not
a meaningful comparison -- different loss *formulas* have different scales
at convergence; `val_mean_error_px`/`val_wrong_spot_count_pct` are the
actual positional/detection metrics and the only fair comparison across loss
families.) Motivates configs 8e/8f below: return to `BCEDiceLoss` (proven
throughout 1-8/8d) and tune its own dice/bce ratio instead of switching loss
families.

## Config 8e-8f -- dice/bce ratio, on top of config 8 directly

Two follow-ups on config 8's checkpoint (not on 8a/8b/8c -- see above),
raising `BCEDiceLoss`'s `dice_weight` (0.5 -> 0.7 in 8e, -> 0.9 in 8f;
`bce_weight` shrinks to match, `pos_weight=50` unchanged). `dice_weight=0.8`
was planned once before as configs 6c/6d (on top of config 5b) but never
actually launched -- no wandb run exists for either name, dropped once the
project pivoted to config 6's radius-4 mask instead -- so this is a
genuinely new, untested direction, not a repeat of an old result.

| | Config 8 | Config 8e | Config 8f |
|---|---|---|---|
| Loss | `BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)` | `BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3)` | `BCEDiceLoss(pos_weight=50, dice_weight=0.9, bce_weight=0.1)` |
| Augmentation | Aug B, `n_triangles_range=(60,300)`, `triangle_max_size=13` | same | same |
| Mask | circular, `square_size=9` (radius 4) | same | same |
| Warm-start | config 7b's own checkpoint | **config 8's own checkpoint** | **config 8's own checkpoint** |

Both warm-started from config 8's own trained checkpoint, independent
siblings (not chained to each other). No warm-start gotcha: `pos_weight=50`
and the loss class itself match config 8's own criterion exactly, so the
standard `strict=False` pattern is safe.

## Follow-up: configs 8d/8e/8f all beat the paper *and* config 8

Notebook 26 positional precision (headline, all wings unfiltered), config 8
as the reference point:

| | precision | vs. config 8 |
|---|---|---|
| config 8 | 0.9463 | -- |
| config 8d (radius 3) | **0.9502** | +0.0039 |
| config 8e (dice_weight 0.7) | 0.9475 | +0.0012 |
| config 8f (dice_weight 0.9) | 0.9479 | +0.0016 |

All three beat both config 8 and the paper's 0.943. Radius gave the single
largest gain; dice_weight 0.7 -> 0.9 only gained +0.0004 more, a shrinking
return. Since radius and dice_weight were only ever changed one at a time
relative to config 8, it's still unknown whether they're independent
(stacking) or redundant (diminishing once combined) -- configs 9a-9e below
test exactly that, warm-started from 8d (not from config 8 directly), so the
model doesn't have to re-derive the radius-3 adaptation while also absorbing
a loss change.

## Config 9a-9e -- do radius and loss changes stack?

Five siblings, all warm-started from config 8d specifically (not config 8),
each changing exactly one more variable on top of 8d's radius 3:

| | `pos_weight` | `dice_weight` | warm-start mechanism |
|---|---|---|---|
| 9a | 50 | 0.7 (= config 8e's value) | standard `strict=False` |
| 9b | 50 | 0.9 (= config 8f's value) | standard `strict=False` |
| 9c | 50 (inert) | 1.0 (pure Dice, `bce_weight=0`) | standard `strict=False` |
| 9d | **100** | 0.5 (config 8d's own value) | **weights-only** (see below) |
| 9e | **100** | 0.7 | **weights-only** |

**9a/9b** directly test whether config 8e/8f's dice_weight gains and config
8d's radius gain stack, by combining each with radius 3 instead of radius 4.
**9c** pushes dice_weight to its extreme (pure Dice) since 8e -> 8f's gain
was already shrinking (+0.0004) -- checks whether the trend continues,
plateaus, or reverses at the limit, rather than assuming 0.9 was close
enough to the top. **9d** tests `pos_weight=100` -- planned once as configs
6a/6b (on top of config 5b) but never actually launched (no wandb run
exists, same as 6c/6d -- dropped for the same reason, the pivot to config
6's radius-4 mask) -- a genuinely untested axis, now combined with radius 3.
**9e** combines 9d's `pos_weight=100` with 9a's `dice_weight=0.7`, asking the
same stacking question one level further.

**9d/9e need the weights-only warm-start**, not the standard
`train(..., checkpoint_path, strict=False)` pattern the other three use:
`pos_weight` changes relative to config 8d's own criterion (50 -> 100), and
`BCEWithLogitsLoss`'s `pos_weight` is a persistent buffer under the same key
in both criteria -- `strict=False` only ignores missing/unexpected keys, it
does not stop a matching key's *value* from being overwritten by the
checkpoint's stored one (first found in `augmented_unet_4b.py`, same fix
applied in `augmented_unet_6a/6b/6d.py`). 9d/9e instead manually load only
the UNet's own weights from 8d's checkpoint and call `train(..., path=None)`.

## Noise fix: smoothed checkpoint selection (applies from config 10a onward)

Pulling wandb histories for the full 8/8a-8f/9a-9e sweep to pick a final
winner surfaced a problem with the selection process itself, not any one
config: the raw per-epoch `val_mean_error_px` that `EarlyStopping`/
`ModelCheckpoint`/the LR-plateau scheduler all monitor picked epoch 5 as
"best" for nearly every config (8, 8d, 8e, 8f, 9a, 9c, 9d, 9e), and the
same-run epoch-to-epoch noise in a +/-5-epoch window around that point
(std ~0.015-0.04px) is *larger* than the actual gap between configs' best
values (~0.001-0.003px) -- e.g. 9a's 2.1180 vs 9d's 2.1165 is well inside
one run's own noise band. The raw signal being used to pick "the best
epoch" can't reliably tell these configs apart at all.

`LitNet.on_validation_epoch_end` (`wings/modeling/litnet.py`) now also
computes and logs `val_mean_error_px_smooth`, `val_wrong_spot_count_pct_smooth`
and `val_loss_smooth` -- a trailing moving average over the last
`smooth_window` (default 5) validation epochs, maintained per-metric in
`deque`s rather than folded into one combined score, so a config can be
checked against all three independently (a model can be smoothed-good on
error but not on wrong_spot_count_pct, which matters just as much -- e.g.
9e tied 9a's test_mean_error_px but had a worse test_wrong_spot_count_pct).
`configure_optimizers`'s `ReduceLROnPlateau` and `train.py`'s
`EarlyStopping`/`ModelCheckpoint` now monitor `val_mean_error_px_smooth`
instead of the raw value. `ModelCheckpoint`'s `save_top_k` is raised from 2
to 5: the smoothed metric picks a more trustworthy single winner than the
raw one did, but the other two smoothed metrics can still rank a nearby
epoch differently, so keeping several real candidate checkpoints on disk
lets that be checked by hand instead of only ever having the one epoch the
callback's own primary metric picked.

## Config 10a -- continues 9a with the smoothed selection, no new hyperparameter

Warm-started from config 9a's own checkpoint. Loss/mask/augmentation
unchanged from 9a (`BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3)`,
circular mask radius 3, Aug B). Only `early_stop_patience` (45 -> 60) and
`num_epochs` (100 -> 150) change, giving the now-smoother signal more room
to keep improving before stopping rather than testing a new hyperparameter
-- this run is specifically about getting a trustworthy, noise-resistant
answer on top of the current best-evidenced lineage (8d -> 9a), not about
exploring a new direction.

## Config 10b -- 9a + stronger triangle noise

Also warm-started from config 9a's own checkpoint. Notebook 26's DeepWings
evaluation has consistently pointed at dense fields of small, dark
triangular debris as the dominant remaining cause of detection failures on
their noisiest real photos (the model over-detects debris as landmarks).
Continues the same gentle-increase pattern config 7->8 already used
(`triangle_max_size` 11->13, `n_triangles_range` 40-240->60-300) one more
step on top of 9a rather than jumping straight to match the worst offenders'
apparent density (config 7's own visual check in
`notebooks/24_online_augmentation.ipynb` found `triangle_max_size=20`
already started obscuring venation at the existing count ceiling):
`n_triangles_range` (60,300) -> (80,360), `triangle_max_size` 13 -> 16.
Loss and mask unchanged from 9a.

Not pursued for this round: pushing `pos_weight` further on top of 9a's
recipe. 9e already tested `pos_weight=100` combined with `dice_weight=0.7`
(the same combination, just on 8d's radius directly) and came out *worse*
on `test_wrong_spot_count_pct` than 9a alone (2.03% vs 1.89%) for the same
`test_mean_error_px` (2.014px both) -- no evidence that axis helps once
`dice_weight=0.7` is already in place.

## Held constant across configs (except where noted)

- Model: `UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)`
  (configs 8a-8c are the sole exception, `sigmoid=True` -- `WeightedDiceLoss`
  needs actual probabilities, not logits -- see their section above)
- Warm-start checkpoint: `models/new_unet/unet-final-k5.ckpt`, loaded with
  `strict=False` (its saved criterion state doesn't necessarily match every
  config's criterion here -- e.g. `BCEDiceLoss`'s `pos_weight` buffer isn't
  present in `WeightedDiceLoss` -- so that mismatched key is ignored while the
  actual UNet weights still load exactly; verified for both loss classes)
- Mask shape/size: square, `square_size=5` (config 4c changes size to 3;
  configs 4d, 5, 5b and 6a-6d change shape to circle at the same radius 2;
  config 6-8c change shape to circle *and* size, radius 4; config 8d back
  down to radius 3 -- see above)
- `rotation_p=1.0` (configs 5 at 0.3 and 5b/6a-6d at 0.8 are the exceptions;
  config 6 onward is back to the default 1.0 -- see above)
- `triangle_min_size=2` (dataclass default, unchanged everywhere);
  `triangle_max_size=6` (dataclass default; config 7/7b changes it to 11,
  config 8 onward to 13 -- see above)
- `n_triangles_range=(10,60)` (dataclass default; every config so far
  overrides this to `(40,240)`, config 8 onward to `(60,300)` -- see above)
- `num_epochs=100`, `batch_size=12`, `num_workers=8`
- `early_stop_patience=25` (configs 7b onward raise this to 45 -- see above),
  `early_stop_min_delta=0.01` (monitor: `val_mean_error_px`)
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

Same pattern for configs 4b/4c/4d/5b/6a/6b/6c/6d/7b/8a/8b/8c/8d/8e/8f/9a/9b/9c/9d/9e/10a/10b
(`N` = `"4b"`/`"4c"`/`"4d"`/`"5b"`/`"6a"`/`"6b"`/`"6c"`/`"6d"`/`"7b"`/`"8a"`/
`"8b"`/`"8c"`/`"8d"`/`"8e"`/`"8f"`/`"9a"`/`"9b"`/`"9c"`/`"9d"`/`"9e"`/`"10a"`/`"10b"`, e.g.
`unet-400-online-augmentation-k5-6a`).

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
