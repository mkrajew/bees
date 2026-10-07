# OBB wing detector, stage 1: labels, dataset and online augmentation

Date: 2026-10-07 · Branch: `yolo-improvements` · Status: design agreed in conversation, this document awaits review

## 1. Background and goal

The current wing detector (`models/yolo26n/best.pt`, YOLO26n, axis-aligned boxes) was trained on upright wings: Ultralytics rotation augmentation of at most ±20° (later ±10° and ±5°), horizontal flips (`fliplr` stays at its default of 0.5) and mosaic. On photos with arbitrary wing orientation it loses wings, and when it does find one, the axis-aligned box of a tilted wing contains a lot of empty background.

Goal of the whole effort: a detector that returns an oriented bounding box (OBB) aligned with the wing axis, works for any rotation (−180° to +180°) and is trained with augmentations applied online.

This document covers **stage 1 only**: the labels, the data on disk, the online augmentation pipeline with its Ultralytics integration, and the notebook that shows them. Training, evaluation and deployment are separate follow-up projects (section 11).

## 2. Decisions

| Topic | Decision | Reason |
|---|---|---|
| Orientation output | The detector gives the wing **axis** (angle modulo 180°). The direction base→tip is stored in the label table but not trained on. | Enough for a tight crop. `final-rotation-2` does not need the direction, and it can be added later without relabeling. |
| Box recipe | **PCA** of the 19 landmarks | Uses all points at once, so the angle is smooth and stable. |
| Source images | `data/raw/{country}-wing-images` and `data/raw/{country}-raw-coordinates.csv` for `COUNTRIES` (21,722 images). Not `data/processed/cropped`. | `cropped` is the output of the old detector and has lost the background the new detector must handle. The CSV coordinates are defined on the raw images. |
| Framework | Ultralytics YOLO26-OBB with our own dataset class and our own geometric pipeline | Built-in `degrees` clips polygons at the frame edge, fills with gray 114, and rotates all mosaic tiles together (section 6). |
| Staging | 1a single-wing samples, then 1b multi-wing compositions right after, then 1c frozen val/test | Order requested by the author. |

## 3. Labels: the PCA recipe

Input: the 19 landmarks `P` (19×2) of one image in top-left pixel coordinates (x to the right, y down). The CSV stores y from the bottom, so `y_top = img_h − y − 1`, as in `wings/detection/dataset.py:read_coordinates`.

1. `c = mean(P)`. `a` is the unit eigenvector of the covariance matrix with the largest eigenvalue. Sign convention: `a_x ≥ 0` (if `a_x = 0` then `a_y > 0`). `n = (−a_y, a_x)`.
2. Project: `u = (P − c)·a`, `v = (P − c)·n`. The box centre is `m = c + (u_min + u_max)/2 · a + (v_min + v_max)/2 · n`, the same midpoint-of-extremes rule as the current axis-aligned label.
3. Sides: `length = (u_max − u_min) · 1.2` along `a` and `width = (v_max − v_min) · 1.4` across. These are the factors the current labels use (`process_bbox`: x 1.2, y 1.4), so the crop stays comparable with the old detector.
4. Corners, clockwise in the (`a`, `n`) frame: `m + (−length/2)a + (−width/2)n`, `m + (+length/2)a + (−width/2)n`, `m + (+length/2)a + (+width/2)n`, `m + (−length/2)a + (+width/2)n`. Axis angle `theta = atan2(a_y, a_x)` in degrees, in (−90°, 90°].
5. `eig_ratio` = largest / smallest eigenvalue. It is stored and reported; a ratio below 1.5 marks an ill-defined axis. No image is dropped in stage 1.
6. `dir_sign` (±1): let `i_lo` and `i_hi` be the landmarks with the smallest and largest projection of the **mean shape** (`data/processed/mask_datasets/rectangle/mean_shape.pth`, same landmark order as the CSVs) on its own principal axis (same sign convention); the pair is computed once. Then `dir_sign = sign((P[i_hi] − P[i_lo]) · a)`. It is a consistent direction label, not used in training.

Ultralytics converts OBB polygons to `(cx, cy, w, h, r)` with `cv2.minAreaRect`, `w ≥ h` and `r` in [−π/4, 3π/4). Our own `corners → xywhr` function must agree with `ultralytics.utils.ops.xyxyxyxy2xywhr` (tested).

## 4. Data on disk

`data/processed/detection-obb/labels.csv` (inside the git-ignored `/data/`), one row per raw image of `COUNTRIES`:

| column | meaning |
|---|---|
| `file` | path relative to `data/raw` |
| `country`, `split` | country code; `train`, `val` or `test` |
| `img_w`, `img_h` | raw image size in pixels |
| `x1`, `y1`, …, `x4`, `y4` | OBB corners in pixels, order as in section 3 |
| `cx`, `cy`, `length`, `width`, `theta_deg` | box centre, side along the axis, side across, axis angle |
| `eig_ratio`, `dir_sign` | see section 3 |
| `bg_b`, `bg_g`, `bg_r` | background colour, BGR 0–255 |

- **Split.** Taken from the file names in `data/processed/detection/images/{train,val,test}` (17,401 / 2,197 / 2,124). The builder fails if a raw image appears in none or in more than one of them. This keeps val and test comparable with the old detector and removes the dependence on the unrecorded random draw. The assignment is copied into `labels.csv`, so the old folder is not needed afterwards.
- **Background colour.** `background_color(img)` follows `pad_image`: the most frequent colour of the pixel rows and columns at distance 5 px from the border; if that colour is pure black or pure white, the dominant colour of the inner border strip (`dominant_inner_border_color`). `pad_image` itself is left unchanged.
- **No image copies.** Training reads raw images in place.

### Frozen val and test sets

Generated once with the same augmentation code and a fixed seed, saved as a standard Ultralytics OBB dataset under `data/processed/detection-obb/{val,test}/{images,labels}`: JPEG quality 95, 640×640, labels as `0 x1 y1 x2 y2 x3 y3 x4 y4` normalized. Each split gets one single-wing sample per image of the split, and (from stage 1b) 500 multi-wing compositions built from images of the same split. A `dataset.yaml` is written with `train: train.txt` (absolute paths of the train images, only to satisfy the Ultralytics path check; the stage 2 trainer builds its own train dataset), `val`, `test`, `nc: 1` and `names: {0: wing}`. Regenerating with the same seed gives identical files.

## 5. Online augmentation pipeline

Implemented as pure functions in `wings/detection/obb_augment.py` and used by `WingOBBDataset`. Each call receives a row of `labels.csv` and a NumPy `Generator`. Numbers below are initial values, tuned in the notebook by looking at samples.

### 5.1 Single-wing sample (stage 1a)

1. Read the raw image at native resolution (BGR).
2. Horizontal flip with probability 0.5 (image and corners).
3. Rotation angle `phi ~ U(−180°, 180°)` about the OBB centre.
4. Scale `s = f · imgsz / length` with `f ~ U(0.25, 0.90)`. `f` is clamped so that the axis-aligned extent of the rotated box, `s · (length·|cos phi| + width·|sin phi|)`, is at most `imgsz − 8`.
5. Translation: the rotated, scaled box centre is drawn uniformly from the region where its axis-aligned extent lies inside `[4, imgsz − 4]` on both axes. The whole OBB is therefore always inside the canvas and never clipped.
6. One `cv2.warpAffine` into the `imgsz × imgsz` canvas (`imgsz = 640`), `BORDER_CONSTANT`, `borderValue` = the image's own background colour. If `s < 1` the image is first reduced with `INTER_AREA`, then warped with `INTER_LINEAR`. The four corners go through the same matrix, so the label is exact.
7. Photometric steps on the canvas, in this order: brightness and contrast factors `U(0.5, 1.5)` each (as in `TrainAugmentConfig`); hue ±0.015 and saturation factor `U(0.6, 1.4)`; triangle debris, `U(40, 200)` dark triangles of size 2–9 px (port of `TriangleNoise` to NumPy); Gaussian blur with probability 0.15 (sigma `U(0.3, 1.2)`); JPEG re-compression with probability 0.2 (quality `U(60, 95)`).

### 5.2 Multi-wing composition (stage 1b)

With probability `p_multi = 0.4` (training), a sample contains `k` wings, `k` in {2, 3, 4} with relative weights 0.5, 0.25, 0.25. The canvas is split into a grid: 1×2 or 2×1 for `k = 2`, 2×2 for `k = 3` (one random cell empty) and `k = 4`. Every wing comes from a different random image, gets its own flip, rotation and scale (5.1 steps 2–4, with the cell instead of the canvas, 4 px padding) and a random position inside its cell, so OBBs cannot overlap. Each wing is pasted only where its warped raw rectangle and its cell overlap. The canvas colour is the background colour of a randomly chosen wing. Photometric steps (5.1 step 7) run once on the whole canvas. Mitigation if seams between differing backgrounds show up in the notebook: feather the pasted edge over about 8 px.

### 5.3 Randomness

`make_sample(row_or_rows, rng)` is pure. `WingOBBDataset.__getitem__` creates its generator from the torch RNG (`np.random.default_rng(int(torch.randint(2**31, ())))`), which PyTorch reseeds per DataLoader worker, so workers never repeat each other (same reasoning as `wings/transforms.py`). The notebook and the frozen sets call `make_sample` with explicit seeds.

## 6. Ultralytics integration (version 8.4.41, checked in the installed sources)

Facts the design relies on:

- `Format` (`data/augment.py`, `return_obb=True`) computes the training boxes with `ops.xyxyxyxy2xywhr(instances.segments)`: `cv2.minAreaRect` on the polygon, `w ≥ h`, angle in [−π/4, 3π/4). The orientation is therefore known modulo 180°.
- `YOLODataset.update_labels_info` resamples every OBB polygon to 100 points.
- `RandomPerspective.apply_segments` clips transformed polygons to the visible region and fills with gray 114. `v8_transforms` applies one `RandomPerspective` to the whole mosaic, so all tiles share one rotation.
- `YOLODataset.build_transforms` is where the transform list (ending with `Format`) is created, and `DetectionTrainer.build_dataset`, inherited by `OBBTrainer`, is where the trainer creates the dataset.
- The YOLO26 OBB head is `OBB26`; `yolo26-obb.yaml` ships with the package. An end-to-end export returns rows `[x, y, w, h, conf, cls, angle]`.

`WingOBBDataset(YOLODataset)` in `wings/detection/obb_augment.py`:

- `get_labels` builds the label list from `labels.csv` for one split (no `.txt` files next to raw images, no Ultralytics cache).
- `update_labels_info` keeps the four corners as they are.
- `build_transforms` returns our pipeline followed by the stock `Format`, built with the same arguments the stock `build_transforms` passes (`bbox_format="xywh"`, `normalize=True`, `return_obb=True`, `batch_idx=True`), so `collate_fn` and the loss inputs are the stock ones.
- `img_path` is the `train.txt` file from section 4 (a list of absolute raw image paths), which `BaseDataset` accepts; `cache` stays off. The label dict handed to `Format` has the same keys Ultralytics produces (`img`, `instances`, `cls`, `im_file`, `ori_shape`, `resized_shape`, `ratio_pad`); the exact set is confirmed against 8.4.41 by the real-`DataLoader` test.
- It reads images itself (native resolution) instead of using `load_image`, which would shrink them to `imgsz` first.

Not part of stage 1 (stage 2): the `OBBTrainer` subclass whose `build_dataset` returns `WingOBBDataset` for training and a stock dataset for validation, and the training arguments that switch off Ultralytics' own geometric and colour augmentations so nothing is applied twice.

## 7. Notebook `notebooks/31_obb_dataset_and_augmentations.ipynb`

1. **Labels.** OBB overlays with landmarks on a dozen raw images; histogram of `theta_deg` over the dataset (raw images are almost horizontal); check that all 19 landmarks lie inside their box; histogram of `eig_ratio`; PCA axis compared with the Procrustes rotation to the mean shape (difference distribution); stability of the PCA angle under 1 px landmark jitter.
2. **One wing, step by step.** Original → flip → rotation → scale and placement → photometric steps, with the OBB drawn at every step.
3. **Sample grid and histograms** over several thousand samples: applied rotation angle (flat on −180°…180°), box angle modulo 180° (flat), wing fraction `f`, centre positions.
4. **Real batch.** A `DataLoader` over `WingOBBDataset` with the stock `collate_fn`; polygons drawn from the `xywhr` tensor that would enter the loss versus our own polygons, with an assertion that the maximum corner error is below 0.5 px.
5. **(1b) Multi-wing compositions.** Sample grid, distribution of `k`, check that no two boxes overlap and all lie inside the canvas.
6. **Frozen val/test.** Examples and a check that regenerating with the same seed reproduces identical files (hash comparison).

## 8. Tests (`tests/`, run with `uv run pytest tests`)

No GPU and no real data; synthetic images and landmark sets only. `testpaths` in `pyproject.toml` stays on the benchmarks, so a bare `pytest` is unchanged.

- `tests/test_obb_labels.py`: all points inside the box; rotating the landmarks by α rotates `theta` by α modulo 180°; margins equal extent × factor; independence of point order; agreement of our `corners → xywhr` with `ops.xyxyxyxy2xywhr`; `dir_sign` flips when the landmark set is rotated by 180°.
- `tests/test_obb_augment.py`: corners after flip, rotation and scaling equal the analytic result; the box always lies inside the canvas; canvas pixels outside the warped image equal the background colour exactly; the same seed gives the same sample and different seeds differ; (1b) boxes in a composition do not overlap and `k` matches the number of labels.
- A dataset test builds a tiny synthetic `labels.csv` plus images in a temporary directory and checks the shapes and keys of `WingOBBDataset[i]` and of a collated batch.

## 9. Files

- `wings/detection/obb_labels.py` (geometry, `background_color`)
- `wings/detection/obb_dataset.py` (builds `labels.csv`, writes the frozen sets; `typer` CLI like the other scripts in `wings/detection`)
- `wings/detection/obb_augment.py` (pipeline, composition, `WingOBBDataset`)
- `notebooks/31_obb_dataset_and_augmentations.ipynb`
- `tests/test_obb_labels.py`, `tests/test_obb_augment.py`

No new dependencies (OpenCV, NumPy, pandas, SciPy, Ultralytics and PyTorch are already installed).

## 10. Acceptance criteria for stage 1

1. `uv run pytest tests` passes.
2. `labels.csv` has 21,722 rows with split sizes 17,401 / 2,197 / 2,124, and every row has all 19 landmarks inside its box (tolerance 0.01 px).
3. The notebook runs from top to bottom on the local data. The real-batch check reports a maximum corner error below 0.5 px.
4. Over 5,000 samples, the Kolmogorov–Smirnov test does not reject uniformity (p > 0.01) of both the applied rotation angle and the box angle modulo 180°.
5. Regenerating the frozen val and test sets with the same seed reproduces identical files.
6. Stage 1b: compositions never overlap or leave the canvas (checked over 5,000 samples).

## 11. Out of scope (separate follow-up projects)

1. **Training:** `OBBTrainer` subclass, `train_obb.py`, HPC job, initialisation from the existing `best.pt` or from `yolo26n-obb.pt` (downloading the latter needs explicit approval).
2. **Evaluation:** old detector versus OBB on the frozen rotated test set, Pulawy and the DeepWings test set (OBB ground truth from their landmarks with the same recipe): rotated IoU and mAP, angle error versus angle, share of landmarks inside the box, crop emptiness, and end-to-end landmark error with `final-rotation-2`.
3. **Deployment:** ONNX export, the decoder and rotated crop in `wingai-app`, mapping landmarks back to the original image.
4. Option 2 from the discussion (axis plus direction, e.g. two classes) and regenerating `data/processed/cropped` with oriented crops.

## 12. Risks and open parameters

- **Ultralytics internals.** Version pinned in `uv.lock`; the real-`DataLoader` assertion in the notebook and the dataset test guard against drift.
- **Seams between backgrounds** in multi-wing compositions: checked in the notebook, mitigation in 5.2.
- **Label noise from a poorly defined axis** (`eig_ratio` below 1.5): reported, not dropped. A decision is taken only if such rows turn out to exist.
- **Parameters tuned in the notebook:** range of `f`, `p_multi` and the distribution of `k`, debris density and size, blur and JPEG probabilities.
- **Start-up cost:** `labels.csv` needs one pass over 21,722 images (background colour), expected to take a few minutes.
