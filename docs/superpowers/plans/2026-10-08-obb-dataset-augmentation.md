# OBB Wing Detector, Stage 1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the oriented-box labels, the online augmentation pipeline (single wings first, multi-wing compositions right after), the Ultralytics training dataset class and the frozen validation and test sets of an OBB wing detector, plus the notebook that shows them.

**Architecture:** `obb_labels.py` holds pure geometry: the 19 landmarks become a PCA oriented box. `obb_augment.py` applies ONE affine warp per wing (flip, rotation about the box centre, scale, translation) to the raw image and the box corners together, then photometric steps; `WingOBBDataset` hands the result to the stock Ultralytics `Format`, so batches are ordinary OBB batches. `obb_dataset.py` builds the label table from the raw data and writes the frozen val/test sets with the same augmentation code.

**Tech Stack:** Python 3.12, NumPy, OpenCV, pandas, SciPy, PyTorch, Ultralytics 8.4.41 (`YOLODataset`, `Format`), typer, loguru, pytest, nbformat (scratch notebook builder only).

**Spec:** `docs/superpowers/specs/2026-10-07-obb-dataset-augmentation-design.md` (read it first; this plan implements it and argues from it).

## Global Constraints

- Source images are `data/raw/{country}-wing-images` and `data/raw/{country}-raw-coordinates.csv` for `COUNTRIES` (21,722 images). Never `data/processed/cropped`.
- Box margins: `length = (u_max − u_min) · 1.2` along the axis, `width = (v_max − v_min) · 1.4` across, centre at the midpoint of the extreme projections.
- Canvas `imgsz = 640`; rotation uniform on −180°…+180°; the whole OBB always inside the canvas and never clipped; areas outside the source image are filled with that image's own background colour (never gray 114 or black).
- Frozen val/test sets: JPEG quality 95, 640×640, labels `0 x1 y1 x2 y2 x3 y3 x4 y4` normalized, one single-wing sample per image of the split plus 500 multi-wing compositions.
- Tests run with `uv run pytest tests`; `testpaths` in `pyproject.toml` stays on the benchmarks; tests need no GPU and no real data. No new dependencies; Ultralytics stays at the version in `uv.lock`.
- The author runs notebooks. Never execute a whole notebook: smoke-test its cells on synthetic data (Task 7).
- Commit messages: imperative sentence case like the existing history, plus the trailer as a second paragraph: `git commit -m "Subject" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"`.
- Work in the repository root on branch `yolo-improvements`. Match the surrounding code style (type hints, docstrings, `loguru` for logging, `typer` for scripts).

## Review Focus

Failure modes the spec implies but a happy-path test would not catch. Each has a test pinned to the task that owns the code.

1. A raw image that cannot be read, a CSV row with non-numeric landmarks, or an image missing from (or doubled in) the split folders must stop the build with an error naming the file, never silently drop the row. Pinned in Task 2.
2. Degenerate landmark sets (identical, collinear, NaN, fewer than three points) must raise `ValueError`, not produce NaN corners that poison the warp. Pinned in Task 1.
3. Extreme sources and placements: a 100×40 px image, a 6000×2400 px image, a box larger than its own source image, and a box that exactly fills the canvas (no slack for the random position, where rounding once produced an empty range) must still give a valid in-canvas box. Pinned in Task 3.
4. A pure black or white frame line, or a tiny image, must not break the background-colour rule. Pinned in Task 1.
5. A labels table that disagrees with the file on disk (image size) must fail loudly, and a one-image split must still yield compositions. Pinned in Tasks 4 and 5.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `wings/detection/obb_labels.py` | Pure geometry: PCA oriented box, corners ↔ `xywhr`, direction sign, background colour, label-table column names | 1 |
| `wings/detection/obb_dataset.py` | Label table (`labels.csv`) and image lists from the raw data; frozen val/test sets; `typer` CLI (`build`, `freeze`) | 2, 6 |
| `wings/detection/obb_augment.py` | One-warp augmentation, photometric steps, compositions, `WingOBBDataset` | 3, 4, 5 |
| `tests/obb_synthetic.py`, `tests/conftest.py` | Synthetic wings and fixtures shared by the tests | 1, 2, 4 |
| `tests/test_obb_labels.py` | Tests of `obb_labels.py` | 1 |
| `tests/test_obb_dataset.py` | Tests of the label table builder and the frozen sets | 2, 6 |
| `tests/test_obb_augment.py` | Tests of the augmentation functions | 3, 5 |
| `tests/test_obb_dataset_class.py` | Tests of `WingOBBDataset` against the stock Ultralytics pipeline | 4, 5 |
| `notebooks/31_obb_dataset_and_augmentations.ipynb` | Shows labels, augmentations, the real batch, compositions, frozen sets | 7 |

Test counts after each task (`uv run pytest tests`): Task 1: 26, Task 2: 36, Task 3: 77, Task 4: 86, Task 5: 98, Task 6: 101.

**How edits are written.** "Find" text occurs exactly once in the file; replace it with the "replace with" text. "Append" means add at the end of the file after two blank lines. Whole files are given in full where a task creates or replaces them.

**Why helpers are imported as `obb_synthetic`, not `tests.…`:** a package named `tests` already exists in `site-packages`, so `from tests.conftest import …` finds the wrong one. pytest puts `tests/` itself on `sys.path`, hence `from obb_synthetic import …`.

---

### Task 1: Oriented box geometry

**Files:**
- Create: `wings/detection/obb_labels.py`
- Create: `tests/obb_synthetic.py`, `tests/conftest.py`
- Test: `tests/test_obb_labels.py`

**Interfaces:**
- Consumes: nothing.
- Produces (`wings/detection/obb_labels.py`):
  - constants `LENGTH_FACTOR = 1.2`, `WIDTH_FACTOR = 1.4`, `MIN_EIG_RATIO = 1.5`, `CORNER_COLUMNS: list[str]` (`x1, y1, …, x4, y4`);
  - `Obb(cx, cy, length, width, theta_deg, eig_ratio)` frozen dataclass with `.axis`, `.normal` and `.corners() -> np.ndarray` of shape (4, 2) in the order (−L/2,−W/2), (+L/2,−W/2), (+L/2,+W/2), (−L/2,+W/2) of the (axis, normal) frame;
  - `principal_axis(points) -> tuple[np.ndarray, float]` (unit vector with x ≥ 0, eigenvalue ratio);
  - `obb_from_landmarks(points, length_factor=1.2, width_factor=1.4) -> Obb`, raises `ValueError` on degenerate input;
  - `points_inside_box(corners, points, tol=1e-6) -> bool`;
  - `obb_to_xywhr(obb) -> (cx, cy, w, h, r)` in the Ultralytics convention (w ≥ h, r in [−π/4, 3π/4) radians);
  - `reference_landmarks(mean_shape) -> (i_lo, i_hi)` and `direction_sign(points, axis, i_lo, i_hi) -> int` (±1);
  - `background_color(image, d=5, offset=2, width=5, q=8) -> tuple[int, int, int]` (BGR).
- Test helpers: `make_landmarks(rng, centre, half_length, half_width, angle_deg)`, `make_wing_image(width, height, background)`; fixtures `rng`, `landmarks`, `wing_image`.

- [ ] **Step 1: Create the shared test helpers**

`tests/obb_synthetic.py`:

```python
"""Synthetic wings for the OBB tests (no GPU, no real data). Imported by name (`from obb_synthetic import ...`):
pytest puts this directory on sys.path, and a package called `tests` already exists in site-packages."""

import cv2
import numpy as np

N_LANDMARKS = 19


def make_landmarks(rng: np.random.Generator, centre=(400.0, 160.0), half_length=300.0, half_width=100.0, angle_deg=0.0) -> np.ndarray:
    """19 points scattered over an ellipse-like wing, rotated by angle_deg (top-left pixel coordinates).
    Points 0 and 1 sit at the two ends of the axis, so the extent along it is exactly 2 * half_length."""
    u = rng.uniform(-1, 1, N_LANDMARKS) * half_length
    v = rng.uniform(-1, 1, N_LANDMARKS) * half_width * np.sqrt(np.clip(1 - (u / half_length) ** 2, 0.05, 1))
    u[0], u[1] = -half_length, half_length
    v[0], v[1] = 0.0, 0.0
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.stack([u, v], axis=1) @ rot.T + np.asarray(centre)


def make_wing_image(width: int = 800, height: int = 326, background=(170, 180, 160)) -> np.ndarray:
    """BGR image with a flat background and a darker ellipse as the 'wing'."""
    img = np.full((height, width, 3), background, np.uint8)
    cv2.ellipse(img, (width // 2, height // 2), (int(width * 0.375), int(height * 0.3)), 0, 0, 360, (60, 70, 50), -1)
    return img
```

`tests/conftest.py`:

```python
import numpy as np
import pytest
from obb_synthetic import make_landmarks, make_wing_image


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def landmarks(rng):
    return make_landmarks(rng, angle_deg=7.0)


@pytest.fixture
def wing_image():
    return make_wing_image()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_obb_labels.py`. It pins Review Focus 2 (`test_degenerate_landmarks_are_rejected`) and 4 (`test_background_colour_ignores_a_pure_white_frame_line`, `test_background_colour_of_a_tiny_image_does_not_crash`):

```python
import numpy as np
import pytest
from obb_synthetic import make_landmarks
from ultralytics.utils import ops

from wings.detection.obb_labels import (
    Obb,
    background_color,
    direction_sign,
    obb_from_landmarks,
    obb_to_xywhr,
    points_inside_box,
    principal_axis,
    reference_landmarks,
)


def test_box_contains_every_landmark(landmarks):
    obb = obb_from_landmarks(landmarks)
    assert points_inside_box(obb.corners(), landmarks)


def test_points_inside_box_notices_a_point_outside(landmarks):
    corners = obb_from_landmarks(landmarks).corners()
    outside = np.vstack([landmarks, corners[0] - 5.0 * (corners[1] - corners[0]) / np.linalg.norm(corners[1] - corners[0])])
    assert not points_inside_box(corners, outside)
    assert points_inside_box(corners, outside, tol=6.0)


def test_margins_scale_the_extents(landmarks):
    tight = obb_from_landmarks(landmarks, 1.0, 1.0)
    obb = obb_from_landmarks(landmarks, 1.2, 1.4)
    assert obb.length == pytest.approx(tight.length * 1.2)
    assert obb.width == pytest.approx(tight.width * 1.4)
    assert (obb.cx, obb.cy) == pytest.approx((tight.cx, tight.cy))


@pytest.mark.parametrize("alpha", [-120.0, -35.0, 12.0, 80.0, 171.0])
def test_rotating_the_landmarks_rotates_the_axis(rng, alpha):
    base = make_landmarks(rng, angle_deg=5.0)
    centre = base.mean(axis=0)
    a = np.deg2rad(alpha)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = (base - centre) @ rot.T + centre
    t0, t1 = obb_from_landmarks(base).theta_deg, obb_from_landmarks(rotated).theta_deg
    diff = (t1 - t0 - alpha + 90.0) % 180.0 - 90.0  # equal modulo 180 degrees
    assert abs(diff) < 1e-6


def test_point_order_does_not_matter(landmarks, rng):
    a = obb_from_landmarks(landmarks)
    b = obb_from_landmarks(landmarks[rng.permutation(len(landmarks))])
    assert (a.cx, a.cy, a.length, a.width, a.theta_deg) == pytest.approx((b.cx, b.cy, b.length, b.width, b.theta_deg))


def test_axis_sign_convention(rng):
    for angle in (-80.0, -10.0, 10.0, 80.0, 100.0, 170.0):
        axis, _ = principal_axis(make_landmarks(rng, angle_deg=angle))
        assert axis[0] > 0 or (abs(axis[0]) < 1e-9 and axis[1] > 0)


@pytest.mark.parametrize("angle", [0.0, 7.0, 40.0, 89.0, 100.0, 133.0, 175.0])
def test_xywhr_agrees_with_ultralytics(rng, angle):
    obb = obb_from_landmarks(make_landmarks(rng, angle_deg=angle))
    ours = np.array(obb_to_xywhr(obb))
    theirs = ops.xyxyxyxy2xywhr(obb.corners().astype(np.float32).reshape(1, 8))[0]
    assert ours[:4] == pytest.approx(theirs[:4], abs=1e-2)
    assert abs((ours[4] - theirs[4] + np.pi / 2) % np.pi - np.pi / 2) < 1e-3
    assert -np.pi / 4 <= ours[4] < 3 * np.pi / 4


def test_xywhr_swaps_sides_when_the_width_is_the_longer_one():
    obb = Obb(10.0, 20.0, 30.0, 80.0, 10.0, 5.0)
    cx, cy, w, h, r = obb_to_xywhr(obb)
    assert (w, h) == (80.0, 30.0)
    assert r == pytest.approx(np.deg2rad(100.0))


def test_direction_sign_flips_when_the_wing_is_turned_by_180(rng):
    mean_shape = make_landmarks(rng, centre=(0.0, 0.0), angle_deg=0.0)
    i_lo, i_hi = reference_landmarks(mean_shape)
    assert {i_lo, i_hi} == {0, 1}  # the two points placed at the extreme ends in make_landmarks
    wing = make_landmarks(np.random.default_rng(5), angle_deg=15.0)
    turned = 2 * wing.mean(axis=0) - wing  # rotation by 180 degrees about the centroid
    sign = direction_sign(wing, principal_axis(wing)[0], i_lo, i_hi)
    assert sign == -direction_sign(turned, principal_axis(turned)[0], i_lo, i_hi)
    assert sign in (-1, 1)


@pytest.mark.parametrize(
    "points",
    [np.zeros((19, 2)), np.column_stack([np.arange(19.0), 2 * np.arange(19.0)]), np.full((19, 2), np.nan), np.zeros((2, 2))],
    ids=["identical", "collinear", "nan", "too_few"],
)
def test_degenerate_landmarks_are_rejected(points):
    with pytest.raises(ValueError):
        obb_from_landmarks(points)


def test_background_colour_of_a_flat_image(wing_image):
    assert background_color(wing_image) == (170, 180, 160)


def test_background_colour_ignores_a_pure_white_frame_line():
    img = np.full((120, 300, 3), (120, 130, 140), np.uint8)
    img[5, :] = 255  # white row exactly where the frame is sampled
    img[-6, :] = 255
    img[:, 5] = 255
    img[:, -6] = 255
    assert background_color(img) != (255, 255, 255)


def test_background_colour_of_a_tiny_image_does_not_crash():
    assert background_color(np.full((8, 12, 3), 77, np.uint8)) == (77, 77, 77)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/test_obb_labels.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'wings.detection.obb_labels'`.

- [ ] **Step 4: Implement the geometry**

`wings/detection/obb_labels.py`:

```python
"""Oriented bounding boxes (OBB) of wings, computed from the 19 landmarks with the PCA recipe.

Coordinates are image pixels with the origin in the top-left corner (x to the right, y down).
The box is described by its centre, its side along the wing axis (`length`), its side across
(`width`) and the axis angle. The margins match the axis-aligned labels of the old detector
(`wings.detection.dataset.process_bbox`: x 1.2, y 1.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LENGTH_FACTOR = 1.2  # margin along the wing axis
WIDTH_FACTOR = 1.4  # margin across the wing axis
MIN_EIG_RATIO = 1.5  # below this the principal axis is ill-defined (reported, never enforced)
CORNER_COLUMNS = [f"{axis}{i}" for i in range(1, 5) for axis in ("x", "y")]  # label-table columns x1, y1, ..., x4, y4


@dataclass(frozen=True)
class Obb:
    """Oriented box. `theta_deg` is the axis angle in (-90, 90]; `eig_ratio` is the ratio of the
    two covariance eigenvalues of the landmarks (large = well-defined axis)."""

    cx: float
    cy: float
    length: float
    width: float
    theta_deg: float
    eig_ratio: float

    @property
    def axis(self) -> np.ndarray:
        t = np.deg2rad(self.theta_deg)
        return np.array([np.cos(t), np.sin(t)])

    @property
    def normal(self) -> np.ndarray:
        a = self.axis
        return np.array([-a[1], a[0]])

    def corners(self) -> np.ndarray:
        """(4, 2) corners, in the (axis, normal) frame: (-L/2,-W/2), (+L/2,-W/2), (+L/2,+W/2), (-L/2,+W/2)."""
        a, n = self.axis, self.normal
        centre = np.array([self.cx, self.cy])
        hl, hw = self.length / 2.0, self.width / 2.0
        return np.stack([centre - hl * a - hw * n, centre + hl * a - hw * n, centre + hl * a + hw * n, centre - hl * a + hw * n])


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit vector of the largest covariance eigenvalue (sign: x >= 0, and y > 0 if x == 0) and
    the ratio largest / smallest eigenvalue."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError(f"expected at least 3 points of shape (n, 2), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("non-finite landmark coordinates")
    eigvals, eigvecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))  # ascending order
    if eigvals[-1] <= 1e-12:
        raise ValueError("degenerate landmark set: no spread")
    axis = eigvecs[:, -1]
    if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
        axis = -axis
    return axis, float(eigvals[-1] / max(eigvals[0], 1e-12))


def obb_from_landmarks(points: np.ndarray, length_factor: float = LENGTH_FACTOR, width_factor: float = WIDTH_FACTOR) -> Obb:
    """PCA recipe: axis = direction of the largest spread; the box spans the extreme projections
    of the points on the axis and on its normal, enlarged by the margin factors about the midpoint."""
    pts = np.asarray(points, dtype=np.float64)
    axis, ratio = principal_axis(pts)
    normal = np.array([-axis[1], axis[0]])
    centroid = pts.mean(axis=0)
    u = (pts - centroid) @ axis
    v = (pts - centroid) @ normal
    centre = centroid + (u.min() + u.max()) / 2.0 * axis + (v.min() + v.max()) / 2.0 * normal
    length = (u.max() - u.min()) * length_factor
    width = (v.max() - v.min()) * width_factor
    if width <= 1e-6 * length:
        raise ValueError("landmarks are collinear: the box has no width")
    theta = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return Obb(float(centre[0]), float(centre[1]), float(length), float(width), theta, ratio)


def points_inside_box(corners: np.ndarray, points: np.ndarray, tol: float = 1e-6) -> bool:
    """True if every point lies inside the rectangle given by `corners` (order as in `Obb.corners()`), within `tol` px."""
    a, n = corners[1] - corners[0], corners[3] - corners[0]
    rel = np.asarray(points, dtype=np.float64) - corners[0]
    s, t = rel @ a / (a @ a), rel @ n / (n @ n)
    slack_s, slack_t = tol / np.linalg.norm(a), tol / np.linalg.norm(n)
    return bool(((s >= -slack_s) & (s <= 1 + slack_s) & (t >= -slack_t) & (t <= 1 + slack_t)).all())


def obb_to_xywhr(obb: Obb) -> tuple[float, float, float, float, float]:
    """(cx, cy, w, h, r) in the Ultralytics convention: w >= h, r in radians within [-pi/4, 3*pi/4)."""
    w, h, r = obb.length, obb.width, float(np.deg2rad(obb.theta_deg))
    if w < h:
        w, h = h, w
        r += np.pi / 2
    while r >= 3 * np.pi / 4:
        r -= np.pi
    while r < -np.pi / 4:
        r += np.pi
    return obb.cx, obb.cy, w, h, r


def reference_landmarks(mean_shape: np.ndarray) -> tuple[int, int]:
    """Indices (i_lo, i_hi) of the mean-shape landmarks with the smallest and largest projection on
    its principal axis. They define a consistent wing direction for every image."""
    shape = np.asarray(mean_shape, dtype=np.float64)
    axis, _ = principal_axis(shape)
    projection = (shape - shape.mean(axis=0)) @ axis
    return int(projection.argmin()), int(projection.argmax())


def direction_sign(points: np.ndarray, axis: np.ndarray, i_lo: int, i_hi: int) -> int:
    """+1 if the vector from landmark i_lo to landmark i_hi points along `axis`, else -1."""
    pts = np.asarray(points, dtype=np.float64)
    return 1 if float((pts[i_hi] - pts[i_lo]) @ axis) >= 0.0 else -1


def background_color(image: np.ndarray, d: int = 5, offset: int = 2, width: int = 5, q: int = 8) -> tuple[int, int, int]:
    """BGR background colour of a wing image, by the same rule as `wings.detection.dataset.pad_image`:
    the most frequent colour of the pixel rows and columns at distance `d` from the border; if that is
    pure black or pure white, the dominant colour of the inner border strip."""
    h, w = image.shape[:2]
    d = max(0, min(d, h // 2 - 1, w // 2 - 1))
    frame = np.concatenate([image[d], image[-d - 1], image[:, d], image[:, -d - 1]])
    colors, counts = np.unique(frame, axis=0, return_counts=True)
    color = tuple(int(c) for c in colors[counts.argmax()])
    if color in [(255, 255, 255), (0, 0, 0)]:
        from wings.detection.dataset import dominant_inner_border_color

        color = dominant_inner_border_color(image, offset=offset, width=width, q=q)
    return color
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/test_obb_labels.py -v`
Expected: `26 passed`.

- [ ] **Step 6: Commit**

```bash
git add wings/detection/obb_labels.py tests/obb_synthetic.py tests/conftest.py tests/test_obb_labels.py
git commit -m "Add the PCA oriented box geometry for wing labels" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Label table and image lists

**Files:**
- Create: `wings/detection/obb_dataset.py`
- Modify: `tests/obb_synthetic.py` (replace the file), `tests/conftest.py` (replace the file)
- Test: `tests/test_obb_dataset.py`

**Interfaces:**
- Consumes: Task 1 (`obb_from_landmarks`, `reference_landmarks`, `direction_sign`, `principal_axis`, `points_inside_box`, `background_color`).
- Produces (`wings/detection/obb_dataset.py`):
  - constants `SPLITS = ("train", "val", "test")`, `IMAGE_EXTENSIONS`, `DEFAULT_OUT_DIR` (`data/processed/detection-obb`), `DEFAULT_SPLIT_DIR` (`data/processed/detection`), `DEFAULT_MEAN_SHAPE`;
  - `read_split_map(split_dir) -> dict[str, str]` (file name → split; raises `ValueError` if a name is in two splits);
  - `build_labels(raw_dir, countries, split_dir, mean_shape, coords_sufx=COORDS_SUFX, img_sufx=IMG_FOLDER_SUFX) -> pd.DataFrame` with columns `file, country, split, img_w, img_h, x1, y1, …, x4, y4, cx, cy, length, width, theta_deg, eig_ratio, dir_sign, bg_b, bg_g, bg_r` (`file` is relative to `data/raw`, e.g. `AT-wing-images/<name>`);
  - `write_image_lists(table, raw_dir, out_dir)` writes `train.txt`, `val.txt`, `test.txt` (absolute paths);
  - `typer` command `build`.
- Test helpers: `build_synthetic_raw(root, plan=None)` (two synthetic countries with CSVs in the real layout, the old detector's split folders, a mean shape); fixture `synthetic_raw`.

- [ ] **Step 1: Replace the test helpers**

`tests/obb_synthetic.py` (replace the whole file):

```python
"""Synthetic wings for the OBB tests (no GPU, no real data). Imported by name (`from obb_synthetic import ...`):
pytest puts this directory on sys.path, and a package called `tests` already exists in site-packages."""

from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

N_LANDMARKS = 19
SPLIT_PLAN = {"AA": ["train", "train", "train", "train", "val", "test"], "BB": ["train", "train", "val", "test"]}


def make_landmarks(rng: np.random.Generator, centre=(400.0, 160.0), half_length=300.0, half_width=100.0, angle_deg=0.0) -> np.ndarray:
    """19 points scattered over an ellipse-like wing, rotated by angle_deg (top-left pixel coordinates).
    Points 0 and 1 sit at the two ends of the axis, so the extent along it is exactly 2 * half_length."""
    u = rng.uniform(-1, 1, N_LANDMARKS) * half_length
    v = rng.uniform(-1, 1, N_LANDMARKS) * half_width * np.sqrt(np.clip(1 - (u / half_length) ** 2, 0.05, 1))
    u[0], u[1] = -half_length, half_length
    v[0], v[1] = 0.0, 0.0
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.stack([u, v], axis=1) @ rot.T + np.asarray(centre)


def make_wing_image(width: int = 800, height: int = 326, background=(170, 180, 160)) -> np.ndarray:
    """BGR image with a flat background and a darker ellipse as the 'wing'."""
    img = np.full((height, width, 3), background, np.uint8)
    cv2.ellipse(img, (width // 2, height // 2), (int(width * 0.375), int(height * 0.3)), 0, 0, 360, (60, 70, 50), -1)
    return img


def build_synthetic_raw(root, plan=None) -> SimpleNamespace:
    """Two 'countries' of raw wings with CSVs (y from the bottom, as in the real data), the old detector's
    split folders (empty files named like the images) and a mean shape in the same landmark order."""
    plan = SPLIT_PLAN if plan is None else plan
    rng = np.random.default_rng(42)
    raw, split_dir = root / "raw", root / "detection"
    for split in ("train", "val", "test"):
        (split_dir / "images" / split).mkdir(parents=True)
    names, landmarks, backgrounds = [], {}, {}
    for country, splits in plan.items():
        (raw / f"{country}-wing-images").mkdir(parents=True)
        records = []
        for i, split in enumerate(splits):
            name = f"{country}-{i:04d}.png"
            width, height = int(rng.integers(600, 900)), int(rng.integers(250, 350))
            background = tuple(int(c) for c in rng.integers(100, 200, 3))
            image = make_wing_image(width, height, background)
            points = make_landmarks(rng, centre=(width / 2, height / 2), half_length=0.35 * width, half_width=0.28 * height, angle_deg=float(rng.uniform(-8, 8)))
            cv2.imwrite(str(raw / f"{country}-wing-images" / name), image)
            (split_dir / "images" / split / name).touch()
            csv_xy = np.column_stack([points[:, 0], height - 1 - points[:, 1]]).reshape(-1)
            records.append([name, *csv_xy.tolist()])
            names.append(name)
            landmarks[name], backgrounds[name] = points, background
        columns = ["file"] + [f"{a}{k}" for k in range(1, 20) for a in "xy"]
        pd.DataFrame(records, columns=columns).to_csv(raw / f"{country}-raw-coordinates.csv", index=False)
    mean_shape = make_landmarks(np.random.default_rng(0), centre=(0.0, 0.0), half_length=1.0, half_width=0.35, angle_deg=0.0)
    return SimpleNamespace(raw=raw, split_dir=split_dir, countries=list(plan), names=names, landmarks=landmarks, backgrounds=backgrounds, mean_shape=mean_shape)
```

`tests/conftest.py` (replace the whole file):

```python
import numpy as np
import pytest
from obb_synthetic import build_synthetic_raw, make_landmarks, make_wing_image


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def landmarks(rng):
    return make_landmarks(rng, angle_deg=7.0)


@pytest.fixture
def wing_image():
    return make_wing_image()


@pytest.fixture
def synthetic_raw(tmp_path):
    return build_synthetic_raw(tmp_path)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_obb_dataset.py`. It pins Review Focus 1 (`test_an_unreadable_image_fails_with_its_name`, `test_an_image_missing_from_every_split_fails`, `test_an_image_in_two_splits_fails`, `test_degenerate_landmarks_fail_with_the_file_name`, `test_non_numeric_landmarks_fail_with_the_file_name`):

```python
import cv2
import numpy as np
import pandas as pd
import pytest

from wings.detection.obb_dataset import build_labels, read_split_map, write_image_lists
from wings.detection.obb_labels import CORNER_COLUMNS, points_inside_box


def build(raw):
    return build_labels(raw.raw, raw.countries, raw.split_dir, raw.mean_shape)


def test_one_row_per_image_with_the_expected_columns_and_split(synthetic_raw):
    table = build(synthetic_raw)
    assert len(table) == 10
    assert table["split"].value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}
    expected = {"file", "country", "split", "img_w", "img_h", *CORNER_COLUMNS, "cx", "cy", "length", "width", "theta_deg", "eig_ratio", "dir_sign", "bg_b", "bg_g", "bg_r"}
    assert expected <= set(table.columns)


def test_every_landmark_lies_inside_its_box(synthetic_raw):
    table = build(synthetic_raw)
    for _, row in table.iterrows():
        corners = row[CORNER_COLUMNS].to_numpy(np.float64).reshape(4, 2)
        assert points_inside_box(corners, synthetic_raw.landmarks[row["file"].split("/")[-1]], tol=1e-6)


def test_size_and_background_columns_match_the_images(synthetic_raw):
    table = build(synthetic_raw)
    for _, row in table.iterrows():
        image = cv2.imread(str(synthetic_raw.raw / row["file"]))
        assert (row["img_h"], row["img_w"]) == image.shape[:2]
        assert (row["bg_b"], row["bg_g"], row["bg_r"]) == synthetic_raw.backgrounds[row["file"].split("/")[-1]]


def test_direction_sign_is_consistent_for_upright_wings(synthetic_raw):
    assert set(build(synthetic_raw)["dir_sign"]) == {1}


def test_image_lists_hold_absolute_paths_of_each_split(synthetic_raw, tmp_path):
    table = build(synthetic_raw)
    write_image_lists(table, synthetic_raw.raw, tmp_path)
    train = (tmp_path / "train.txt").read_text().split()
    assert len(train) == 6 and all(p.endswith(".png") for p in train)
    assert all(pd.io.common.file_exists(p) for p in train)


def test_an_unreadable_image_fails_with_its_name(synthetic_raw):
    (synthetic_raw.raw / "AA-wing-images" / "AA-0002.png").unlink()
    with pytest.raises(FileNotFoundError, match="AA-0002"):
        build(synthetic_raw)


def test_an_image_missing_from_every_split_fails(synthetic_raw):
    (synthetic_raw.split_dir / "images" / "train" / "AA-0001.png").unlink()
    with pytest.raises(ValueError, match="AA-0001"):
        build(synthetic_raw)


def test_an_image_in_two_splits_fails(synthetic_raw):
    (synthetic_raw.split_dir / "images" / "val" / "AA-0001.png").touch()
    with pytest.raises(ValueError, match="AA-0001"):
        read_split_map(synthetic_raw.split_dir)


def test_degenerate_landmarks_fail_with_the_file_name(synthetic_raw):
    path = synthetic_raw.raw / "BB-raw-coordinates.csv"
    table = pd.read_csv(path)
    table.iloc[1, 1:] = 50.0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="BB-0001"):
        build(synthetic_raw)


def test_non_numeric_landmarks_fail_with_the_file_name(synthetic_raw):
    path = synthetic_raw.raw / "BB-raw-coordinates.csv"
    table = pd.read_csv(path).astype(object)
    table.iloc[0, 3] = "n/a"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="BB-0000"):
        build(synthetic_raw)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/test_obb_dataset.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'wings.detection.obb_dataset'`.

- [ ] **Step 4: Implement the builder**

`wings/detection/obb_dataset.py`:

```python
"""Build the OBB label table (`labels.csv`) from the raw wing images.

    uv run python -m wings.detection.obb_dataset build      # labels.csv, train.txt, val.txt, test.txt
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from wings.config import COORDS_SUFX, COUNTRIES, IMG_FOLDER_SUFX, PROCESSED_DATA_DIR, RAW_DATA_DIR
from wings.detection.obb_labels import background_color, direction_sign, obb_from_landmarks, points_inside_box, principal_axis, reference_landmarks

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_OUT_DIR = PROCESSED_DATA_DIR / "detection-obb"
DEFAULT_SPLIT_DIR = PROCESSED_DATA_DIR / "detection"  # the folders of the old axis-aligned detector, source of the split
DEFAULT_MEAN_SHAPE = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle" / "mean_shape.pth"

app = typer.Typer(help="OBB wing detector: label table and frozen val/test sets.")


def read_split_map(split_dir: Path) -> dict[str, str]:
    """File name -> split, from the layout {split_dir}/images/{train,val,test}/<file>."""
    mapping: dict[str, str] = {}
    for split in SPLITS:
        for path in sorted((split_dir / "images" / split).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.name in mapping:
                raise ValueError(f"{path.name} is in both '{mapping[path.name]}' and '{split}'")
            mapping[path.name] = split
    return mapping


def build_labels(
    raw_dir: Path,
    countries: list[str],
    split_dir: Path,
    mean_shape: np.ndarray,
    coords_sufx: str = COORDS_SUFX,
    img_sufx: str = IMG_FOLDER_SUFX,
) -> pd.DataFrame:
    """One row per raw image: OBB corners and parameters, split, image size, background colour, direction sign.
    Fails loudly (with the file name) on an unreadable image, bad landmarks or an image missing from the split."""
    i_lo, i_hi = reference_landmarks(mean_shape)
    split_of = read_split_map(split_dir)
    rows = []
    for country in countries:
        coords = pd.read_csv(raw_dir / f"{country}{coords_sufx}")
        for _, record in tqdm(coords.iterrows(), total=len(coords), desc=country, unit="img"):
            name = str(record["file"])
            relative = f"{country}{img_sufx}/{name}"
            image = cv2.imread(str(raw_dir / relative), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cannot read {raw_dir / relative}")
            if name not in split_of:
                raise ValueError(f"{name} is not in any split folder under {split_dir}")
            height, width = image.shape[:2]
            values = pd.to_numeric(record.iloc[1:], errors="coerce").to_numpy(np.float64)
            if len(values) != 38 or not np.isfinite(values).all():
                raise ValueError(f"bad landmark values for {name}")
            points = np.column_stack([values[0::2], height - values[1::2] - 1])  # CSV: y from the bottom
            try:
                obb = obb_from_landmarks(points)
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
            corners = obb.corners()
            if not points_inside_box(corners, points, tol=0.01):
                raise ValueError(f"{name}: a landmark lies outside its own box")
            bgr = background_color(image)
            row = {"file": relative, "country": country, "split": split_of[name], "img_w": width, "img_h": height}
            row.update({f"{axis}{i + 1}": float(corners[i, j]) for i in range(4) for j, axis in enumerate("xy")})
            row.update(
                cx=obb.cx, cy=obb.cy, length=obb.length, width=obb.width, theta_deg=obb.theta_deg, eig_ratio=obb.eig_ratio,
                dir_sign=direction_sign(points, principal_axis(points)[0], i_lo, i_hi), bg_b=bgr[0], bg_g=bgr[1], bg_r=bgr[2],
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_image_lists(table: pd.DataFrame, raw_dir: Path, out_dir: Path) -> None:
    """{split}.txt with the absolute paths of the raw images of each split (input of `WingOBBDataset`)."""
    for split in SPLITS:
        paths = [str((raw_dir / f).resolve()) for f in table.loc[table["split"] == split, "file"]]
        (out_dir / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


@app.command()
def build(
    out: Path = typer.Option(DEFAULT_OUT_DIR, "--out", "-o", help="Output folder."),
    split_dir: Path = typer.Option(DEFAULT_SPLIT_DIR, "--split-dir", help="Old detection dataset whose train/val/test folders define the split."),
    mean_shape_path: Path = typer.Option(DEFAULT_MEAN_SHAPE, "--mean-shape", help="Mean wing shape (.pth) used for the direction sign."),
) -> None:
    """Write labels.csv and the {train,val,test}.txt image lists."""
    import torch

    mean_shape = np.asarray(torch.load(mean_shape_path, weights_only=False), dtype=np.float64)
    out.mkdir(parents=True, exist_ok=True)
    table = build_labels(RAW_DATA_DIR, COUNTRIES, split_dir, mean_shape)
    table.to_csv(out / "labels.csv", index=False)
    write_image_lists(table, RAW_DATA_DIR, out)
    logger.info(f"{len(table)} rows -> {out / 'labels.csv'}; splits: {table['split'].value_counts().to_dict()}")
    weak = int((table["eig_ratio"] < 1.5).sum())
    logger.info(f"rows with an ill-defined axis (eig_ratio < 1.5): {weak}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests -v`
Expected: `36 passed`.

- [ ] **Step 6: Build the real label table**

Run: `uv run python -m wings.detection.obb_dataset build`
This reads all 21,722 raw images once and takes a few minutes. It stops with an error naming the file if an image is unreadable, a landmark is bad or outside its own box, or an image is missing from the split folders. If it stops, do not work around it: report the file to the author.

Check the result:

```bash
uv run python -c "import pandas as pd; t = pd.read_csv('data/processed/detection-obb/labels.csv'); print(len(t), t['split'].value_counts().to_dict()); print(t[['theta_deg', 'eig_ratio', 'length', 'width']].describe().round(2).to_string()); print('dir_sign:', t['dir_sign'].value_counts().to_dict()); print('eig_ratio < 1.5:', int((t['eig_ratio'] < 1.5).sum()))"
```

Expected: `21722 {'train': 17401, 'val': 2197, 'test': 2124}`. `theta_deg` should be concentrated within a few tens of degrees of 0 (raw wings are almost horizontal); if a large share sits near ±90 the axis picks the wrong direction, so stop and report. Record the `dir_sign` counts and the number of rows with `eig_ratio < 1.5` in the task report (no row is dropped either way).

- [ ] **Step 7: Commit**

```bash
git add wings/detection/obb_dataset.py tests/obb_synthetic.py tests/conftest.py tests/test_obb_dataset.py
git commit -m "Build the OBB label table and image lists from the raw wing images" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

(The generated files live under the git-ignored `/data/`.)

---

### Task 3: Single-wing augmentation

**Files:**
- Create: `wings/detection/obb_augment.py`
- Test: `tests/test_obb_augment.py`

**Interfaces:**
- Consumes: `CORNER_COLUMNS` (Task 1); a label-table row (Task 2) as a pandas `Series`.
- Produces (`wings/detection/obb_augment.py`):
  - `AugConfig` frozen dataclass (all fields have defaults; `imgsz=640`, `flip_p=0.5`, `fraction_range=(0.25, 0.90)`, `edge_margin=4`, `photometric=True`, and the photometric ranges);
  - `Item(image, corners, background)`, `Placement(matrix, flip, angle_deg, scale, corners)`, `Sample(image, corners)` frozen dataclasses (`Sample.corners` has shape (n, 4, 2), float32);
  - `load_item(path, row) -> Item` (raises `FileNotFoundError` if unreadable, `ValueError` if the size differs from the row);
  - `apply_affine(matrix, points)`, `rotation_matrix(centre, angle_deg) -> (3, 3)`, `box_sides(corners) -> (length, width)`;
  - `plan_placement(corners, image_width, region, rng, cfg) -> Placement` with `region = (x0, y0, x1, y1)`;
  - `warp_image(image, matrix, size, background) -> np.ndarray`, `warp_mask(shape_hw, matrix, size) -> np.ndarray`;
  - `apply_photometric(image, rng, cfg) -> np.ndarray`;
  - `make_single_sample(item, rng, cfg=AugConfig()) -> Sample`.

- [ ] **Step 1: Write the failing tests**

`tests/test_obb_augment.py`. The key test is `test_the_label_follows_the_pixels`: it paints the box itself into the source image and checks that the minimum-area rectangle of the painted pixels on the canvas matches the label corners. Review Focus 3 is pinned by `test_extreme_image_sizes_still_give_an_inside_box`, `test_a_box_that_extends_beyond_the_source_image_is_fine` and `test_a_box_that_exactly_fills_the_region_is_still_placed`:

```python
import cv2
import numpy as np
import pytest
from obb_synthetic import make_landmarks, make_wing_image
from scipy import stats

from wings.detection.obb_augment import (
    AugConfig,
    Item,
    apply_affine,
    apply_photometric,
    make_single_sample,
    plan_placement,
    warp_image,
    warp_mask,
)
from wings.detection.obb_labels import Obb, obb_from_landmarks

PLAIN = AugConfig(photometric=False)
BACKGROUND = (200, 210, 190)


def painted_item(corners, fill, size=(326, 800), background=BACKGROUND) -> Item:
    """Source image whose only dark structure is the filled box itself, so label and pixels can be compared."""
    img = np.full((size[0], size[1], 3), background, np.uint8)
    cv2.fillPoly(img, [np.round(corners).astype(np.int32)], fill)
    return Item(img, np.asarray(corners, np.float64), background)


def painted_box_corners(canvas: np.ndarray, fill) -> np.ndarray:
    """Corners of the minimum-area rectangle of the pixels that have (about) the colour `fill`."""
    distance = np.abs(canvas.astype(np.int32) - np.array(fill)).sum(axis=2)
    ys, xs = np.nonzero(distance < 60)
    return cv2.boxPoints(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32)))


def corner_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from a corner of one box to the nearest corner of the other (both directions)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


BOX = Obb(cx=400.0, cy=163.0, length=500.0, width=180.0, theta_deg=3.0, eig_ratio=9.0).corners()


@pytest.mark.parametrize("seed", range(20))
def test_the_label_follows_the_pixels(seed):
    item = painted_item(BOX, fill=(30, 30, 30))
    sample = make_single_sample(item, np.random.default_rng(seed), PLAIN)
    assert sample.image.shape == (640, 640, 3) and sample.corners.shape == (1, 4, 2)
    assert corner_distance(sample.corners[0], painted_box_corners(sample.image, (30, 30, 30))) < 2.5


@pytest.mark.parametrize("seed", range(10))
def test_pixels_outside_the_source_image_have_exactly_the_background_colour(seed):
    item = Item(np.full((326, 800, 3), (200, 100, 50), np.uint8), BOX, (10, 20, 30))
    rng = np.random.default_rng(seed)
    placement = plan_placement(item.corners, 800, (0, 0, 640, 640), rng, PLAIN)
    canvas = warp_image(item.image, placement.matrix, 640, item.background)
    outside = cv2.erode((warp_mask(item.image.shape[:2], placement.matrix, 640) == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert outside.any()
    assert (canvas[outside] == (10, 20, 30)).all()


def random_boxes(rng, n):
    for _ in range(n):
        yield obb_from_landmarks(make_landmarks(rng, centre=(400, 160), half_length=rng.uniform(120, 380), half_width=rng.uniform(40, 150), angle_deg=rng.uniform(-30, 30))).corners()


def test_the_box_always_stays_inside_the_canvas_with_the_margin():
    rng = np.random.default_rng(0)
    cfg = AugConfig()
    for corners in random_boxes(rng, 100):
        for _ in range(5):
            placement = plan_placement(corners, 800, (0, 0, 640, 640), rng, cfg)
            assert placement.corners.min() >= cfg.edge_margin - 1e-6
            assert placement.corners.max() <= 640 - cfg.edge_margin + 1e-6


def test_a_box_that_exactly_fills_the_region_is_still_placed():
    """With fraction 1.0 the fit constraint binds at most angles and leaves no room to move; rounding must not break that."""
    cfg = AugConfig(fraction_range=(1.0, 1.0))
    rng = np.random.default_rng(4)
    for _ in range(2000):
        placement = plan_placement(BOX, 800, (0, 0, 640, 640), rng, cfg)
        assert placement.corners.min() >= cfg.edge_margin - 1e-6
        assert placement.corners.max() <= 640 - cfg.edge_margin + 1e-6


def test_the_rotation_angle_is_uniform():
    rng = np.random.default_rng(1)
    angles = np.array([plan_placement(BOX, 800, (0, 0, 640, 640), rng, AugConfig()).angle_deg for _ in range(4000)])
    assert angles.min() >= -180 and angles.max() <= 180
    assert stats.kstest((angles + 180) / 360, "uniform").pvalue > 0.01


def test_the_box_angle_modulo_180_is_uniform():
    rng = np.random.default_rng(2)
    thetas = []
    for _ in range(4000):
        corners = plan_placement(BOX, 800, (0, 0, 640, 640), rng, AugConfig()).corners
        edge = corners[1] - corners[0]
        thetas.append(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
    assert stats.kstest(np.array(thetas) / 180.0, "uniform").pvalue > 0.01


def test_flip_probability_and_scale_range():
    rng = np.random.default_rng(3)
    cfg = AugConfig()
    placements = [plan_placement(BOX, 800, (0, 0, 640, 640), rng, cfg) for _ in range(2000)]
    assert 0.45 < np.mean([p.flip for p in placements]) < 0.55
    lengths = np.array([np.linalg.norm(p.corners[1] - p.corners[0]) for p in placements])
    assert lengths.max() <= cfg.fraction_range[1] * 640 + 1e-6


def test_same_seed_same_sample_different_seed_different_sample():
    item = painted_item(BOX, fill=(30, 30, 30))
    a = make_single_sample(item, np.random.default_rng(5))
    b = make_single_sample(item, np.random.default_rng(5))
    c = make_single_sample(item, np.random.default_rng(6))
    assert (a.image == b.image).all() and (a.corners == b.corners).all()
    assert not np.allclose(a.corners, c.corners)


@pytest.mark.parametrize("size", [(40, 100), (2400, 6000)], ids=["tiny", "huge"])
def test_extreme_image_sizes_still_give_an_inside_box(size):
    h, w = size
    corners = Obb(cx=w / 2, cy=h / 2, length=0.9 * w, width=0.5 * h, theta_deg=0.0, eig_ratio=5.0).corners()
    item = Item(np.full((h, w, 3), BACKGROUND, np.uint8), corners, BACKGROUND)
    sample = make_single_sample(item, np.random.default_rng(0), PLAIN)
    assert sample.image.shape == (640, 640, 3)
    assert sample.corners.min() >= 0 and sample.corners.max() <= 640


def test_a_box_that_extends_beyond_the_source_image_is_fine():
    corners = Obb(cx=400.0, cy=163.0, length=860.0, width=360.0, theta_deg=0.0, eig_ratio=5.0).corners()  # larger than the 800x326 image
    item = Item(make_wing_image(), corners, BACKGROUND)
    sample = make_single_sample(item, np.random.default_rng(1), PLAIN)
    assert sample.corners.min() >= 0 and sample.corners.max() <= 640


def test_photometric_steps_keep_shape_and_dtype_and_change_pixels():
    img = make_wing_image(640, 640)
    out = apply_photometric(img, np.random.default_rng(0), AugConfig(blur_p=1.0, jpeg_p=1.0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 1.0


def test_apply_affine_accepts_two_by_three_and_three_by_three():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = np.array([[0.0, -1.0, 5.0], [1.0, 0.0, 6.0]])
    assert np.allclose(apply_affine(m, pts), apply_affine(np.vstack([m, [0, 0, 1]]), pts))
    assert np.allclose(apply_affine(m, pts)[0], [3.0, 7.0])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_obb_augment.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'wings.detection.obb_augment'`.

- [ ] **Step 3: Implement the augmentation core**

`wings/detection/obb_augment.py`:

```python
"""Online augmentation for the oriented-box (OBB) wing detector.

A training sample is built from raw wing images and their OBB corners with ONE affine warp per wing
(flip, rotation about the box centre, scale, translation), so the corners are transformed exactly and
the whole box always stays inside the canvas. Rotated areas outside the source image are filled with
that image's own background colour. Photometric steps run afterwards on the whole canvas.

`WingOBBDataset` plugs the pipeline into Ultralytics: it replaces the stock geometric augmentations and
feeds the stock `Format` transform, so batches are identical to a normal OBB training batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wings.detection.obb_labels import CORNER_COLUMNS


@dataclass(frozen=True)
class AugConfig:
    """Initial values; tuned in notebook 31 by looking at samples."""

    imgsz: int = 640
    flip_p: float = 0.5
    fraction_range: tuple[float, float] = (0.25, 0.90)  # box length / canvas side
    edge_margin: int = 4  # minimal distance between the box and the canvas (or cell) border, px
    photometric: bool = True
    brightness_range: tuple[float, float] = (0.5, 1.5)
    contrast_range: tuple[float, float] = (0.5, 1.5)
    hue_delta: float = 0.015  # fraction of the hue circle
    saturation_range: tuple[float, float] = (0.6, 1.4)
    triangles_range: tuple[int, int] = (40, 200)
    triangle_size_range: tuple[float, float] = (2.0, 9.0)  # px on the canvas
    blur_p: float = 0.15
    blur_sigma_range: tuple[float, float] = (0.3, 1.2)
    jpeg_p: float = 0.2
    jpeg_quality_range: tuple[int, int] = (60, 95)


@dataclass(frozen=True)
class Item:
    """One source wing: raw BGR image, its OBB corners (pixels of that image) and background colour (BGR)."""

    image: np.ndarray
    corners: np.ndarray
    background: tuple[int, int, int]


@dataclass(frozen=True)
class Placement:
    matrix: np.ndarray  # (2, 3) affine, source pixels -> canvas pixels
    flip: bool
    angle_deg: float
    scale: float
    corners: np.ndarray  # (4, 2) corners after the transform


@dataclass(frozen=True)
class Sample:
    image: np.ndarray  # (S, S, 3) uint8 BGR
    corners: np.ndarray  # (n, 4, 2) float32, pixels of `image`


def load_item(path: str | Path, row) -> Item:
    """Read the raw image at `path` and combine it with the label-table row `row` (a pandas Series).
    Fails loudly if the file is unreadable or its size differs from the one in the table."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    if image.shape[:2] != (int(row["img_h"]), int(row["img_w"])):
        raise ValueError(f"{path} has shape {image.shape[:2]}, the labels table says {(int(row['img_h']), int(row['img_w']))}")
    corners = row[CORNER_COLUMNS].to_numpy(np.float64).reshape(4, 2)
    return Item(image, corners, (int(row["bg_b"]), int(row["bg_g"]), int(row["bg_r"])))


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a (2, 3) or (3, 3) affine matrix to (n, 2) points."""
    return np.asarray(points, dtype=np.float64) @ matrix[:2, :2].T + matrix[:2, 2]


def rotation_matrix(centre: np.ndarray, angle_deg: float) -> np.ndarray:
    """(3, 3) rotation about `centre`, counter-clockwise on the screen (y down) for a positive angle, like
    `cv2.getRotationMatrix2D` but in float64 (OpenCV rounds the centre to float32, which shifts labels by ~1e-5 px)."""
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, s, (1 - c) * centre[0] - s * centre[1]], [-s, c, s * centre[0] + (1 - c) * centre[1]], [0.0, 0.0, 1.0]])


def box_sides(corners: np.ndarray) -> tuple[float, float]:
    """(length, width) of a box given as in `Obb.corners()`."""
    return float(np.linalg.norm(corners[1] - corners[0])), float(np.linalg.norm(corners[3] - corners[0]))


def plan_placement(corners: np.ndarray, image_width: int, region: tuple[int, int, int, int], rng: np.random.Generator, cfg: AugConfig) -> Placement:
    """Random flip, rotation (uniform on -180..180 degrees), scale and position such that the whole
    rotated box lies inside `region` = (x0, y0, x1, y1) with `cfg.edge_margin` to spare."""
    corners = np.asarray(corners, dtype=np.float64)
    length, width = box_sides(corners)
    x0, y0, x1, y1 = region
    margin = cfg.edge_margin
    avail_w, avail_h = (x1 - x0) - 2 * margin, (y1 - y0) - 2 * margin
    if avail_w <= 0 or avail_h <= 0 or length <= 0 or width <= 0:
        raise ValueError(f"cannot place a {length:.1f}x{width:.1f} box in region {region} with margin {margin}")

    flip = bool(rng.random() < cfg.flip_p)
    flip_m = np.array([[-1.0, 0.0, image_width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) if flip else np.eye(3)
    centre = apply_affine(flip_m, corners).mean(axis=0)
    angle = float(rng.uniform(-180.0, 180.0))
    fraction = float(rng.uniform(*cfg.fraction_range))

    rot_m = rotation_matrix(centre, angle)
    rotated = apply_affine(rot_m @ flip_m, corners)  # unit scale, still centred on `centre`
    extent_x, extent_y = float(np.ptp(rotated[:, 0])), float(np.ptp(rotated[:, 1]))
    scale = min(fraction * min(x1 - x0, y1 - y0) / length, avail_w / extent_x, avail_h / extent_y)

    half_x, half_y = scale * extent_x / 2.0, scale * extent_y / 2.0
    low_x, low_y = x0 + margin + half_x, y0 + margin + half_y
    # When the box fills the region the range is a single point; rounding must not turn it into a negative one.
    target_x = float(rng.uniform(low_x, max(low_x, x1 - margin - half_x)))
    target_y = float(rng.uniform(low_y, max(low_y, y1 - margin - half_y)))

    scale_m = np.array([[scale, 0.0, centre[0] * (1 - scale)], [0.0, scale, centre[1] * (1 - scale)], [0.0, 0.0, 1.0]])
    move_m = np.array([[1.0, 0.0, target_x - centre[0]], [0.0, 1.0, target_y - centre[1]], [0.0, 0.0, 1.0]])
    total = move_m @ scale_m @ rot_m @ flip_m
    return Placement(total[:2], flip, angle, float(scale), apply_affine(total, corners))


def warp_image(image: np.ndarray, matrix: np.ndarray, size: int, background: tuple[int, int, int]) -> np.ndarray:
    """Warp `image` into a `size` x `size` canvas. Border pixels get `background`. If the transform shrinks the
    image it is first reduced with INTER_AREA (pixel-centre aligned), which keeps thin structures intact."""
    m3 = np.vstack([matrix, [0.0, 0.0, 1.0]]) if matrix.shape == (2, 3) else np.asarray(matrix, dtype=np.float64)
    scale = math.hypot(m3[0, 0], m3[0, 1])
    source = image
    if scale < 1.0:
        h, w = image.shape[:2]
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        source = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / w, new_h / h
        to_small = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])  # source index -> small index
        m3 = m3 @ np.linalg.inv(to_small)
    return cv2.warpAffine(source, m3[:2], (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(c) for c in background))


def warp_mask(shape_hw: tuple[int, int], matrix: np.ndarray, size: int) -> np.ndarray:
    """255 where the warped source image covers the canvas, 0 elsewhere."""
    return cv2.warpAffine(np.full(shape_hw, 255, np.uint8), matrix[:2], (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _add_triangles(image: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """Small dark triangles at random places and orientations (debris), as in `augment_dataset.add_triangle_noise`."""
    out = image.copy()
    h, w = out.shape[:2]
    unit = np.array([[0.0, -0.9], [-0.55, 0.6], [0.55, 0.6]])
    for _ in range(int(rng.integers(cfg.triangles_range[0], cfg.triangles_range[1] + 1))):
        size = rng.uniform(*cfg.triangle_size_range)
        t = rng.uniform(0.0, 2.0 * np.pi)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pts = (unit * size) @ rot.T + [rng.uniform(0, w), rng.uniform(0, h)]
        cv2.fillPoly(out, [np.round(pts).astype(np.int32)], (0, 0, 0))
    return out


def apply_photometric(image: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """Brightness and contrast, hue and saturation, debris, blur, JPEG re-compression (uint8 BGR in and out)."""
    img = image.astype(np.float32) * rng.uniform(*cfg.brightness_range)
    mean = img.mean()
    img = np.clip(mean + rng.uniform(*cfg.contrast_range) * (img - mean), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue_delta, cfg.hue_delta) * 180.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*cfg.saturation_range), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = _add_triangles(img, rng, cfg)
    if rng.random() < cfg.blur_p:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(*cfg.blur_sigma_range)))
    if rng.random() < cfg.jpeg_p:
        quality = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1] + 1))
        ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return img


def make_single_sample(item: Item, rng: np.random.Generator, cfg: AugConfig = AugConfig()) -> Sample:
    """One wing on a `cfg.imgsz` square canvas."""
    placement = plan_placement(item.corners, item.image.shape[1], (0, 0, cfg.imgsz, cfg.imgsz), rng, cfg)
    canvas = warp_image(item.image, placement.matrix, cfg.imgsz, item.background)
    if cfg.photometric:
        canvas = apply_photometric(canvas, rng, cfg)
    return Sample(canvas, placement.corners[None].astype(np.float32))
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests -v`
Expected: `77 passed`.

- [ ] **Step 5: Commit**

```bash
git add wings/detection/obb_augment.py tests/test_obb_augment.py
git commit -m "Add the single-wing online augmentation for the OBB detector" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Ultralytics training dataset

**Files:**
- Modify: `wings/detection/obb_augment.py` (replace the import block, append the class)
- Modify: `tests/conftest.py` (replace the file)
- Test: `tests/test_obb_dataset_class.py`

**Interfaces:**
- Consumes: Task 3 (`AugConfig`, `Item`, `load_item`, `make_single_sample`), Task 2 (`labels.csv` and `{split}.txt` produced by `build_labels` and `write_image_lists`).
- Produces: `WingOBBDataset(labels_csv, split, image_list, raw_dir=None, cfg=AugConfig(), hyp=DEFAULT_CFG, batch_size=16)`, a `YOLODataset` subclass, with `load_item(index) -> Item`, `make_label(index, rng) -> dict` (the label dict Ultralytics hands to `Format`, before formatting), `label_for(index, seed) -> dict` (deterministic) and the stock static `collate_fn`. Test fixture `label_files` (`SimpleNamespace(out, table, raw)`: `labels.csv` and image lists built from the synthetic raw data).

- [ ] **Step 1: Replace the shared fixtures**

`tests/conftest.py` (replace the whole file; it gains the `label_files` fixture):

```python
from types import SimpleNamespace

import numpy as np
import pytest
from obb_synthetic import build_synthetic_raw, make_landmarks, make_wing_image


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def landmarks(rng):
    return make_landmarks(rng, angle_deg=7.0)


@pytest.fixture
def wing_image():
    return make_wing_image()


@pytest.fixture
def synthetic_raw(tmp_path):
    return build_synthetic_raw(tmp_path)


@pytest.fixture
def label_files(synthetic_raw, tmp_path):
    """labels.csv and the {split}.txt image lists built from the synthetic raw data."""
    from wings.detection.obb_dataset import build_labels, write_image_lists

    out = tmp_path / "detection-obb"
    out.mkdir()
    table = build_labels(synthetic_raw.raw, synthetic_raw.countries, synthetic_raw.split_dir, synthetic_raw.mean_shape)
    table.to_csv(out / "labels.csv", index=False)
    write_image_lists(table, synthetic_raw.raw, out)
    return SimpleNamespace(out=out, table=table, raw=synthetic_raw.raw)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_obb_dataset_class.py`. The core check is `test_boxes_given_to_the_loss_match_our_corners`: the boxes that `Format` hands to the loss, turned back into corners, equal our own corners. Review Focus 5 (size mismatch) is pinned by `test_a_size_mismatch_between_table_and_image_fails_loudly`:

```python
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from ultralytics.utils import DEFAULT_CFG, ops

from wings.detection.obb_augment import AugConfig, WingOBBDataset


def make_dataset(files, **cfg_kwargs):
    return WingOBBDataset(files.out / "labels.csv", "train", files.out / "train.txt", raw_dir=files.raw, cfg=AugConfig(**cfg_kwargs))


def vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def assert_loss_boxes_match(dataset) -> int:
    """For every image: the boxes `Format` hands to the loss, turned back into corners, equal our own corners.
    Returns the largest number of boxes seen in one sample."""
    most = 0
    for index in range(len(dataset)):
        label = dataset.label_for(index, seed=index)
        ours = label["instances"].segments.copy()
        out = dataset.transforms(label)
        rboxes = out["bboxes"].clone()
        rboxes[:, [0, 2]] *= 640
        rboxes[:, [1, 3]] *= 640
        theirs = ops.xywhr2xyxyxyxy(rboxes).numpy()
        assert len(theirs) == len(ours)
        for polygon in ours:
            assert min(vertex_distance(polygon, other) for other in theirs) < 0.5
        most = max(most, len(ours))
    return most


def test_dataset_covers_the_requested_split(label_files):
    assert len(make_dataset(label_files)) == 6


def test_item_has_the_keys_and_shapes_ultralytics_expects(label_files):
    item = make_dataset(label_files)[0]
    assert item["img"].shape == (3, 640, 640) and item["img"].dtype == torch.uint8
    assert item["bboxes"].shape == (1, 5) and item["cls"].shape == (1, 1) and item["batch_idx"].shape == (1,)
    assert 0.0 < float(item["bboxes"][:, :4].max()) <= 1.0
    assert {"im_file", "ori_shape", "resized_shape", "ratio_pad"} <= set(item)


def test_collated_batch_is_a_stock_obb_batch(label_files):
    dataset = make_dataset(label_files)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=WingOBBDataset.collate_fn)))
    assert batch["img"].shape == (4, 3, 640, 640)
    assert batch["bboxes"].shape == (4, 5) and batch["cls"].shape == (4, 1)
    assert batch["batch_idx"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_boxes_given_to_the_loss_match_our_corners(label_files):
    assert assert_loss_boxes_match(make_dataset(label_files)) == 1


def test_label_for_is_deterministic(label_files):
    dataset = make_dataset(label_files)
    a, b, c = dataset.label_for(0, 5), dataset.label_for(0, 5), dataset.label_for(0, 6)
    assert (a["img"] == b["img"]).all() and np.array_equal(a["instances"].segments, b["instances"].segments)
    assert not np.array_equal(a["instances"].segments, c["instances"].segments)


def test_items_follow_the_torch_seed_and_differ_between_calls(label_files):
    dataset = make_dataset(label_files)
    torch.manual_seed(0)
    first = dataset[0]["bboxes"]
    second = dataset[0]["bboxes"]
    torch.manual_seed(0)
    again = dataset[0]["bboxes"]
    assert torch.equal(first, again) and not torch.equal(first, second)


def test_a_size_mismatch_between_table_and_image_fails_loudly(label_files):
    table = pd.read_csv(label_files.out / "labels.csv")
    table.loc[table["split"] == "train", "img_w"] += 10
    table.to_csv(label_files.out / "labels.csv", index=False)
    with pytest.raises(ValueError, match="labels table says"):
        make_dataset(label_files).label_for(0, 1)


def test_an_image_missing_from_the_table_is_rejected(label_files):
    table = pd.read_csv(label_files.out / "labels.csv")
    table[table["file"] != table.loc[table["split"] == "train", "file"].iloc[0]].to_csv(label_files.out / "labels.csv", index=False)
    with pytest.raises(ValueError, match="missing from the labels table"):
        make_dataset(label_files)


def test_close_mosaic_keeps_working(label_files):
    dataset = make_dataset(label_files)
    dataset.close_mosaic(copy.copy(DEFAULT_CFG))
    assert dataset[0]["img"].shape == (3, 640, 640)
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/test_obb_dataset_class.py -v`
Expected: collection error `ImportError: cannot import name 'WingOBBDataset' from 'wings.detection.obb_augment'`.

- [ ] **Step 4: Implement the dataset class**

**Edit 1 of 2.** In `wings/detection/obb_augment.py`, find this text (it occurs exactly once):

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wings.detection.obb_labels import CORNER_COLUMNS
```

and replace it with:

```python
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics.data.augment import Compose, Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.instance import Instances

from wings.detection.obb_labels import CORNER_COLUMNS
```

**Edit 2 of 2.** Append to the end of `wings/detection/obb_augment.py`, after two blank lines:

```python
class WingOBBDataset(YOLODataset):
    """Training dataset: raw wing images + OBB corners from `labels.csv`, augmented online.

    `image_list` is a text file with the absolute paths of the images of `split` (written by
    `wings.detection.obb_dataset`). Stock Ultralytics label files are not used."""

    def __init__(
        self,
        labels_csv: str | Path,
        split: str,
        image_list: str | Path,
        raw_dir: str | Path | None = None,
        cfg: AugConfig = AugConfig(),
        hyp=DEFAULT_CFG,
        batch_size: int = 16,
    ) -> None:
        if raw_dir is None:
            from wings.config import RAW_DATA_DIR

            raw_dir = RAW_DATA_DIR
        self.raw_dir = Path(raw_dir)
        self.cfg = cfg
        table = pd.read_csv(labels_csv)
        self.table = table[table["split"] == split].reset_index(drop=True)
        self._row_of = {os.path.normcase(os.path.normpath(str(self.raw_dir / f))): i for i, f in enumerate(self.table["file"])}
        super().__init__(
            img_path=str(image_list),
            imgsz=cfg.imgsz,
            augment=True,
            hyp=hyp,
            batch_size=batch_size,
            rect=False,
            cache=None,
            task="obb",
            data={"names": {0: "wing"}, "channels": 3},
            prefix=f"{split}: ",
        )

    def get_labels(self) -> list[dict]:
        """Placeholder labels (the real ones are built per sample); keeps `im_files` and `labels` aligned."""
        labels, self._rows = [], []
        for im_file in self.im_files:
            key = os.path.normcase(os.path.normpath(im_file))
            if key not in self._row_of:
                raise ValueError(f"{im_file} is listed in the image list but missing from the labels table")
            row = self.table.iloc[self._row_of[key]]
            self._rows.append(self._row_of[key])
            corners = row[CORNER_COLUMNS].to_numpy(np.float64).reshape(4, 2) / [row["img_w"], row["img_h"]]
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (int(row["img_h"]), int(row["img_w"])),
                    "cls": np.zeros((1, 1), np.float32),
                    "bboxes": np.zeros((1, 4), np.float32),
                    "segments": [corners.astype(np.float32)],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    def build_transforms(self, hyp=None) -> Compose:
        """Geometry and photometry happen in `get_image_and_label`; only the stock `Format` is left."""
        hyp = hyp if hyp is not None else DEFAULT_CFG
        return Compose(
            [
                Format(
                    bbox_format="xywh",
                    normalize=True,
                    return_mask=False,
                    return_keypoint=False,
                    return_obb=True,
                    mask_ratio=hyp.mask_ratio,
                    mask_overlap=hyp.overlap_mask,
                    batch_idx=True,
                    bgr=hyp.bgr if self.augment else 0.0,
                )
            ]
        )

    def load_item(self, index: int) -> Item:
        return load_item(self.im_files[index], self.table.iloc[self._rows[index]])

    def make_label(self, index: int, rng: np.random.Generator) -> dict:
        """The label dict Ultralytics hands to `Format` (before formatting), for sample `index` drawn with `rng`."""
        sample = make_single_sample(self.load_item(index), rng, self.cfg)
        polygons = sample.corners
        n = len(polygons)
        boxes = np.concatenate([polygons.min(axis=1), polygons.max(axis=1)], axis=1)
        h0, w0 = self.labels[index]["shape"]
        return {
            "im_file": self.im_files[index],
            "img": sample.image,
            "cls": np.zeros((n, 1), np.float32),
            "instances": Instances(boxes.astype(np.float32), polygons.astype(np.float32), None, bbox_format="xyxy", normalized=False),
            "ori_shape": (h0, w0),
            "resized_shape": sample.image.shape[:2],
            "ratio_pad": (1.0, 1.0),
        }

    def label_for(self, index: int, seed: int) -> dict:
        """Deterministic sample (for the notebook and tests)."""
        return self.make_label(index, np.random.default_rng(seed))

    def get_image_and_label(self, index: int) -> dict:
        # The torch RNG is reseeded by PyTorch in every DataLoader worker, so workers never repeat each other.
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        return self.make_label(index, np.random.default_rng(seed))
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests -v`
Expected: `86 passed`.

- [ ] **Step 6: Commit**

```bash
git add wings/detection/obb_augment.py tests/conftest.py tests/test_obb_dataset_class.py
git commit -m "Add WingOBBDataset, the Ultralytics training dataset of the OBB detector" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Multi-wing compositions

**Files:**
- Modify: `wings/detection/obb_augment.py`
- Modify: `tests/test_obb_augment.py`, `tests/test_obb_dataset_class.py`

**Interfaces:**
- Consumes: Tasks 3 and 4 (`plan_placement`, `warp_image`, `warp_mask`, `apply_photometric`, `Sample`, `WingOBBDataset.make_label`).
- Produces: `AugConfig.p_multi: float = 0.4` and `AugConfig.multi_k_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)`; `choose_k(rng, cfg=AugConfig()) -> int` (2, 3 or 4); `grid_cells(k, size, rng) -> list[tuple[int, int, int, int]]` (disjoint cells `(x0, y0, x1, y1)`); `make_multi_sample(items, rng, cfg=AugConfig()) -> Sample`. `WingOBBDataset.make_label` now returns a composition with probability `cfg.p_multi`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_obb_augment.py` (the last appended test checks 5,000 compositions on a small canvas, as the spec's acceptance criterion asks):

**Edit 1 of 2.** In `tests/test_obb_augment.py`, find this text (it occurs exactly once):

```python
    apply_photometric,
    make_single_sample,
```

and replace it with:

```python
    apply_photometric,
    choose_k,
    grid_cells,
    make_multi_sample,
    make_single_sample,
```

**Edit 2 of 2.** Append to the end of `tests/test_obb_augment.py`, after two blank lines:

```python
def test_choose_k_follows_the_weights():
    rng = np.random.default_rng(0)
    ks = np.array([choose_k(rng) for _ in range(4000)])
    assert set(ks) == {2, 3, 4}
    assert abs((ks == 2).mean() - 0.5) < 0.05 and abs((ks == 3).mean() - 0.25) < 0.05


@pytest.mark.parametrize("k", [2, 3, 4])
def test_grid_cells_are_disjoint_and_inside_the_canvas(k):
    for seed in range(10):
        cells = grid_cells(k, 640, np.random.default_rng(seed))
        assert len(cells) == k
        for i, (x0, y0, x1, y1) in enumerate(cells):
            assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 640
            for x2, y2, x3, y3 in cells[i + 1 :]:
                assert x1 <= x2 or x3 <= x0 or y1 <= y2 or y3 <= y0


@pytest.mark.parametrize("k", [2, 3, 4])
def test_multi_wing_boxes_do_not_overlap_and_stay_inside(k):
    rng = np.random.default_rng(k)
    for _ in range(40):
        items = [painted_item(BOX, fill=(30, 30, 30)) for _ in range(k)]
        sample = make_multi_sample(items, rng, PLAIN)
        assert sample.corners.shape == (k, 4, 2)
        assert sample.corners.min() >= 0 and sample.corners.max() <= 640
        boxes = np.concatenate([sample.corners.min(axis=1), sample.corners.max(axis=1)], axis=1)  # x0, y0, x1, y1
        for i in range(k):
            for j in range(i + 1, k):
                a, b = boxes[i], boxes[j]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_every_wing_of_a_composition_keeps_its_own_pixels():
    fills = [(30, 30, 30), (30, 30, 230), (30, 230, 30)]
    items = [painted_item(BOX, fill=f, background=BACKGROUND) for f in fills]
    sample = make_multi_sample(items, np.random.default_rng(11), PLAIN)
    for fill in fills:
        painted = painted_box_corners(sample.image, fill)
        assert min(corner_distance(label, painted) for label in sample.corners) < 2.5


def test_five_thousand_compositions_never_overlap_or_leave_the_canvas():
    """The geometry of the full-size case on a 160 px canvas, so that 5,000 samples take seconds."""
    cfg = AugConfig(imgsz=160, photometric=False)
    rng = np.random.default_rng(7)
    box = Obb(cx=50.0, cy=20.0, length=70.0, width=26.0, theta_deg=3.0, eig_ratio=9.0).corners()
    item = Item(np.full((40, 100, 3), BACKGROUND, np.uint8), box, BACKGROUND)
    for _ in range(5000):
        k = choose_k(rng, cfg)
        sample = make_multi_sample([item] * k, rng, cfg)
        assert sample.corners.shape == (k, 4, 2)
        assert sample.corners.min() >= 0 and sample.corners.max() <= cfg.imgsz
        boxes = np.concatenate([sample.corners.min(axis=1), sample.corners.max(axis=1)], axis=1)  # x0, y0, x1, y1
        for i in range(k):
            for j in range(i + 1, k):
                a, b = boxes[i], boxes[j]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
```

In `tests/test_obb_dataset_class.py` (the helper now defaults to single wings so the earlier tests stay valid; the appended tests cover compositions and pin Review Focus 5, `test_a_one_image_split_can_still_make_compositions`):

**Edit 1 of 2.** In `tests/test_obb_dataset_class.py`, find this text (it occurs exactly once):

```python
def make_dataset(files, **cfg_kwargs):
    return WingOBBDataset(files.out / "labels.csv", "train", files.out / "train.txt", raw_dir=files.raw, cfg=AugConfig(**cfg_kwargs))
```

and replace it with:

```python
def make_dataset(files, **cfg_kwargs):
    cfg_kwargs.setdefault("p_multi", 0.0)
    return WingOBBDataset(files.out / "labels.csv", "train", files.out / "train.txt", raw_dir=files.raw, cfg=AugConfig(**cfg_kwargs))
```

**Edit 2 of 2.** Append to the end of `tests/test_obb_dataset_class.py`, after two blank lines:

```python
def test_boxes_given_to_the_loss_match_our_corners_for_compositions(label_files):
    assert 2 <= assert_loss_boxes_match(make_dataset(label_files, p_multi=1.0)) <= 4


def test_the_default_mix_contains_single_wings_and_compositions(label_files):
    dataset = make_dataset(label_files, p_multi=0.4)
    counts = [len(dataset.label_for(i % len(dataset), seed=i)["instances"].segments) for i in range(200)]
    assert min(counts) == 1 and max(counts) >= 2
    assert 0.2 < np.mean(np.array(counts) > 1) < 0.6


def test_a_one_image_split_can_still_make_compositions(label_files, tmp_path):
    table = pd.read_csv(label_files.out / "labels.csv")
    only = table[table["split"] == "train"].iloc[:1]
    small = tmp_path / "small"
    small.mkdir()
    only.to_csv(small / "labels.csv", index=False)
    (small / "train.txt").write_text(str((label_files.raw / only.iloc[0]["file"]).resolve()) + "\n")
    dataset = WingOBBDataset(small / "labels.csv", "train", small / "train.txt", raw_dir=label_files.raw, cfg=AugConfig(p_multi=1.0))
    assert len(dataset) == 1 and len(dataset.label_for(0, 3)["instances"].segments) >= 2
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_obb_augment.py tests/test_obb_dataset_class.py -v`
Expected: collection error `ImportError: cannot import name 'choose_k' from 'wings.detection.obb_augment'`.

- [ ] **Step 3: Implement compositions**

In `wings/detection/obb_augment.py`:

**Edit 1 of 4.** In `wings/detection/obb_augment.py`, find this text (it occurs exactly once):

```python
import os
from dataclasses import dataclass
```

and replace it with:

```python
import os
from collections.abc import Sequence
from dataclasses import dataclass
```

**Edit 2 of 4.** In `wings/detection/obb_augment.py`, find this text (it occurs exactly once):

```python
    jpeg_quality_range: tuple[int, int] = (60, 95)
```

and replace it with:

```python
    jpeg_quality_range: tuple[int, int] = (60, 95)
    p_multi: float = 0.4  # probability of a multi-wing composition
    multi_k_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)  # k = 2, 3, 4
```

**Edit 3 of 4.** In `wings/detection/obb_augment.py`, find this text (it occurs exactly once):

```python
    return Sample(canvas, placement.corners[None].astype(np.float32))


class WingOBBDataset(YOLODataset):
```

and replace it with:

```python
    return Sample(canvas, placement.corners[None].astype(np.float32))


def choose_k(rng: np.random.Generator, cfg: AugConfig = AugConfig()) -> int:
    """Number of wings of a composition: 2, 3 or 4 with the weights in `cfg.multi_k_weights`."""
    weights = np.asarray(cfg.multi_k_weights, dtype=np.float64)
    return int(rng.choice([2, 3, 4], p=weights / weights.sum()))


def grid_cells(k: int, size: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    """Non-overlapping cells (x0, y0, x1, y1): two halves for k=2, three of four quadrants for k=3, all four for k=4."""
    half = size // 2
    quadrants = [(0, 0, half, half), (half, 0, size, half), (0, half, half, size), (half, half, size, size)]
    if k == 2:
        if rng.random() < 0.5:
            cells = [(0, 0, half, size), (half, 0, size, size)]
        else:
            cells = [(0, 0, size, half), (0, half, size, size)]
    elif k == 3:
        drop = int(rng.integers(0, 4))
        cells = [q for i, q in enumerate(quadrants) if i != drop]
    elif k == 4:
        cells = quadrants
    else:
        raise ValueError(f"k must be 2, 3 or 4, got {k}")
    return [cells[i] for i in rng.permutation(len(cells))]


def make_multi_sample(items: Sequence[Item], rng: np.random.Generator, cfg: AugConfig = AugConfig()) -> Sample:
    """2-4 wings, each with its own flip, rotation and scale, in separate cells of the canvas (boxes cannot overlap).
    The canvas colour is the background colour of a random wing; each wing is pasted only where its warped
    source image covers its own cell."""
    k, size = len(items), cfg.imgsz
    cells = grid_cells(k, size, rng)
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = items[int(rng.integers(0, k))].background
    all_corners = []
    for item, (x0, y0, x1, y1) in zip(items, cells):
        placement = plan_placement(item.corners, item.image.shape[1], (x0, y0, x1, y1), rng, cfg)
        warped = warp_image(item.image, placement.matrix, size, item.background)
        mask = warp_mask(item.image.shape[:2], placement.matrix, size)
        paste = np.zeros((size, size), bool)
        paste[y0:y1, x0:x1] = mask[y0:y1, x0:x1] > 0
        canvas[paste] = warped[paste]
        all_corners.append(placement.corners)
    if cfg.photometric:
        canvas = apply_photometric(canvas, rng, cfg)
    return Sample(canvas, np.stack(all_corners).astype(np.float32))


class WingOBBDataset(YOLODataset):
```

**Edit 4 of 4.** In `wings/detection/obb_augment.py`, find this text (it occurs exactly once):

```python
        sample = make_single_sample(self.load_item(index), rng, self.cfg)
```

and replace it with:

```python
        if rng.random() < self.cfg.p_multi:
            k = choose_k(rng, self.cfg)
            items = [self.load_item(index)] + [self.load_item(int(j)) for j in rng.integers(0, len(self), k - 1)]
            sample = make_multi_sample(items, rng, self.cfg)
        else:
            sample = make_single_sample(self.load_item(index), rng, self.cfg)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests -v`
Expected: `98 passed`.

- [ ] **Step 5: Commit**

```bash
git add wings/detection/obb_augment.py tests/test_obb_augment.py tests/test_obb_dataset_class.py
git commit -m "Add multi-wing compositions to the OBB augmentation" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Frozen validation and test sets

**Files:**
- Modify: `wings/detection/obb_dataset.py`
- Modify: `tests/test_obb_dataset.py`

**Interfaces:**
- Consumes: Task 2 (`labels.csv`, `SPLITS`), Tasks 3 and 5 (`AugConfig`, `Sample`, `choose_k`, `load_item`, `make_single_sample`, `make_multi_sample`).
- Produces: `FROZEN_JPEG_QUALITY = 95`; `write_sample(images_dir, labels_dir, stem, sample)`; `freeze_split(table, split, raw_dir, out_dir, seed, n_multi, cfg=AugConfig()) -> int` (number of samples written; the sample of row `index` uses `np.random.default_rng([seed, SPLITS.index(split), index, 0])`, composition `j` uses `[seed, SPLITS.index(split), j, 1]`); `write_dataset_yaml(out_dir) -> Path`; `typer` command `freeze` (`--seed 7`, `--n-multi 500`). Output folders `data/processed/detection-obb/{val,test}/{images,labels}` and `dataset.yaml`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_obb_dataset.py`. Besides completeness and byte-for-byte reproducibility, `test_the_stock_ultralytics_dataset_reads_the_frozen_labels_back` checks that the stock dataset (which stage 2 validates with) reproduces our corners:

**Edit 1 of 2.** In `tests/test_obb_dataset.py`, find this text (it occurs exactly once):

```python
import cv2
import numpy as np
import pandas as pd
import pytest

from wings.detection.obb_dataset import build_labels, read_split_map, write_image_lists
from wings.detection.obb_labels import CORNER_COLUMNS, points_inside_box
```

and replace it with:

```python
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG, ops

from wings.detection.obb_dataset import build_labels, freeze_split, read_split_map, write_dataset_yaml, write_image_lists
from wings.detection.obb_labels import CORNER_COLUMNS, points_inside_box
```

**Edit 2 of 2.** Append to the end of `tests/test_obb_dataset.py`, after two blank lines:

```python
def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_frozen_split_is_complete_valid_and_reproducible(label_files, tmp_path):
    table = label_files.table
    first, second, other = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    assert freeze_split(table, "val", label_files.raw, first, seed=7, n_multi=3) == 2 + 3
    freeze_split(table, "val", label_files.raw, second, seed=7, n_multi=3)
    freeze_split(table, "val", label_files.raw, other, seed=8, n_multi=3)
    assert digest(first) == digest(second)
    assert digest(first) != digest(other)

    images = sorted((first / "val" / "images").glob("*.jpg"))
    labels = sorted((first / "val" / "labels").glob("*.txt"))
    assert [p.stem for p in images] == [p.stem for p in labels] and len(images) == 5
    for image_path, label_path in zip(images, labels):
        assert cv2.imread(str(image_path)).shape == (640, 640, 3)
        for line in label_path.read_text().strip().splitlines():
            values = line.split()
            assert values[0] == "0" and len(values) == 9
            assert all(0.0 <= float(v) <= 1.0 for v in values[1:])
    assert sum(len(p.read_text().strip().splitlines()) for p in labels if p.stem.startswith("multi_")) >= 3 * 2


def test_the_stock_ultralytics_dataset_reads_the_frozen_labels_back(label_files, tmp_path):
    """Stage 2 validates on these folders with the stock dataset, so it must reproduce our corners."""
    freeze_split(label_files.table, "val", label_files.raw, tmp_path, seed=7, n_multi=3)
    dataset = YOLODataset(
        img_path=str(tmp_path / "val" / "images"), imgsz=640, augment=False, hyp=DEFAULT_CFG, rect=False, cache=None,
        data={"names": {0: "wing"}, "channels": 3}, task="obb", batch_size=4, stride=32, pad=0.5, prefix="val: ",
    )
    assert len(dataset) == 5
    for index, image_file in enumerate(dataset.im_files):
        rboxes = dataset[index]["bboxes"].clone()
        rboxes[:, [0, 2]] *= 640
        rboxes[:, [1, 3]] *= 640
        theirs = ops.xywhr2xyxyxyxy(rboxes).numpy().reshape(-1, 2)
        label_file = tmp_path / "val" / "labels" / (Path(image_file).stem + ".txt")
        for line in label_file.read_text().strip().splitlines():
            polygon = np.array(line.split()[1:], dtype=float).reshape(4, 2) * 640
            assert np.linalg.norm(polygon[:, None, :] - theirs[None], axis=-1).min(axis=1).max() < 0.5


def test_dataset_yaml(tmp_path):
    content = yaml.safe_load(write_dataset_yaml(tmp_path).read_text())
    assert content["train"] == "train.txt" and content["val"] == "val/images" and content["test"] == "test/images"
    assert content["nc"] == 1 and content["names"] == {0: "wing"}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_obb_dataset.py -v`
Expected: collection error `ImportError: cannot import name 'freeze_split' from 'wings.detection.obb_dataset'`.

- [ ] **Step 3: Implement the frozen sets**

In `wings/detection/obb_dataset.py`:

**Edit 1 of 6.** In `wings/detection/obb_dataset.py`, find this text (it occurs exactly once):

```python
"""Build the OBB label table (`labels.csv`) from the raw wing images.

    uv run python -m wings.detection.obb_dataset build      # labels.csv, train.txt, val.txt, test.txt
```

and replace it with:

```python
"""Build the OBB label table (`labels.csv`) from the raw wing images and write the frozen val/test sets.

    uv run python -m wings.detection.obb_dataset build      # labels.csv, train.txt, val.txt, test.txt
    uv run python -m wings.detection.obb_dataset freeze     # frozen val and test sets + dataset.yaml
```

**Edit 2 of 6.** In `wings/detection/obb_dataset.py`, find this text (it occurs exactly once):

```python
import typer
from loguru import logger
```

and replace it with:

```python
import typer
import yaml
from loguru import logger
```

**Edit 3 of 6.** In `wings/detection/obb_dataset.py`, find this text (it occurs exactly once):

```python
from wings.config import COORDS_SUFX, COUNTRIES, IMG_FOLDER_SUFX, PROCESSED_DATA_DIR, RAW_DATA_DIR
```

and replace it with:

```python
from wings.config import COORDS_SUFX, COUNTRIES, IMG_FOLDER_SUFX, PROCESSED_DATA_DIR, RAW_DATA_DIR
from wings.detection.obb_augment import AugConfig, Sample, choose_k, load_item, make_multi_sample, make_single_sample
```

**Edit 4 of 6.** In `wings/detection/obb_dataset.py`, find this text (it occurs exactly once):

```python
DEFAULT_MEAN_SHAPE = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle" / "mean_shape.pth"
```

and replace it with:

```python
DEFAULT_MEAN_SHAPE = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle" / "mean_shape.pth"
FROZEN_JPEG_QUALITY = 95
```

**Edit 5 of 6.** In `wings/detection/obb_dataset.py`, find this text (it occurs exactly once):

```python
@app.command()
def build(
```

and replace it with:

```python
def write_sample(images_dir: Path, labels_dir: Path, stem: str, sample: Sample) -> None:
    """JPEG + Ultralytics OBB label file (`0 x1 y1 x2 y2 x3 y3 x4 y4`, normalized) of one frozen sample."""
    size = sample.image.shape[0]
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), sample.image, [cv2.IMWRITE_JPEG_QUALITY, FROZEN_JPEG_QUALITY])
    lines = ["0 " + " ".join(f"{v:.6f}" for v in (polygon / size).clip(0.0, 1.0).reshape(-1)) for polygon in sample.corners]
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_split(table: pd.DataFrame, split: str, raw_dir: Path, out_dir: Path, seed: int, n_multi: int, cfg: AugConfig = AugConfig()) -> int:
    """Write the frozen version of `split`: one single-wing sample per image and `n_multi` multi-wing compositions.
    Every sample is drawn from its own generator seeded with (seed, split, index, kind), so reruns are identical."""
    images_dir, labels_dir = out_dir / split / "images", out_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows = table[table["split"] == split]
    split_id = SPLITS.index(split)
    for index, row in tqdm(rows.iterrows(), total=len(rows), desc=f"{split} single", unit="img"):
        rng = np.random.default_rng([seed, split_id, int(index), 0])
        sample = make_single_sample(load_item(raw_dir / row["file"], row), rng, cfg)
        write_sample(images_dir, labels_dir, Path(row["file"]).stem, sample)
    for j in tqdm(range(n_multi), desc=f"{split} multi", unit="img"):
        rng = np.random.default_rng([seed, split_id, j, 1])
        picks = rng.integers(0, len(rows), choose_k(rng, cfg))
        items = [load_item(raw_dir / rows.iloc[int(p)]["file"], rows.iloc[int(p)]) for p in picks]
        write_sample(images_dir, labels_dir, f"multi_{j:05d}", make_multi_sample(items, rng, cfg))
    return len(rows) + n_multi


def write_dataset_yaml(out_dir: Path) -> Path:
    path = out_dir / "dataset.yaml"
    content = {"path": out_dir.as_posix(), "train": "train.txt", "val": "val/images", "test": "test/images", "nc": 1, "names": {0: "wing"}}
    path.write_text(yaml.dump(content, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    return path


@app.command()
def build(
```

**Edit 6 of 6.** In `wings/detection/obb_dataset.py`, find this text (it occurs exactly once):

```python
if __name__ == "__main__":
```

and replace it with:

```python
@app.command()
def freeze(
    out: Path = typer.Option(DEFAULT_OUT_DIR, "--out", "-o", help="Folder with labels.csv; the frozen sets are written next to it."),
    seed: int = typer.Option(7, "--seed", help="Seed of the frozen samples."),
    n_multi: int = typer.Option(500, "--n-multi", help="Multi-wing compositions per split."),
) -> None:
    """Write the frozen val and test sets (JPEG + OBB labels) and dataset.yaml."""
    table = pd.read_csv(out / "labels.csv")
    for split in ("val", "test"):
        count = freeze_split(table, split, RAW_DATA_DIR, out, seed, n_multi)
        logger.info(f"{split}: {count} samples")
    logger.info(f"dataset yaml: {write_dataset_yaml(out)}")


if __name__ == "__main__":
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests -v`
Expected: `101 passed`.

- [ ] **Step 5: Write the real frozen sets**

Run: `uv run python -m wings.detection.obb_dataset freeze`
This writes about 5,300 JPEGs and takes a few minutes. Check:

```bash
ls data/processed/detection-obb/val/images | wc -l
ls data/processed/detection-obb/test/images | wc -l
cat data/processed/detection-obb/dataset.yaml
```

Expected: `2697` and `2624` files (2,197 / 2,124 single-wing samples plus 500 compositions each), and a `dataset.yaml` with `train: train.txt`, `val: val/images`, `test: test/images`, `nc: 1`, `names: {0: wing}`.

- [ ] **Step 6: Commit**

```bash
git add wings/detection/obb_dataset.py tests/test_obb_dataset.py
git commit -m "Write the frozen validation and test sets of the OBB detector" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Notebook 31

**Files:**
- Create: `notebooks/31_obb_dataset_and_augmentations.ipynb` (generated, without outputs)
- Scratch, not committed: `<scratchpad>/build_notebook_31.py`, `<scratchpad>/smoke_notebook_31.py` (`<scratchpad>` is the session's scratchpad directory; any temporary directory outside the repository works)

**Interfaces:**
- Consumes: everything above. The notebook imports `AugConfig`, `WingOBBDataset`, `apply_photometric`, `choose_k`, `load_item`, `make_single_sample`, `plan_placement`, `warp_image` from `wings.detection.obb_augment`; `DEFAULT_MEAN_SHAPE`, `DEFAULT_OUT_DIR`, `FROZEN_JPEG_QUALITY` from `wings.detection.obb_dataset`; `CORNER_COLUMNS`, `obb_from_landmarks`, `points_inside_box`, `principal_axis` from `wings.detection.obb_labels`.
- Produces: the notebook (sections 1–6 of the spec). Cell 1 defines `OUT_DIR`, `RAW_DIR` and `MEAN_SHAPE_PATH` on three separate lines; the smoke test replaces exactly those three lines.

The author runs the notebook. Do not execute it; the smoke test below runs its cells on a small synthetic dataset.

- [ ] **Step 1: Create the notebook builder in the scratchpad**

`<scratchpad>/build_notebook_31.py` (the cell sources are the notebook's content):

```python
"""Builds notebooks/31_obb_dataset_and_augmentations.ipynb from the cell sources below (outputs are left empty:
the author runs the notebook). Usage: python build_notebook_31.py <output.ipynb>"""

import sys
from pathlib import Path

import nbformat

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# OBB wing detector: labels, dataset and online augmentation

Stage 1 of the oriented-box detector. Design: `docs/superpowers/specs/2026-10-07-obb-dataset-augmentation-design.md`.

1. **Labels**: the oriented box computed from the 19 landmarks (PCA recipe), and how stable it is.
2. **One wing, step by step**: flip, rotation, scale and placement, photometric steps.
3. **Training samples**: what the model will see, and that the rotation angles are spread evenly over -180...180 degrees.
4. **Real batch**: the boxes handed to the Ultralytics loss are exactly the ones we drew.
5. **Multi-wing compositions**.
6. **Frozen validation and test sets**.

Before running: `uv run python -m wings.detection.obb_dataset build` writes `data/processed/detection-obb/labels.csv`
and the image lists; for section 6 also `uv run python -m wings.detection.obb_dataset freeze`."""),
    (CODE, """from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader
from ultralytics.utils import ops

from wings.config import RAW_DATA_DIR
from wings.detection.obb_augment import AugConfig, WingOBBDataset, apply_photometric, load_item, plan_placement, warp_image
from wings.detection.obb_dataset import DEFAULT_MEAN_SHAPE, DEFAULT_OUT_DIR
from wings.detection.obb_labels import CORNER_COLUMNS, obb_from_landmarks, points_inside_box, principal_axis

OUT_DIR = DEFAULT_OUT_DIR
RAW_DIR = RAW_DATA_DIR
MEAN_SHAPE_PATH = DEFAULT_MEAN_SHAPE

table = pd.read_csv(OUT_DIR / "labels.csv")
by_file = table.set_index("file")
cfg = AugConfig()
print(f"{len(table)} wings, splits: {table['split'].value_counts().to_dict()}")

_coords = {}


def landmarks_of(file: str, frame: str = "image") -> np.ndarray:
    \"\"\"The 19 landmarks of a raw image. frame="image": origin top-left, y down; frame="csv": as in the CSV, y up.\"\"\"
    country, name = file.split("-wing-images/")
    if country not in _coords:
        _coords[country] = pd.read_csv(RAW_DIR / f"{country}-raw-coordinates.csv").set_index("file")
    values = _coords[country].loc[name].to_numpy(np.float64)
    if frame == "csv":
        return np.column_stack([values[0::2], values[1::2]])
    return np.column_stack([values[0::2], int(by_file.loc[file, "img_h"]) - values[1::2] - 1])


def corners_of(row) -> np.ndarray:
    return row[CORNER_COLUMNS].to_numpy(np.float64).reshape(4, 2)


def draw_polygons(ax, polygons, color="lime", lw=2):
    for polygon in polygons:
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw)


def show(ax, image_bgr, title=""):
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))"""),
    (MD, """## 1. Labels (PCA recipe)

The box axis is the direction of the largest spread of the 19 landmarks; the sides span the extreme projections,
enlarged by the same margins as the old axis-aligned labels (x 1.2 along the axis, 1.4 across)."""),
    (CODE, """sample_rows = table.sample(12, random_state=0)
fig, axes = plt.subplots(3, 4, figsize=(16, 7))
for ax, (_, row) in zip(axes.flat, sample_rows.iterrows()):
    show(ax, cv2.imread(str(RAW_DIR / row["file"])), f"{Path(row['file']).name}  axis {row['theta_deg']:.1f} deg")
    points = landmarks_of(row["file"])
    ax.scatter(points[:, 0], points[:, 1], s=10, c="red")
    draw_polygons(ax, [corners_of(row)])
plt.tight_layout()
plt.show()"""),
    (CODE, """fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
axes[0].hist(table["theta_deg"], bins=60)
axes[0].set(title="axis angle [deg]: raw wings are almost horizontal")
axes[1].hist(np.log10(table["eig_ratio"]), bins=60)
axes[1].set(title="log10(eigenvalue ratio): the axis is well defined if >> 0")
axes[2].hist(table["length"] / table["width"], bins=60)
axes[2].set(title="box length / width")
plt.tight_layout()
plt.show()
print(f"axis angle: mean {table['theta_deg'].mean():.2f}, std {table['theta_deg'].std():.2f}, min {table['theta_deg'].min():.1f}, max {table['theta_deg'].max():.1f}")
print("rows with an ill-defined axis (eig_ratio < 1.5):", int((table["eig_ratio"] < 1.5).sum()))

checked = table.sample(min(3000, len(table)), random_state=1)
outside = [row["file"] for _, row in checked.iterrows() if not points_inside_box(corners_of(row), landmarks_of(row["file"]), tol=0.01)]
print(f"wings with a landmark outside their box: {len(outside)} of {len(checked)} checked")"""),
    (CODE, """from wings.gpa import center_shape, normalize_shape, procrustes_align

mean_shape = torch.load(MEAN_SHAPE_PATH, weights_only=False).float()
mean_unit = normalize_shape(center_shape(mean_shape))
mean_axis, _ = principal_axis(mean_shape.numpy())


def procrustes_axis_deg(points_csv: np.ndarray) -> float:
    \"\"\"Axis angle implied by the Procrustes alignment to the mean shape (CSV frame, y up), modulo 180 degrees.\"\"\"
    unit = normalize_shape(center_shape(torch.tensor(points_csv, dtype=torch.float32)))
    rotation = procrustes_align(unit, mean_unit, only_matrix=True, allow_reflection=True)
    direction = mean_axis @ rotation.numpy().T  # unit @ R ~ mean, so the mean axis seen from the wing is mean_axis @ R.T
    return float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)


differences = []
for _, row in table.sample(min(1500, len(table)), random_state=2).iterrows():
    points_csv = landmarks_of(row["file"], frame="csv")
    pca = obb_from_landmarks(points_csv).theta_deg % 180.0
    differences.append((pca - procrustes_axis_deg(points_csv) + 90.0) % 180.0 - 90.0)
differences = np.array(differences)
plt.hist(differences, bins=60)
plt.title("PCA axis minus Procrustes axis [deg]")
plt.show()
print(f"median {np.median(differences):.2f}, 5-95%: {np.percentile(differences, 5):.2f} .. {np.percentile(differences, 95):.2f}, largest |difference| {np.abs(differences).max():.2f}")"""),
    (CODE, """jitter_rng = np.random.default_rng(3)
spreads = []
for _, row in table.sample(min(300, len(table)), random_state=3).iterrows():
    points = landmarks_of(row["file"])
    angles = [obb_from_landmarks(points + jitter_rng.normal(0, 1.0, points.shape)).theta_deg for _ in range(20)]
    spreads.append(np.std(angles))
print(f"axis angle std under 1 px landmark jitter [deg]: median {np.median(spreads):.3f}, 95th percentile {np.percentile(spreads, 95):.3f}")"""),
    (MD, """## 2. One wing, step by step

Flip, rotation about the box centre, scale and position are ONE affine warp, so the corners are transformed exactly.
Areas outside the source image get the image's own background colour (third panel). Gray 114, the Ultralytics default
(fourth panel), would make the rotated corners an easy cue that never appears on real photos."""),
    (CODE, """row = table[table["split"] == "train"].sample(1, random_state=5).iloc[0]
item = load_item(RAW_DIR / row["file"], row)
step_rng = np.random.default_rng(11)
placement = plan_placement(item.corners, item.image.shape[1], (0, 0, cfg.imgsz, cfg.imgsz), step_rng, cfg)
flipped = cv2.flip(item.image, 1)
flipped_corners = item.corners * [-1, 1] + [item.image.shape[1] - 1, 0]
final = warp_image(item.image, placement.matrix, cfg.imgsz, item.background)
gray_fill = warp_image(item.image, placement.matrix, cfg.imgsz, (114, 114, 114))
photometric = apply_photometric(final, step_rng, cfg)
print(f"flip={placement.flip}  angle={placement.angle_deg:.1f} deg  scale={placement.scale:.3f}  background (BGR)={item.background}")

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
show(axes[0], item.image, "1. raw image and box")
draw_polygons(axes[0], [item.corners])
show(axes[1], flipped if placement.flip else item.image, "2. flipped" if placement.flip else "2. not flipped in this draw")
draw_polygons(axes[1], [flipped_corners if placement.flip else item.corners])
show(axes[2], final, "3. rotated, scaled, placed (background fill)")
draw_polygons(axes[2], [placement.corners])
show(axes[3], gray_fill, "3'. same with gray 114 fill")
draw_polygons(axes[3], [placement.corners])
show(axes[4], photometric, "4. photometric steps")
draw_polygons(axes[4], [placement.corners])
plt.tight_layout()
plt.show()"""),
    (MD, """## 3. Training samples

Single-wing samples only (`p_multi=0`). Green: the box that goes into the label."""),
    (CODE, """single = WingOBBDataset(OUT_DIR / "labels.csv", "train", OUT_DIR / "train.txt", raw_dir=RAW_DIR, cfg=AugConfig(p_multi=0.0))
fig, axes = plt.subplots(4, 6, figsize=(21, 14))
for k, ax in enumerate(axes.flat):
    label = single.label_for(int(np.random.default_rng(k).integers(0, len(single))), seed=k)
    show(ax, label["img"], f"sample {k}")
    draw_polygons(ax, label["instances"].segments)
plt.tight_layout()
plt.show()"""),
    (CODE, """train_rows = table[table["split"] == "train"]
draw_rng = np.random.default_rng(21)
angles, box_angles, fractions, centres = [], [], [], []
for _ in range(5000):
    row = train_rows.iloc[int(draw_rng.integers(0, len(train_rows)))]
    placement = plan_placement(corners_of(row), int(row["img_w"]), (0, 0, cfg.imgsz, cfg.imgsz), draw_rng, cfg)
    edge = placement.corners[1] - placement.corners[0]
    angles.append(placement.angle_deg)
    box_angles.append(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
    fractions.append(np.linalg.norm(edge) / cfg.imgsz)
    centres.append(placement.corners.mean(axis=0))
centres = np.array(centres)

fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
axes[0].hist(angles, bins=18, range=(-180, 180))
axes[0].set(title="applied rotation [deg]: should be flat")
axes[1].hist(box_angles, bins=18, range=(0, 180))
axes[1].set(title="box angle modulo 180 [deg]: should be flat")
axes[2].hist(fractions, bins=30)
axes[2].set(title="box length / canvas side")
axes[3].hist2d(centres[:, 0], centres[:, 1], bins=16, range=[[0, cfg.imgsz], [0, cfg.imgsz]])
axes[3].invert_yaxis()
axes[3].set(title="box centres")
plt.tight_layout()
plt.show()
p_rotation = stats.kstest((np.array(angles) + 180) / 360, "uniform").pvalue
p_box = stats.kstest(np.array(box_angles) / 180, "uniform").pvalue
print(f"Kolmogorov-Smirnov p-values for uniformity: rotation {p_rotation:.3f}, box angle {p_box:.3f} (uniform is rejected below 0.01)")
assert p_rotation > 0.01 and p_box > 0.01"""),
    (MD, """## 4. Real batch

A `DataLoader` with the stock Ultralytics `collate_fn`. Yellow: boxes rebuilt from the `xywhr` tensor that enters the loss.
Below, the same boxes are compared numerically with our own corners."""),
    (CODE, """loader = DataLoader(single, batch_size=8, shuffle=True, num_workers=0, collate_fn=WingOBBDataset.collate_fn)
batch = next(iter(loader))
print({key: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__ for key, value in batch.items()})
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, ax in enumerate(axes.flat):
    image = batch["img"][i].permute(1, 2, 0).numpy()  # RGB, uint8
    ax.imshow(image)
    ax.axis("off")
    boxes = batch["bboxes"][batch["batch_idx"] == i].clone()
    boxes[:, [0, 2]] *= image.shape[1]
    boxes[:, [1, 3]] *= image.shape[0]
    draw_polygons(ax, ops.xywhr2xyxyxyxy(boxes).numpy(), color="yellow")
plt.tight_layout()
plt.show()

worst = 0.0
for i in range(64):
    label = single.label_for(i % len(single), seed=i)
    ours = label["instances"].segments.copy()
    formatted = single.transforms(label)
    rboxes = formatted["bboxes"].clone()
    rboxes[:, [0, 2]] *= cfg.imgsz
    rboxes[:, [1, 3]] *= cfg.imgsz
    theirs = ops.xywhr2xyxyxyxy(rboxes).numpy()
    for polygon in ours:
        worst = max(worst, min(vertex_distance(polygon, other) for other in theirs))
print(f"largest corner difference between our boxes and the tensor given to the loss: {worst:.4f} px")
assert worst < 0.5"""),
    (MD, """## 5. Multi-wing compositions

2-4 wings on one canvas, each with its own flip, rotation and scale, in separate cells of a grid, so the boxes cannot overlap."""),
    (CODE, """from wings.detection.obb_augment import choose_k

multi = WingOBBDataset(OUT_DIR / "labels.csv", "train", OUT_DIR / "train.txt", raw_dir=RAW_DIR, cfg=AugConfig(p_multi=1.0))
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
for k, ax in enumerate(axes.flat):
    label = multi.label_for(int(np.random.default_rng(k).integers(0, len(multi))), seed=100 + k)
    show(ax, label["img"], f"composition {k}: {len(label['instances'].segments)} wings")
    draw_polygons(ax, label["instances"].segments)
plt.tight_layout()
plt.show()

ks = pd.Series([choose_k(np.random.default_rng(i)) for i in range(2000)]).value_counts(normalize=True).sort_index()
print("share of compositions with k wings:", ks.round(3).to_dict())

overlapping, outside_canvas = 0, 0
for i in range(300):
    polygons = multi.label_for(i % len(multi), seed=1000 + i)["instances"].segments
    boxes = np.concatenate([polygons.min(axis=1), polygons.max(axis=1)], axis=1)  # x0, y0, x1, y1
    outside_canvas += int(polygons.min() < 0 or polygons.max() > cfg.imgsz)
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            A, B = boxes[a], boxes[b]
            overlapping += int(not (A[2] <= B[0] or B[2] <= A[0] or A[3] <= B[1] or B[3] <= A[1]))
print(f"300 compositions: {overlapping} overlapping pairs, {outside_canvas} boxes outside the canvas")
assert overlapping == 0 and outside_canvas == 0"""),
    (MD, """## 6. Frozen validation and test sets

Generated once by `uv run python -m wings.detection.obb_dataset freeze` with a fixed seed. Below: examples, and a check that
regenerating the first validation samples with the same seeds gives exactly the stored files."""),
    (CODE, """from wings.detection.obb_augment import make_single_sample
from wings.detection.obb_dataset import FROZEN_JPEG_QUALITY

val_dir = OUT_DIR / "val"
if not (val_dir / "images").exists():
    print("Frozen sets not found. Run: uv run python -m wings.detection.obb_dataset freeze")
else:
    stems = sorted(p.stem for p in (val_dir / "images").glob("*.jpg"))
    shown = [s for s in stems if not s.startswith("multi_")][:4] + [s for s in stems if s.startswith("multi_")][:4]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, stem in zip(axes.flat, shown):
        image = cv2.imread(str(val_dir / "images" / f"{stem}.jpg"))
        show(ax, image, stem)
        for line in (val_dir / "labels" / f"{stem}.txt").read_text().strip().splitlines():
            draw_polygons(ax, [np.array(line.split()[1:], dtype=float).reshape(4, 2) * image.shape[0]], color="yellow")
    plt.tight_layout()
    plt.show()

    identical = []
    for _, row in table[table["split"] == "val"].head(10).iterrows():
        rng = np.random.default_rng([7, 1, int(row.name), 0])  # (seed, split id of val, row index, single-wing)
        regenerated = make_single_sample(load_item(RAW_DIR / row["file"], row), rng, cfg).image
        _, buffer = cv2.imencode(".jpg", regenerated, [cv2.IMWRITE_JPEG_QUALITY, FROZEN_JPEG_QUALITY])
        stored = cv2.imread(str(val_dir / "images" / f"{Path(row['file']).stem}.jpg"))
        identical.append(bool((cv2.imdecode(buffer, cv2.IMREAD_COLOR) == stored).all()))
    print("regenerated samples identical to the stored files:", all(identical), identical)
    assert all(identical)"""),
]


def build(path: Path, cells: list) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "wings (3.12.11)", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_markdown_cell(src) if kind == MD else nbformat.v4.new_code_cell(src) for kind, src in cells]
    nbformat.write(nb, path)
    print(f"wrote {path} with {len(nb.cells)} cells")


if __name__ == "__main__":
    build(Path(sys.argv[1]), CELLS)
```

- [ ] **Step 2: Generate the notebook**

Run: `uv run python <scratchpad>/build_notebook_31.py notebooks/31_obb_dataset_and_augmentations.ipynb`
Expected: `wrote notebooks/31_obb_dataset_and_augmentations.ipynb with 18 cells`.

- [ ] **Step 3: Create the smoke test in the scratchpad and run it**

`<scratchpad>/smoke_notebook_31.py`:

```python
"""Smoke test of the cells of notebook 31 on a small synthetic dataset: every code cell must run and every assert pass.
The notebook itself is run by the author on the real data.

Run from the repository root:  uv run python <scratchpad>/smoke_notebook_31.py notebooks/31_obb_dataset_and_augmentations.ipynb
"""

import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nbformat  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "tests")  # the synthetic-data helpers of the unit tests
from obb_synthetic import build_synthetic_raw  # noqa: E402

from wings.detection.obb_dataset import build_labels, freeze_split, write_image_lists  # noqa: E402

plan = {"AA": ["train"] * 30 + ["val"] * 5 + ["test"] * 5, "BB": ["train"] * 25 + ["val"] * 5 + ["test"] * 5}
root = Path(tempfile.mkdtemp())
raw = build_synthetic_raw(root, plan)
out = root / "detection-obb"
out.mkdir()
table = build_labels(raw.raw, raw.countries, raw.split_dir, raw.mean_shape)
table.to_csv(out / "labels.csv", index=False)
write_image_lists(table, raw.raw, out)
torch.save(torch.tensor(raw.mean_shape, dtype=torch.float32), root / "mean_shape.pth")
freeze_split(table, "val", raw.raw, out, seed=7, n_multi=4)

replacements = {
    "OUT_DIR = DEFAULT_OUT_DIR": f'OUT_DIR = Path(r"{out}")',
    "RAW_DIR = RAW_DATA_DIR": f'RAW_DIR = Path(r"{raw.raw}")',
    "MEAN_SHAPE_PATH = DEFAULT_MEAN_SHAPE": f'MEAN_SHAPE_PATH = Path(r"{root / "mean_shape.pth"}")',
}
notebook = nbformat.read(sys.argv[1], as_version=4)
namespace: dict = {}
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != "code":
        continue
    source = cell.source
    for old, new in replacements.items():
        assert old in source or index != 1, f"cell 1 no longer contains: {old}"
        source = source.replace(old, new)
    start = time.time()
    exec(compile(source, f"cell{index}", "exec"), namespace)
    print(f"cell {index}: ok ({time.time() - start:.1f}s)")
print("ALL CELLS RAN")
```

Run (from the repository root): `uv run python <scratchpad>/smoke_notebook_31.py notebooks/31_obb_dataset_and_augmentations.ipynb`
Expected: one `cell N: ok` line per code cell (cells 1, 3, 4, 5, 6, 8, 10, 11, 13, 15, 17) and finally `ALL CELLS RAN`. The PCA-versus-Procrustes numbers it prints are meaningless on synthetic wings (their landmarks have no consistent identity); only the real notebook shows them.

- [ ] **Step 4: Run the whole unit-test suite once more**

Run: `uv run pytest tests -v`
Expected: `101 passed`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/31_obb_dataset_and_augmentations.ipynb
git commit -m "Add notebook 31 showing the OBB labels and augmentations" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Hand over to the author**

Tell the author (do not run it yourself):
1. The label table and the frozen sets already exist (Tasks 2 and 6); the notebook reads them from `data/processed/detection-obb/`.
2. Open `notebooks/31_obb_dataset_and_augmentations.ipynb` and run all cells. Section 3's histogram and section 5's overlap check take the longest (about a minute each).
3. What to look at: the box overlays and the PCA-versus-Procrustes difference in section 1; the gray-fill panel in section 2 (why the background colour fill matters); the sample grid in section 3 (is the scale range and the debris density right?); the flat angle histograms and the two Kolmogorov–Smirnov p-values (both above 0.01); the "largest corner difference" line in section 4 (below 0.5 px); section 5's compositions and the seam check by eye; section 6's reproducibility line (`True`).
4. The parameters the author may want to change after looking at samples live in `AugConfig` (`fraction_range`, `p_multi`, `multi_k_weights`, `triangles_range`, `triangle_size_range`, `blur_p`, `jpeg_p`, the brightness and contrast ranges). A change needs re-running `freeze` for the frozen sets.

Stage 1 is then complete. Not in this plan, each with its own spec: stage 2 (the `OBBTrainer` subclass, `train_obb.py`, HPC run), evaluation against the old detector, and the ONNX export with the `wingai-app` integration.
