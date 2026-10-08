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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics.data.augment import Compose, Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.instance import Instances

from wings.detection.dataset import fill_white_spots, is_near_white
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
    p_multi: float = 0.4  # probability of a multi-wing composition
    multi_k_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)  # k = 2, 3, 4


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
    Fails loudly if the file is unreadable or its size differs from the one in the table.
    As in `pad_image`, pure-white spots (the wedges the scan rotation left in the corners of most raw scans) are
    painted over with the background colour unless that colour is itself near white: rotated with the wing they
    would be a sharp cue for its axis that real photos do not have."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    if image.shape[:2] != (int(row["img_h"]), int(row["img_w"])):
        raise ValueError(f"{path} has shape {image.shape[:2]}, the labels table says {(int(row['img_h']), int(row['img_w']))}")
    corners = row[CORNER_COLUMNS].to_numpy(np.float64).reshape(4, 2)
    background = (int(row["bg_b"]), int(row["bg_g"]), int(row["bg_r"]))
    if not is_near_white(background):
        image = fill_white_spots(image, background)
    return Item(image, corners, background)


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


@lru_cache(maxsize=None)
def _resolved_folder(folder: str) -> str:
    return os.path.realpath(folder)


def _canonical_path(path: str | os.PathLike) -> str:
    """Comparable form of an image path: normalised, links in its folder resolved, case folded on Windows. The image
    lists hold resolved paths (`write_image_lists`) while the raw folder may be given as a link, as `data/` often is
    on clusters. Only the folder is resolved (and cached), so 20,000 images cost a handful of lookups."""
    folder, name = os.path.split(os.path.normpath(str(path)))
    return os.path.normcase(os.path.join(_resolved_folder(folder), name))


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
        self._row_of = {_canonical_path(self.raw_dir / f): i for i, f in enumerate(self.table["file"])}
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
            key = _canonical_path(im_file)
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
        if rng.random() < self.cfg.p_multi:
            k = choose_k(rng, self.cfg)
            items = [self.load_item(index)] + [self.load_item(int(j)) for j in rng.integers(0, len(self), k - 1)]
            sample = make_multi_sample(items, rng, self.cfg)
        else:
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
