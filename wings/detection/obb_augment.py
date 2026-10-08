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
