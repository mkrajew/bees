"""
Create an augmented cropped-wing dataset.

For each raw image the pipeline produces two samples:
  1. Original: rotate image (0°) → YOLO crop → add triangle noise
  2. Rotated:  rotate image (random ±angle_range°) → YOLO crop → add triangle noise

Coordinates are transformed consistently at every step.
All coordinates in intermediate steps use the top-left (image) convention;
they are saved back to bottom-left convention (as in the original CSVs).
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO

from wings.config import (
    COORDS_SUFX,
    COUNTRIES,
    IMG_FOLDER_SUFX,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

DEFAULT_OUTPUT_FOLDER = PROCESSED_DATA_DIR / "cropped-augmented"
DEFAULT_YOLO_MODEL = MODELS_DIR / "yolo26n" / "best.pt"

app = typer.Typer()


def rotate_image_and_coords(
    image: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate image and coordinates by `angle` degrees (counter-clockwise).

    The canvas is expanded so no wing content is clipped during rotation.
    Coordinates must be in top-left (image) convention on input and are
    returned in the same convention.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))

    pts = np.column_stack([x_coords, y_coords, np.ones(len(x_coords))])
    transformed = pts @ M.T
    new_x = transformed[:, 0]
    new_y = transformed[:, 1]

    return rotated, new_x, new_y


def add_triangle_noise(
    image: np.ndarray,
    n_triangles: int = 80,
    min_size: int = 2,
    max_size: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw random small dark filled triangles at random positions and orientations.

    Each triangle is an isoceles shape placed at a random centre, scaled to a
    random size in [min_size, max_size] pixels, and rotated by a random angle.
    This mimics the debris/dust noise seen in real microscopy wing images.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = image.copy()
    h, w = result.shape[:2]

    # Unit isoceles triangle (tip up, base down), centred at origin
    tip_ratio = 0.9
    base_half = 0.55
    unit_tri = np.array([
        [0.0,        -tip_ratio],
        [-base_half,  0.6],
        [ base_half,  0.6],
    ], dtype=np.float32)

    for _ in range(n_triangles):
        cx = float(rng.uniform(0, w))
        cy = float(rng.uniform(0, h))
        size = float(rng.uniform(min_size, max_size))
        angle = float(rng.uniform(0, 2 * np.pi))

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        pts = (rot @ (unit_tri * size).T).T
        pts[:, 0] += cx
        pts[:, 1] += cy

        cv2.fillPoly(result, [pts.astype(np.int32)], (0, 0, 0))

    return result


def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random brightness and contrast jitter to an image."""
    if rng is None:
        rng = np.random.default_rng()

    result = image.astype(np.float32)

    b_factor = float(rng.uniform(1 - brightness, 1 + brightness))
    result *= b_factor

    c_factor = float(rng.uniform(1 - contrast, 1 + contrast))
    mean = result.mean()
    result = mean + c_factor * (result - mean)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers (mirrors of process_dataset.py)
# ---------------------------------------------------------------------------

def _read_row_coordinates(
    row: pd.Series, image_height: int
) -> tuple[np.ndarray, np.ndarray]:
    raw_values = pd.to_numeric(row.iloc[1:], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(raw_values).any():
        raise ValueError(f"Found non-numeric coordinates for file: {row['file']}")
    x_coords = raw_values[::2]
    y_coords_bottom = raw_values[1::2]
    y_coords_top = image_height - y_coords_bottom - 1
    return x_coords, y_coords_top


def _interleave_xy(x_coords: np.ndarray, y_coords: np.ndarray) -> list[int]:
    coords = np.empty(x_coords.size + y_coords.size, dtype=np.int32)
    coords[0::2] = np.rint(x_coords).astype(np.int32)
    coords[1::2] = np.rint(y_coords).astype(np.int32)
    return coords.tolist()


def _pick_best_box(result: Any) -> tuple[int, int, int, int] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(conf))
    x1, y1, x2, y2 = boxes_xyxy[best_idx]
    return (
        int(np.floor(min(x1, x2))),
        int(np.floor(min(y1, y2))),
        int(np.ceil(max(x1, x2))),
        int(np.ceil(max(y1, y2))),
    )


def _clip_box(
    box: tuple[int, int, int, int], image_shape: tuple
) -> tuple[int, int, int, int] | None:
    h, w = image_shape[:2]
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0, min(xmin, w)), max(0, min(xmax, w))
    ymin, ymax = max(0, min(ymin, h)), max(0, min(ymax, h))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _detect_and_crop(
    image: np.ndarray,
    x_coords_top: np.ndarray,
    y_coords_top: np.ndarray,
    model: YOLO,
    conf_threshold: float,
    keep_original_on_fail: bool,
    clip_coords: bool,
    rng: np.random.Generator,
    n_triangles: int,
    min_size: int,
    max_size: int,
    brightness: float,
    contrast: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Run YOLO, crop, apply augmentations. Returns (crop, new_x_top, new_y_bottom) or None."""
    yolo_result = model.predict(image, conf=conf_threshold, verbose=False)[0]
    best_box = _pick_best_box(yolo_result)
    clipped_box = _clip_box(best_box, image.shape) if best_box else None

    if clipped_box is None:
        if not keep_original_on_fail:
            return None
        crop = image.copy()
        new_x_top = x_coords_top.copy()
        new_y_top = y_coords_top.copy()
    else:
        xmin, ymin, xmax, ymax = clipped_box
        crop = image[ymin:ymax, xmin:xmax]
        new_x_top = x_coords_top - xmin
        new_y_top = y_coords_top - ymin

    crop_h, crop_w = crop.shape[:2]
    if clip_coords:
        new_x_top = np.clip(new_x_top, 0, crop_w - 1)
        new_y_top = np.clip(new_y_top, 0, crop_h - 1)

    crop = add_triangle_noise(crop, n_triangles=n_triangles, min_size=min_size, max_size=max_size, rng=rng)
    crop = apply_color_jitter(crop, brightness=brightness, contrast=contrast, rng=rng)
    new_y_bottom = crop_h - new_y_top - 1

    return crop, new_x_top, new_y_bottom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_country_augmented(
    *,
    country: str,
    model: YOLO,
    output_folder: Path,
    conf_threshold: float,
    keep_original_on_fail: bool,
    clip_coords: bool,
    min_angle: float,
    max_angle: float,
    n_triangles: int,
    min_size: int,
    max_size: int,
    brightness: float,
    contrast: float,
    seed: int | None = None,
) -> dict[str, int]:
    coords_path = RAW_DATA_DIR / f"{country}{COORDS_SUFX}"
    image_dir = RAW_DATA_DIR / f"{country}{IMG_FOLDER_SUFX}"
    out_image_dir = output_folder / f"{country}{IMG_FOLDER_SUFX}"
    out_coords_path = output_folder / f"{country}{COORDS_SUFX}"

    if not coords_path.exists():
        logger.warning(f"Skipping {country}: missing {coords_path}")
        return {"total": 0, "original": 0, "rotated": 0, "skipped": 0}
    if not image_dir.exists():
        logger.warning(f"Skipping {country}: missing {image_dir}")
        return {"total": 0, "original": 0, "rotated": 0, "skipped": 0}

    out_image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(coords_path)
    out_columns = list(df.columns)
    out_records: list[list[Any]] = []

    rng = np.random.default_rng(seed)
    stats = {"total": 0, "original": 0, "rotated": 0, "skipped": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{country}", unit="img"):
        filename = str(row["file"])
        image_path = image_dir / filename
        stats["total"] += 1

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"Unreadable image skipped: {image_path}")
            stats["skipped"] += 1
            continue

        h = image.shape[0]
        try:
            x_coords, y_coords_top = _read_row_coordinates(row, h)
        except ValueError as exc:
            logger.warning(str(exc))
            stats["skipped"] += 1
            continue

        # --- Original (no rotation) ---
        orig_result = _detect_and_crop(
            image, x_coords, y_coords_top,
            model, conf_threshold, keep_original_on_fail, clip_coords,
            rng, n_triangles, min_size, max_size, brightness, contrast,
        )
        if orig_result is not None:
            crop, new_x, new_y_bottom = orig_result
            out_records.append([filename, *_interleave_xy(new_x, new_y_bottom)])
            cv2.imwrite(str(out_image_dir / filename), crop)
            stats["original"] += 1
        else:
            logger.warning(f"No YOLO box for original, skipped: {image_path}")
            stats["skipped"] += 1

        # --- Rotated version ---
        sign = rng.choice([-1.0, 1.0])
        angle = float(sign * rng.uniform(min_angle, max_angle))
        rot_image, rot_x, rot_y_top = rotate_image_and_coords(image, x_coords, y_coords_top, angle)

        rot_result = _detect_and_crop(
            rot_image, rot_x, rot_y_top,
            model, conf_threshold, keep_original_on_fail, clip_coords,
            rng, n_triangles, min_size, max_size, brightness, contrast,
        )
        if rot_result is not None:
            rot_crop, rot_new_x, rot_new_y_bottom = rot_result
            stem, suffix = Path(filename).stem, Path(filename).suffix
            rot_filename = f"{stem}_rot{angle:.1f}{suffix}"
            out_records.append([rot_filename, *_interleave_xy(rot_new_x, rot_new_y_bottom)])
            cv2.imwrite(str(out_image_dir / rot_filename), rot_crop)
            stats["rotated"] += 1
        else:
            logger.warning(f"No YOLO box for rotated image, skipped: {image_path}")

    out_df = pd.DataFrame(out_records, columns=out_columns)
    out_df.to_csv(out_coords_path, index=False)
    logger.info(f"Wrote {len(out_records)} records to {out_coords_path}")

    return stats


def create_augmented_dataset(
    *,
    yolo_model_path: Path = DEFAULT_YOLO_MODEL,
    output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    countries: list[str] | None = None,
    conf_threshold: float = 0.25,
    keep_original_on_fail: bool = True,
    clip_coords: bool = True,
    min_angle: float = 20.0,
    max_angle: float = 45.0,
    n_triangles: int = 80,
    min_size: int = 2,
    max_size: int = 6,
    brightness: float = 0.3,
    contrast: float = 0.3,
    seed: int | None = None,
) -> dict[str, int]:
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")

    selected = countries or COUNTRIES
    output_folder.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(yolo_model_path))

    totals = {"total": 0, "original": 0, "rotated": 0, "skipped": 0}
    for country in selected:
        logger.info(f"Processing country: {country}")
        stats = process_country_augmented(
            country=country,
            model=model,
            output_folder=output_folder,
            conf_threshold=conf_threshold,
            keep_original_on_fail=keep_original_on_fail,
            clip_coords=clip_coords,
            min_angle=min_angle,
            max_angle=max_angle,
            n_triangles=n_triangles,
            min_size=min_size,
            max_size=max_size,
            brightness=brightness,
            contrast=contrast,
            seed=seed,
        )
        for key in totals:
            totals[key] += stats[key]

    logger.info(f"Done. Output: {output_folder}")
    logger.info(f"Summary: {totals}")
    return totals


@app.command()
def main(
    model: Path = typer.Option(DEFAULT_YOLO_MODEL, "--model", "-m", help="YOLO weights path"),
    output: Path = typer.Option(DEFAULT_OUTPUT_FOLDER, "--output", "-o", help="Output folder"),
    countries: list[str] | None = typer.Option(None, "--country", "-c", help="Country code (repeatable)"),
    conf: float = typer.Option(0.25, "--conf", help="YOLO confidence threshold"),
    keep_original_on_fail: bool = typer.Option(True, "--keep-original-on-fail/--skip-on-fail"),
    clip_coords: bool = typer.Option(True, "--clip-coords/--no-clip-coords"),
    min_angle: float = typer.Option(20.0, "--min-angle", help="Minimum rotation magnitude in degrees"),
    max_angle: float = typer.Option(45.0, "--max-angle", help="Maximum rotation magnitude in degrees"),
    n_triangles: int = typer.Option(80, "--n-triangles", help="Number of triangle noise shapes per image"),
    min_size: int = typer.Option(2, "--min-size", help="Minimum triangle size in pixels"),
    max_size: int = typer.Option(6, "--max-size", help="Maximum triangle size in pixels"),
    brightness: float = typer.Option(0.3, "--brightness", help="Brightness jitter factor"),
    contrast: float = typer.Option(0.3, "--contrast", help="Contrast jitter factor"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Create an augmented dataset: original + rotated version of each image, both with triangle noise."""
    create_augmented_dataset(
        yolo_model_path=model,
        output_folder=output,
        countries=list(countries) if countries else None,
        conf_threshold=conf,
        keep_original_on_fail=keep_original_on_fail,
        clip_coords=clip_coords,
        min_angle=min_angle,
        max_angle=max_angle,
        n_triangles=n_triangles,
        min_size=min_size,
        max_size=max_size,
        brightness=brightness,
        contrast=contrast,
        seed=seed,
    )


if __name__ == "__main__":
    app()
