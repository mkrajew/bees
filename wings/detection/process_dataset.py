"""Crop raw wing images with a YOLO detector and update landmark coordinates."""

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
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
)

DEFAULT_OUTPUT_FOLDER = PROCESSED_DATA_DIR / "cropped"
DEFAULT_YOLO_MODEL = MODELS_DIR / "best.pt"

app = typer.Typer()


def _read_row_coordinates(
    row: pd.Series, image_height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read CSV coordinates and convert y to image top-left convention."""
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
    """Return the highest-confidence bounding box as integer xyxy coords."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(conf))
    x1, y1, x2, y2 = boxes_xyxy[best_idx]

    xmin = int(np.floor(min(x1, x2)))
    ymin = int(np.floor(min(y1, y2)))
    xmax = int(np.ceil(max(x1, x2)))
    ymax = int(np.ceil(max(y1, y2)))
    return xmin, ymin, xmax, ymax


def _clip_box_to_image(
    box: tuple[int, int, int, int], image_shape: tuple[int, int, int]
) -> tuple[int, int, int, int] | None:
    h, w = image_shape[:2]
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(xmin, w))
    xmax = max(0, min(xmax, w))
    ymin = max(0, min(ymin, h))
    ymax = max(0, min(ymax, h))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def process_country(
    *,
    country: str,
    model: YOLO,
    output_folder: Path,
    conf_threshold: float,
    keep_original_on_fail: bool,
    clip_coords: bool,
) -> dict[str, int]:
    coords_path = RAW_DATA_DIR / f"{country}{COORDS_SUFX}"
    image_dir = RAW_DATA_DIR / f"{country}{IMG_FOLDER_SUFX}"
    out_image_dir = output_folder / f"{country}{IMG_FOLDER_SUFX}"
    out_coords_path = output_folder / f"{country}{COORDS_SUFX}"

    if not coords_path.exists():
        logger.warning(f"Skipping {country}: missing coordinates file {coords_path}")
        return {"total": 0, "cropped": 0, "fallback": 0, "skipped": 0}

    if not image_dir.exists():
        logger.warning(f"Skipping {country}: missing image directory {image_dir}")
        return {"total": 0, "cropped": 0, "fallback": 0, "skipped": 0}

    out_image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(coords_path)
    out_columns = list(df.columns)
    out_records: list[list[Any]] = []

    stats = {"total": 0, "cropped": 0, "fallback": 0, "skipped": 0}

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"{country} images", unit="img"
    ):
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

        result = model.predict(image, conf=conf_threshold, verbose=False)[0]
        best_box = _pick_best_box(result)
        clipped_box = _clip_box_to_image(best_box, image.shape) if best_box else None

        if clipped_box is None:
            if not keep_original_on_fail:
                logger.warning(f"No valid YOLO box, sample skipped: {image_path}")
                stats["skipped"] += 1
                continue
            crop = image
            new_x_top = x_coords
            new_y_top = y_coords_top
            stats["fallback"] += 1
        else:
            xmin, ymin, xmax, ymax = clipped_box
            crop = image[ymin:ymax, xmin:xmax]
            new_x_top = x_coords - xmin
            new_y_top = y_coords_top - ymin
            stats["cropped"] += 1

        crop_h, crop_w = crop.shape[:2]
        if clip_coords:
            new_x_top = np.clip(new_x_top, 0, crop_w - 1)
            new_y_top = np.clip(new_y_top, 0, crop_h - 1)

        new_y_bottom = crop_h - new_y_top - 1
        updated_coords = _interleave_xy(new_x_top, new_y_bottom)
        out_records.append([filename, *updated_coords])

        out_image_path = out_image_dir / filename
        cv2.imwrite(str(out_image_path), crop)

    out_df = pd.DataFrame(out_records, columns=out_columns)
    out_df.to_csv(out_coords_path, index=False)
    logger.info(f"Wrote updated coordinates: {out_coords_path}")

    return stats


def process_dataset(
    *,
    yolo_model_path: Path,
    output_folder: Path,
    countries: list[str],
    conf_threshold: float,
    keep_original_on_fail: bool,
    clip_coords: bool,
) -> dict[str, int]:
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")

    output_folder.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(yolo_model_path))

    totals = {"total": 0, "cropped": 0, "fallback": 0, "skipped": 0}
    for country in countries:
        logger.info(f"Processing country: {country}")
        stats = process_country(
            country=country,
            model=model,
            output_folder=output_folder,
            conf_threshold=conf_threshold,
            keep_original_on_fail=keep_original_on_fail,
            clip_coords=clip_coords,
        )
        for key in totals:
            totals[key] += stats[key]

    logger.info(f"Completed dataset processing into: {output_folder}")
    logger.info(f"Summary: {totals}")
    return totals


@app.command()
def main(
    model: Path = typer.Option(
        DEFAULT_YOLO_MODEL,
        "--model",
        "-m",
        help="Path to trained YOLO weights file.",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT_FOLDER,
        "--output",
        "-o",
        help="Output folder for cropped images and updated CSV files.",
    ),
    countries: list[str] | None = typer.Option(
        None,
        "--country",
        "-c",
        help="Country code to process. Repeat the option to pass multiple countries.",
    ),
    conf: float = typer.Option(
        0.25,
        "--conf",
        help="YOLO confidence threshold.",
    ),
    keep_original_on_fail: bool = typer.Option(
        True,
        "--keep-original-on-fail/--skip-on-fail",
        help="Keep original image/coords when YOLO fails, or skip such samples.",
    ),
    clip_coords: bool = typer.Option(
        True,
        "--clip-coords/--no-clip-coords",
        help="Clip updated coordinates to the cropped image boundaries.",
    ),
) -> None:
    selected_countries = countries or COUNTRIES
    process_dataset(
        yolo_model_path=model,
        output_folder=output,
        countries=selected_countries,
        conf_threshold=conf,
        keep_original_on_fail=keep_original_on_fail,
        clip_coords=clip_coords,
    )


if __name__ == "__main__":
    app()
