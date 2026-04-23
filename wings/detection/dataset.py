"""Build an Ultralytics YOLO detection dataset from raw wing images."""

from pathlib import Path

import cv2
import numpy as np
import typer
import yaml
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray
import pandas as pd

from wings.config import (
    COUNTRIES,
    IMG_FOLDER_SUFX,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    COORDS_SUFX,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_OUTPUT_FOLDER = PROCESSED_DATA_DIR / "detection"
app = typer.Typer()


def collect_image_paths(raw_data_dir: Path, countries: list[str]) -> list[Path]:
    """Collect image paths from country-specific image directories."""
    image_paths: list[Path] = []
    logger.info(f"Collecting image paths from: {raw_data_dir}")

    for country in countries:
        country_dir = raw_data_dir / f"{country}{IMG_FOLDER_SUFX}"
        if not country_dir.exists():
            logger.warning(f"Missing directory: {country_dir}")
            continue

        for path in sorted(country_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

    return image_paths


def choose_split() -> str:
    """Choose a split with default train/val/test probabilities."""
    value = np.random.random()
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def is_near_white(color: tuple[int, int, int], threshold: int = 240) -> bool:
    return all(c >= threshold for c in color)


def fill_white_spots(
    img: NDArray[np.uint8],
    fill_color: tuple[int, int, int],
    white_threshold: int = 245,
) -> NDArray[np.uint8]:
    out = img.copy()
    mask = np.all(out >= white_threshold, axis=2)
    out[mask] = fill_color
    return out


def dominant_inner_border_color(img, offset=2, width=5, q=8):
    frame = np.concatenate(
        [
            img[offset : offset + width].reshape(-1, 3),
            img[-offset - width : -offset].reshape(-1, 3),
            img[:, offset : offset + width].reshape(-1, 3),
            img[:, -offset - width : -offset].reshape(-1, 3),
        ]
    )

    frame_q = (frame // q) * q
    colors, counts = np.unique(frame_q, axis=0, return_counts=True)
    return tuple(map(int, colors[counts.argmax()]))


def pad_image(
    img: NDArray[np.uint8],
    pad: tuple[int, int, int, int],
    *,
    d=5,
    offset=2,
    width=5,
    q=8,
    white_threshold: int = 245,
    near_white_threshold: int = 240,
):
    frame = np.concatenate([img[d], img[-d - 1], img[:, d], img[:, -d - 1]])
    colors, counts = np.unique(frame, axis=0, return_counts=True)
    color = tuple(map(int, colors[counts.argmax()]))
    if color in [(255, 255, 255), (0, 0, 0)]:
        color = dominant_inner_border_color(img, offset=offset, width=width, q=q)

    if not is_near_white(color, threshold=near_white_threshold):
        img = fill_white_spots(img, color, white_threshold=white_threshold)

    out = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=color)
    return out


def read_coordinates(filepath: Path, img: np.ndarray):
    filename = filepath.name
    country = filename.split("-", 1)[0]

    df = pd.read_csv(RAW_DATA_DIR / f"{country}{COORDS_SUFX}")
    row = df[df["file"] == filename].iloc[0]

    targets = pd.to_numeric(row.iloc[1:].values)

    x_coords, y_coords = targets[::2], targets[1::2]
    y_size = img.shape[0]
    y_coords = y_size - y_coords - 1

    return x_coords, y_coords


def process_bbox(x_coords, y_coords, img, x_factor=1.2, y_factor=1.4):
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    w = (x_max - x_min) * x_factor
    h = (y_max - y_min) * y_factor

    x_size, y_size = img.shape[1], img.shape[0]

    cx = round(cx / x_size, 6)
    cy = round(cy / y_size, 6)
    w = round(w / x_size, 6)
    h = round(h / y_size, 6)

    return [0, cx, cy, w, h]


def build_dataset_yolo(
    output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    countries: list[str] | None = None,
) -> dict[str, int]:
    """Generate an Ultralytics YOLO dataset with images/{split}/ and labels/{split}/ layout."""
    selected_countries = countries or COUNTRIES
    logger.info(f"Starting YOLO dataset build for {len(selected_countries)} countries")
    logger.info(f"Output folder: {output_folder}")

    image_paths = collect_image_paths(RAW_DATA_DIR, selected_countries)
    if not image_paths:
        raise RuntimeError("No images found for the selected countries.")
    logger.info(f"Total candidate images: {len(image_paths)}")

    for split in ("train", "val", "test"):
        (output_folder / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_folder / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0, "test": 0}
    written = 0

    for image_path in tqdm(image_paths, desc="Processing images", unit="img"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"Unreadable image skipped: {image_path}")
            continue

        x_coords, y_coords = read_coordinates(image_path, image)

        pad_values = tuple(np.random.randint(100, 600) for _ in range(4))
        padded = pad_image(image, pad_values)

        x_coords = x_coords + pad_values[2]
        y_coords = y_coords + pad_values[0]

        bbox = process_bbox(x_coords, y_coords, padded)
        split = choose_split()

        img_save_path = output_folder / "images" / split / image_path.name
        cv2.imwrite(str(img_save_path), padded)

        label_save_path = output_folder / "labels" / split / (image_path.stem + ".txt")
        class_id, cx, cy, w, h = bbox
        label_save_path.write_text(f"{class_id} {cx} {cy} {w} {h}\n", encoding="utf-8")

        split_counts[split] += 1
        written += 1

    if written == 0:
        raise RuntimeError("No records were written (all images were unreadable).")

    yaml_path = output_folder / "dataset.yaml"
    yaml_content = {
        "path": output_folder.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "wing"},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Saved {written} image records to {output_folder}")
    logger.info(f"Split counts: {split_counts}")
    logger.info(f"Dataset YAML written to {yaml_path}")

    return {"total": written, **split_counts}


@app.command()
def main(
    output: Path = typer.Option(
        DEFAULT_OUTPUT_FOLDER,
        "--output",
        "-o",
        help="Output folder.",
    ),
) -> None:
    """Create Ultralytics YOLO dataset from raw wing images."""
    build_dataset_yolo(output_folder=output)


if __name__ == "__main__":
    app()
