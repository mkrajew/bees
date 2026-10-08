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
