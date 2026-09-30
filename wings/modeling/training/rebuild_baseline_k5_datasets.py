"""Rebuilds the static square_size=3 mask datasets that unet_final_k5_v2.py's
load_datasets() reads (data/processed/mask_datasets/rectangle-cropped/
{train,val,test}_mask_dataset_ch1_400.pth, no suffix).

Not the same as wings/dataset.py's own __main__: that one has since moved to
square_size=5 and writes *_sq5.pth-suffixed files for a different, later need
in this project, so it no longer rebuilds the exact files (name or mask size)
unet-final-k5's own recipe depends on. This script targets that original
combination specifically -- square_size=3 (confirmed in
jobs/online/README.md's "Config 4c" section as unet-final-k5.ckpt's own
mask size), unsuffixed filenames.

MaskRectangleDataset.split()'s default seed=42 keeps the same train/val/test
membership as every previous build of these files, so rerunning this only
changes the "label"/mask target field (which now rounds instead of
truncates, per commit 8ef4e61) -- ground truth (orig_label/orig_size) and
which images land in the held-out test set are unaffected.
"""

from functools import partial

import torch
from loguru import logger

from wings.config import COUNTRIES, PROCESSED_DATA_DIR
from wings.dataset import MaskRectangleDataset
from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess

if __name__ == "__main__":
    square_size = 3
    preprocess = partial(unet_fit_rectangle_preprocess, output_size=400)

    mask_dataset = MaskRectangleDataset(
        COUNTRIES, PROCESSED_DATA_DIR / "cropped", preprocess, square_size=square_size
    )

    train_mask_dataset, val_mask_dataset, test_mask_dataset = mask_dataset.split(0.2, 0.1)

    folder = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle-cropped"
    folder.mkdir(parents=True, exist_ok=True)

    torch.save(train_mask_dataset, folder / "train_mask_dataset_ch1_400.pth")
    torch.save(val_mask_dataset, folder / "val_mask_dataset_ch1_400.pth")
    torch.save(test_mask_dataset, folder / "test_mask_dataset_ch1_400.pth")

    logger.info(f"Rebuilt train/val/test mask datasets (square_size={square_size}) in {folder}")
