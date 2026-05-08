"""
Training UNET with masks with square size 5.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR
from wings.dataset import load_datasets
from wings.modeling.loss import DiceLoss
from wings.modeling.train import train

from wings.modeling.unet import UNet

run_num = 1
run_name = "unet-training"
model_name = "custom-unet"
PARAMETERS = {
    "project_name": "bees-wings-modeling3",
    "logger_save_dir": TRAINING_DIR,
    "run_name": f"{run_name}_{run_num}",
    "checkpoint_save_dir": TRAINING_DIR / "lightning-checkpoints" / model_name,
    "checkpoint_filename": model_name
    + "-{epoch:02d}-{val_loss:.2f}-"
    + f"{run_name}_{run_num}",
    "num_epochs": 100,
    "batch_size": 16,
    "num_workers": 10,
    "early_stop_min_delta": 0.1,
    "early_stop_patience": 10,
    "criterion": DiceLoss(),
}

if __name__ == "__main__":
    data_dir = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle"
    train_val_test_datasets = load_datasets(
        [
            data_dir / "train_mask_dataset_ch1_400.pth",
            data_dir / "val_mask_dataset_ch1_400.pth",
            data_dir / "test_mask_dataset_ch1_400.pth",
        ]
    )
    logger.info("Loaded datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=3)
    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS)
