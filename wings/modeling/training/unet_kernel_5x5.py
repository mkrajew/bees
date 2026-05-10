"""
Training UNET with masks with square size 5.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from wings.dataset import load_datasets
from wings.modeling.loss import DiceLoss, WeightedDiceLoss
from wings.modeling.train import train
from wings.dataset import MaskRectangleDataset
from wings.modeling.unet import UNet, load_lightning_3x3_into_unet_5x5

run_num = 1
run_name = "kernel-5x5"
model_name = "kernel-5x5"
PARAMETERS = {
    "project_name": "wingai",
    "logger_save_dir": TRAINING_DIR,
    "run_name": f"{run_name}-{run_num}",
    "checkpoint_save_dir": TRAINING_DIR / "lightning-checkpoints" / model_name,
    "checkpoint_filename": model_name
    + "-{epoch:02d}-{val_loss:.2f}-"
    + f"{run_name}-{run_num}",
    "num_epochs": 100,
    "batch_size": 12,
    "num_workers": 8,
    "early_stop_min_delta": 0.001,
    "early_stop_patience": 50,
    "criterion": WeightedDiceLoss(landmark_weight=50.0, background_weight=1.0),
}

if __name__ == "__main__":
    data_dir = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle-cropped"
    train_val_test_datasets = load_datasets(
        [
            data_dir / "train_mask_dataset_ch1_400.pth",
            data_dir / "val_mask_dataset_ch1_400.pth",
            data_dir / "test_mask_dataset_ch1_400.pth",
        ]
    )
    logger.info("Loaded datasets.")
    checkpoint_path = (
        MODELS_DIR
        / "new_unet"
        / "custom-unet-pretrained-epoch=49-val_loss=0.02-custom-unet-training_1.ckpt"
    )
    model = UNet(in_channels=1, out_channels=1, kernel_size=5)
    model = load_lightning_3x3_into_unet_5x5(
        model,
        checkpoint_path,
    )
    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS)
