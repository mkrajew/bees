"""
Training UNET with masks with square size 3.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from wings.dataset import load_datasets
from wings.modeling.loss import DiceLoss, WeightedDiceLoss, BCEDiceLoss
from wings.modeling.train import train
from wings.dataset import MaskRectangleDataset
from wings.modeling.unet import UNet

run_num = 3
run_name = "weighted-bce-dice"
model_name = "unet-400-bce-dice"
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
    "early_stop_min_delta": 0.01,
    "early_stop_patience": 25,
    "criterion": BCEDiceLoss(pos_weight=50.0, dice_weight=0.8, bce_weight=0.2),
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

    model = UNet(in_channels=1, out_channels=1, kernel_size=3, sigmoid=False)

    checkpoint_path = MODELS_DIR / "new_unet" / "last-v1.ckpt"

    # checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    # state_dict = checkpoint["state_dict"]

    # # Remove LightningModule prefix: "model."
    # state_dict = {
    #     k.replace("model.", "", 1): v
    #     for k, v in state_dict.items()
    #     if k.startswith("model.")
    # }

    # model.load_state_dict(state_dict)
    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS)
