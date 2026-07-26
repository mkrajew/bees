"""
Training UNET with masks with square size 3.

Augmentation (random rotation, triangle noise, color jitter) is applied online,
freshly on every training access, via `wings.dataset.build_mask_datasets` +
`wings.transforms` -- see wings/transforms.py for the transform pipeline. Reads
from the plain YOLO-cropped `data/processed/cropped/` folder; there is no offline
augmented copy on disk anymore.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets
from wings.modeling.loss import DiceLoss, WeightedDiceLoss, BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = 3
run_name = "weighted-bce-dice"
model_name = "unet-400-bce-dice"
PARAMETERS = {
    "project_name": "wingai",
    "logger_save_dir": TRAINING_DIR,
    "run_name": f"{run_name}-{run_num}",
    "checkpoint_save_dir": TRAINING_DIR / "lightning-checkpoints" / model_name,
    "checkpoint_filename": model_name
    + "-{epoch:02d}-{val_mean_error_px:.4f}-"
    + f"{run_name}-{run_num}",
    "num_epochs": 100,
    "batch_size": 12,
    "num_workers": 8,
    "early_stop_min_delta": 0.01,
    "early_stop_patience": 25,
    "criterion": WeightedDiceLoss(landmark_weight=100),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig()

if __name__ == "__main__":
    train_val_test_datasets = build_mask_datasets(
        countries=COUNTRIES,
        data_folder=PROCESSED_DATA_DIR / "cropped",
        output_size=400,
        square_size=5,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
    )
    logger.info("Built datasets.")

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

    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path)
