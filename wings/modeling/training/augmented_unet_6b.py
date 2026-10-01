"""
Online augmentation training -- config 6b (see jobs/online/README.md for the
full comparison table). Second of four loss-tuning follow-ups on top of
config 5b's result -- see augmented_unet_6a.py for the shared motivation and
why these are retargeted from config 5 to config 5b (5's rotation_p=0.3 cost
real rotation robustness; 5b corrects it to 0.8).

Config 6b pushes pos_weight even further than 6a's 75, up to 100, to map out
the shape of the recall/precision trade-off's effect on the position-error
metric rather than testing a single point.

Loss:       BCEDiceLoss(pos_weight=100, dice_weight=0.5, bce_weight=0.5).
Augment:    identical to config 5b: Aug B with rotation_p=0.8.
Mask:       circular, square_size=5 (radius 2) -- identical to config 5b.

Model UNet(kernel_size=5), warm-started from config 5b's own trained checkpoint
-- same weights-only loading as config 6a, for the same reason (pos_weight
differs from config 5b's own 50, so the usual strict=False warm-start would
silently reload 50 and discard this file's 100 -- see augmented_unet_4b.py /
6a.py for the full explanation).

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-5/5b/6a.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "6b"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-6b"
PARAMETERS = {
    "project_name": "wingai-online-augmentation",
    "logger_save_dir": TRAINING_DIR / "online",
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
    "criterion": BCEDiceLoss(pos_weight=100, dice_weight=0.5, bce_weight=0.5),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-90.0, 90.0),
    rotation_p=0.8,
    triangle_noise_p=0.8,
    n_triangles_range=(40, 240),
    color_jitter_p=1.0,
    color_jitter_brightness_range=(0.5, 1.5),
    color_jitter_contrast_range=(0.5, 1.5),
)

if __name__ == "__main__":
    train_val_test_datasets = build_mask_datasets(
        countries=COUNTRIES,
        data_folder=PROCESSED_DATA_DIR / "cropped",
        output_size=400,
        square_size=5,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)

    # Warm-start UNet weights only from config 5b's own trained checkpoint --
    # see module docstring for why the criterion must NOT be loaded from it.
    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-5b" / "last.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = {
        key[len("model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state)
    logger.info(f"Warm-started UNet weights from {checkpoint_path}")

    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS, path=None)
