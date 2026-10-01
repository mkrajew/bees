"""
Online augmentation training -- config 9e (see jobs/online/README.md for the
full comparison table). Combines 9d's `pos_weight=100` with 9a's
`dice_weight=0.7`, both on top of config 8d's radius-3 checkpoint -- checks
whether these two independently-motivated loss changes stack, the same
question this whole five-config sweep is asking about radius vs. dice_weight
(see augmented_unet_9a.py's docstring).

Loss:       BCEDiceLoss(pos_weight=100, dice_weight=0.7, bce_weight=0.3).
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8/8d.
Mask:       circular, square_size=7 (radius 3) -- matches config 8d.

Model UNet(kernel_size=5, sigmoid=False), warm-started from config 8d's own
trained checkpoint. Same weights-only loading as config 9d (not the usual
train(..., checkpoint_path, strict=False)): pos_weight changes relative to
8d's own criterion (50 -> 100), so the standard warm-start would silently
have the checkpoint's saved pos_weight=50 buffer overwrite this config's
pos_weight=100 -- see augmented_unet_9d.py's docstring for the full
explanation of this gotcha (first found in augmented_unet_4b.py).

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a-8f/9a-9d.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "9e"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-9e"
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
    "early_stop_patience": 45,
    "criterion": BCEDiceLoss(pos_weight=100, dice_weight=0.7, bce_weight=0.3),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-90.0, 90.0),
    rotation_p=1.0,
    triangle_noise_p=0.8,
    n_triangles_range=(60, 300),
    triangle_max_size=13,
    color_jitter_p=1.0,
    color_jitter_brightness_range=(0.5, 1.5),
    color_jitter_contrast_range=(0.5, 1.5),
)

if __name__ == "__main__":
    train_val_test_datasets = build_mask_datasets(
        countries=COUNTRIES,
        data_folder=PROCESSED_DATA_DIR / "cropped",
        output_size=400,
        square_size=7,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)

    # Warm-start UNet weights only -- see augmented_unet_9d.py's docstring
    # for why (pos_weight changes relative to 8d's own criterion).
    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8d" / "last.ckpt"
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
