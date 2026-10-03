"""
Online augmentation training -- config 12 (see jobs/online/README.md).
Continues config 11 from its own best epoch (epoch 38) with exactly one
change: `rotation_degrees` widened from (-90, 90) to (-180, 180), so training
now also covers upside-down wings -- config 11 never saw anything past +/-90.

Everything else is config 11's: BCEDiceLoss(pos_weight=50, dice_weight=0.7,
bce_weight=0.3), circular mask radius 3, Aug B (rotation_p=1.0, triangles
60-300, size 13, color jitter 0.5-1.5), num_epochs=100,
early_stop_patience=60, plain val_mean_error_px checkpoint monitoring,
save_top_k=5.

Warm-start: config 11's epoch-38 checkpoint from the cluster's own
lightning-checkpoints dir (saved locally as models/final/final-rotation.ckpt).
The filename embeds that epoch's val_mean_error_px, so it's resolved by
globbing for "epoch=38" rather than hardcoding the full name. Same criterion
class and pos_weight as the checkpoint's own, so the standard strict load is
safe (verified locally against final-rotation.ckpt).
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "12"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-12"
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
    "early_stop_patience": 60,
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3),
    "checkpoint_monitor": "val_mean_error_px",
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-180.0, 180.0),
    rotation_p=1.0,
    triangle_noise_p=0.8,
    n_triangles_range=(60, 300),
    triangle_max_size=13,
    color_jitter_p=1.0,
    color_jitter_brightness_range=(0.5, 1.5),
    color_jitter_contrast_range=(0.5, 1.5),
)

WARM_START_DIR = (
    TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-11"
)
WARM_START_EPOCH = 38

if __name__ == "__main__":
    # Resolved first so a wrong/missing checkpoint fails immediately, not
    # after the datasets have been built.
    matches = sorted(WARM_START_DIR.glob(f"*-epoch={WARM_START_EPOCH}-*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one epoch={WARM_START_EPOCH} checkpoint in "
            f"{WARM_START_DIR}, found {[p.name for p in matches]}; the directory "
            f"contains {sorted(p.name for p in WARM_START_DIR.glob('*.ckpt'))}"
        )
    checkpoint_path = matches[0]
    logger.info(f"Warm-starting from {checkpoint_path}")

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
    model.to(DEVICE)

    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path)
