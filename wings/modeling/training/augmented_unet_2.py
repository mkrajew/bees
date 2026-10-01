"""
Online augmentation training -- config 2 (see jobs/online/README.md for the
full comparison table across configs 1-4).

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5)
Augment:    rotation +/-90 deg, triangle noise 50% (80-120 triangles), color jitter 100% (0.5-1.5x)
            (horizontal_flip_p and triangle size left at TrainAugmentConfig's
            own defaults -- not overridden for this sweep)

Model UNet(kernel_size=5), warm-started from models/new_unet/unet-final-k5.ckpt.
Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so configs 1-4 can be compared directly.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = 2
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-2"
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
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-90.0, 90.0),
    triangle_noise_p=0.5,
    n_triangles_range=(80, 120),
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
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = MODELS_DIR / "new_unet" / "unet-final-k5.ckpt"

    # strict=False: unet-final-k5.ckpt's saved criterion state (e.g. BCEDiceLoss's
    # pos_weight buffer) does not necessarily match this config's criterion, so
    # that mismatched key is ignored -- the actual UNet weights still load exactly.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
