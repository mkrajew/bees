"""
Online augmentation training -- config 9c (see jobs/online/README.md for the
full comparison table). Pushes the dice/bce ratio to its extreme (1.0/0.0,
pure Dice, no BCE term at all) on top of config 8d's radius-3 checkpoint,
since 8e -> 8f (dice_weight 0.7 -> 0.9) only gained +0.0004 positional
precision -- a shrinking return -- so this checks whether the trend
continues, plateaus, or reverses at the limit, rather than assuming 0.9 was
close enough to the top. See augmented_unet_9a.py's docstring for the
shared rationale (why these five configs warm-start from 8d specifically).

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=1.0, bce_weight=0.0) --
            `pos_weight` has no effect here: with bce_weight=0 the BCE term
            (the only place pos_weight is used) is multiplied by zero and
            drops out of the loss entirely. Kept in the constructor call
            anyway for consistency with every other config in this series,
            not because it does anything in this specific one.
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8/8d.
Mask:       circular, square_size=7 (radius 3) -- matches config 8d.

Model UNet(kernel_size=5, sigmoid=False), warm-started from config 8d's own
trained checkpoint. strict=False is safe here for the same reason as 9a/9b:
pos_weight is unchanged from 8d's own criterion (even though it's now inert),
so there's no buffer-value mismatch to worry about.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a-8f/9a/9b.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "9c"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-9c"
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
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=1.0, bce_weight=0.0),
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
    model.to(DEVICE)

    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8d" / "last.ckpt"
    )

    # strict=False: safe here -- see augmented_unet_9a.py's docstring.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
