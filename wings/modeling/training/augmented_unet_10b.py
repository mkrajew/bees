"""
Online augmentation training -- config 10b (see jobs/online/README.md for
the full comparison table). A follow-up on config 9a's own checkpoint,
pushing triangle-noise density/size further -- the same gentle-increase
pattern config 7->8 already used (max_size 11->13, n_triangles 40-240->60-300),
continued one more step on top of the current best lineage (8d->9a) instead
of on config 8 directly.

Motivation: notebook 26's DeepWings evaluation has consistently shown dense
fields of small dark triangular debris as the dominant cause of the
remaining gap to the paper's 0.943 -- the model over-detects debris as
landmarks on the noisiest real photos. Config 7's own visual check
(notebooks/24_online_augmentation.ipynb) found triangle_max_size=20 already
started obscuring venation at the existing triangle-count ceiling, so this
keeps the step proportionate rather than jumping to match the worst
offenders directly: n_triangles_range (60,300) -> (80,360), triangle_max_size
13 -> 16.

Loss/mask unchanged from config 9a (BCEDiceLoss(pos_weight=50,
dice_weight=0.7, bce_weight=0.3), circular mask radius 3). Also benefits
from the new smoothed-metric checkpoint selection (see config 10a's
docstring) since it shares wings/modeling/train.py.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "10b"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-10b"
PARAMETERS = {
    "project_name": "wingai-online-augmentation",
    "logger_save_dir": TRAINING_DIR / "online",
    "run_name": f"{run_name}-{run_num}",
    "checkpoint_save_dir": TRAINING_DIR / "lightning-checkpoints" / model_name,
    "checkpoint_filename": model_name
    + "-{epoch:02d}-{val_mean_error_px:.4f}-"
    + f"{run_name}-{run_num}",
    "num_epochs": 150,
    "batch_size": 12,
    "num_workers": 8,
    "early_stop_min_delta": 0.01,
    "early_stop_patience": 60,
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-90.0, 90.0),
    rotation_p=1.0,
    triangle_noise_p=0.8,
    n_triangles_range=(80, 360),
    triangle_max_size=16,
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
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-9a" / "last.ckpt"
    )

    # strict=False is safe: same criterion class and pos_weight as 9a's own
    # checkpoint -- only the augmentation config changes, not anything
    # stored in the criterion's own state.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
