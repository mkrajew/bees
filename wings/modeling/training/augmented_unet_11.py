"""
Online augmentation training -- config 11 (see jobs/online/README.md for
the full comparison table). Not a new hypothesis: re-runs config 9a's
exact recipe now that wings/dataset.py's coordinate-rounding bug is fixed
(commit 8ef4e61) -- generate_landmark_mask/generate_circular_landmark_mask
used to truncate scaled landmark coordinates toward zero instead of
rounding to nearest, biasing every training target this whole series has
ever learned from by up to 1px, in x and (with opposite sign, because of
the intervening y-flip) in y. That's shared by every config in this file's
family, old and new alike, so nothing about which config to re-run changes
-- only the label generation under it does.

Loss/mask/augmentation: unchanged from config 9a (BCEDiceLoss(
pos_weight=50, dice_weight=0.7, bce_weight=0.3), circular mask radius 3,
Aug B: rotation_p=1.0, n_triangles_range=(60,300), triangle_max_size=13).
num_epochs=100 (9a's own value); early_stop_patience=60 (9a's own 45,
raised to 10a's more patient value -- the only piece of 10a's changes kept
here besides save_top_k).

Deliberately skips config 10a's smoothed-metric checkpoint monitoring
(val_mean_error_px_smooth): in practice it didn't earn its keep, so this
goes back to plain val_mean_error_px, exactly like every config before 10a.
The one thing kept from that change is ModelCheckpoint's save_top_k=5
(train.py, unconditional either way) -- more real candidate epochs on disk
to check by hand, same reasoning, without the smoothing machinery itself.

Warm-start: config 9a's own checkpoint (not 10a's -- 10a's actual run only
reached epoch 8 of its planned 150 before stopping, barely diverged from
9a's own already-converged endpoint).
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "11"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-11"
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
    # Back to plain per-epoch monitoring, matching config 9a -- see this
    # file's module docstring for why config 10a's smoothed variant is
    # deliberately not used here.
    "checkpoint_monitor": "val_mean_error_px",
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
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-9a" / "last.ckpt"
    )

    # strict=False is safe: same criterion class and pos_weight as 9a's own
    # checkpoint (only continuing training, no hyperparameter changed here).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
