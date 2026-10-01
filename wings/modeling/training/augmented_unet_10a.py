"""
Online augmentation training -- config 10a (see jobs/online/README.md for
the full comparison table). Continues config 9a's own training with more
patience, relying on the new smoothed-metric checkpoint selection
(LitNet.on_validation_epoch_end / wings/modeling/train.py) instead of a
new hyperparameter change.

Why this exists: across the whole online-augmentation series, the raw
per-epoch val_mean_error_px's "best" epoch landed on epoch 5 for nearly
every config (8, 8d, 8e, 8f, 9a, 9c, 9d, 9e all did), with same-run
epoch-to-epoch noise (std ~0.015-0.04px in a +/-5-epoch window around that
point) *larger* than the actual gap between configs' best values
(~0.001-0.003px) -- the raw signal being monitored for checkpointing/
early-stopping/LR-plateau couldn't reliably tell configs apart. LitNet now
also logs val_mean_error_px_smooth / val_wrong_spot_count_pct_smooth /
val_loss_smooth (5-epoch trailing moving average), and train.py's
EarlyStopping/ModelCheckpoint monitor the smoothed error metric --
ModelCheckpoint also keeps the top 5 by that metric (up from 2), so the
other two smoothed metrics can be checked by hand against several real
candidate epochs instead of only ever having the one the callback picked.

Loss/mask/augmentation: unchanged from config 9a (BCEDiceLoss(pos_weight=50,
dice_weight=0.7, bce_weight=0.3), circular mask radius 3, Aug B). Only the
warm-start (9a's own checkpoint, not 8d's) and early_stop_patience (60, up
from 45) differ, to give the now-smoother signal more room to keep
improving before stopping rather than testing a new hyperparameter.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "10a"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-10a"
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
