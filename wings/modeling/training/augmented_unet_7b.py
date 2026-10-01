"""
Online augmentation training -- config 7b (see jobs/online/README.md for the
full comparison table). A continuation of config 7's own run, not a new
hypothesis: config 7 (bigger triangle noise, triangle_max_size=11) early-
stopped at epoch 36/100 because `val_mean_error_px` plateaued almost
immediately (noisy, flat ~2.25-2.37px from epoch 0 onward) -- but
`val_wrong_spot_count_pct`, the metric config 7's own triangle-size increase
specifically targets, kept trending down over the same epochs (8.08% ->
mostly 7.0-7.4% by the end, best single epoch 6.79% at epoch 30). Both
`EarlyStopping` and `ReduceLROnPlateau` (wings/modeling/litnet.py's
`configure_optimizers`) monitor only `val_mean_error_px`, so training stopped
on a metric that had already saturated, before the metric config 7 actually
cares about had a chance to finish improving. This is already visible in the
final numbers: config 7 (36 epochs) already beats config 6 (full 100 epochs)
on test_wrong_spot_count_pct (7.69% vs. 7.97%) despite the shorter run.

Only `early_stop_patience` changes (25 -> 45): everything else -- loss, mask,
augmentation config -- is identical to config 7. Warm-starting from config
7's own checkpoint (rather than just resuming its Trainer state, which this
project's `train()` doesn't support -- every warm-start here is a fresh
Trainer over pre-trained weights) also resets `configure_optimizers`' AdamW/
ReduceLROnPlateau to lr=1e-5 from scratch: config 7's own LR had likely
already decayed significantly by epoch 36 (ReduceLROnPlateau patience=8,
factor=0.5, keyed to the same plateaued val_mean_error_px), so this
continuation gets a genuinely fresh, larger effective step size to keep
making progress with, not just a longer patience window on an
already-shrunk LR.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- unchanged.
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (40-240 triangles,
            size 2-11 px), color jitter 100% (0.5-1.5x) -- identical to config 7.
Mask:       circular, square_size=9 (radius 4) -- unchanged from config 6/7.

Model UNet(kernel_size=5), warm-started from config 7's own trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-7/last.ckpt).
Its criterion (BCEDiceLoss, pos_weight=50) matches this config's exactly, so
the usual strict=False warm-start is safe.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-7.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "7b"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-7b"
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
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-90.0, 90.0),
    rotation_p=1.0,
    triangle_noise_p=0.8,
    n_triangles_range=(40, 240),
    triangle_max_size=11,
    color_jitter_p=1.0,
    color_jitter_brightness_range=(0.5, 1.5),
    color_jitter_contrast_range=(0.5, 1.5),
)

if __name__ == "__main__":
    train_val_test_datasets = build_mask_datasets(
        countries=COUNTRIES,
        data_folder=PROCESSED_DATA_DIR / "cropped",
        output_size=400,
        square_size=9,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-7" / "last.ckpt"
    )

    # strict=False: safe here since config 7's criterion (BCEDiceLoss,
    # pos_weight=50) matches this config's exactly -- see module docstring.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
