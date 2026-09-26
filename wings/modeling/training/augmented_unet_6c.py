"""
Online augmentation training -- config 6c (see jobs/online/README.md for the
full comparison table). Third of four loss-tuning follow-ups on top of
config 5's own result -- see augmented_unet_6a.py for the shared motivation
(config 5's persistent ~3-4% val_wrong_spot_count_pct tail holding back
val_mean_error_px despite a still-improving median).

Config 6c varies a different axis than 6a/6b: the Dice/BCE weighting inside
BCEDiceLoss, rather than pos_weight. dice_weight=0.8/bce_weight=0.2 (vs.
config 5's 0.5/0.5) is a deliberate echo of the pre-online-augmentation
lineage: the actual training script that produced unet-final-k5.ckpt
(unet_kernel_5x5.py, per git history) used this exact 0.8/0.2 split, and
that model's own landmark positional precision (~1.38px, fully converged)
is still the best anyone in this whole online-augmentation series has
matched. This ratio has never been varied in this series until now -- every
config 1-6b kept it at the README's stated 0.5/0.5. Dice rewards overall
region overlap more forgivingly than BCE's harder per-pixel classification
penalty, which may reduce the "completely missed landmark" tail from a
different angle than pos_weight does.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.8, bce_weight=0.2) --
            pos_weight unchanged from config 5; only the dice/bce ratio moves.
Augment:    identical to config 5: Aug B with rotation_p=0.3.
Mask:       circular, square_size=5 (radius 2) -- identical to config 5.

Model UNet(kernel_size=5), warm-started from config 5's own trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-5/last.ckpt).

Unlike 6a/6b/6d, this one does NOT need the weights-only warm-start
workaround: pos_weight=50 here matches config 5's own saved criterion
exactly, so the strict=False reload of that buffer is a no-op (same value
either way), and dice_weight/bce_weight are plain Python floats on
BCEDiceLoss, never registered as buffers/parameters, so they never appear in
the checkpoint's state_dict to begin with -- there is nothing to overwrite.
The standard train(..., checkpoint_path, strict=False) pattern (same as
configs 1-5) is safe as-is.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-5/6a/6b.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "6c"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-6c"
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
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.8, bce_weight=0.2),
}

TRAIN_AUGMENT_CONFIG = TrainAugmentConfig(
    rotation_degrees=(-90.0, 90.0),
    rotation_p=0.3,
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
    model.to(DEVICE)

    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-5" / "last.ckpt"
    )

    # strict=False: kept for consistency with configs 1-5 -- see module docstring
    # for why this one (unlike 6a/6b/6d) has no pos_weight-buffer gotcha to work
    # around.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
