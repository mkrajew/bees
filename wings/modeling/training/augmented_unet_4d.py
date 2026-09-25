"""
Online augmentation training -- config 4d (see jobs/online/README.md for the
full comparison table). Tests a second, complementary hypothesis alongside
config 4c about why config 4/4b's val_mean_error_px is much worse than
unet-final-k5.ckpt's own despite higher val_dice: config 4c changes the mask
target's *size* (square_size 5 -> 3); config 4d instead changes its *shape*
(square -> circle), holding size at 5 (same square_size value, so the
circle's radius is square_size // 2 = 2 -- see generate_circular_landmark_mask
in wings/dataset.py). A circle inscribed in a 5x5 square has roughly half its
area (measured: 13 vs 25 px/landmark) and, unlike a square, has no corners --
the pixels farthest from the true landmark center, and the ones a model can
"cover" most cheaply for Dice credit without actually sharpening its
localization. If square_4's diffuse/imprecise blobs are partly explained by
the model exploiting those corners, a circular target should push it toward
smaller, better-centered blobs somewhat like config 4c's smaller square does,
without changing the target's nominal "size" knob at all -- config 4c and 4d
together tell apart whether it's the *area*, the *corners*, or both, that
matter.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- same as config 4/2/4c.
Augment:    same as config 4 ("Aug B"): rotation +/-90 deg, triangle noise 80%
            (40-240 triangles), color jitter 100% (0.5-1.5x)
            (horizontal_flip_p and triangle size left at TrainAugmentConfig's
            own defaults -- not overridden for this sweep)
Mask:       square_size=5 (unchanged from config 4), but circular
            (generate_circular_landmark_mask) instead of square.

Model UNet(kernel_size=5), warm-started from models/new_unet/unet-final-k5.ckpt
-- same source as configs 1-4/4c (not config 4b's own checkpoint): its own
criterion (BCEDiceLoss, pos_weight=50) matches this config's exactly, so the
usual strict=False warm-start is safe (see augmented_unet_4c.py / 4b.py for
why that's not always true across these configs).

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-4/4b/4c.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "4d"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-4d"
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

    checkpoint_path = MODELS_DIR / "new_unet" / "unet-final-k5.ckpt"

    # strict=False: kept for consistency with configs 1-4/4c -- unet-final-k5.ckpt's
    # criterion here actually matches exactly (BCEDiceLoss, pos_weight=50), so this
    # is just future-proofing against an unrelated saved key, not masking a real
    # mismatch (contrast with augmented_unet_4b.py's pos_weight=25 case).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
