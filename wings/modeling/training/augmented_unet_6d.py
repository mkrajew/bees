"""
Online augmentation training -- config 6d (see jobs/online/README.md for the
full comparison table). Fourth of four loss-tuning follow-ups on top of
config 5's own result -- see augmented_unet_6a.py for the shared motivation
(config 5's persistent ~3-4% val_wrong_spot_count_pct tail holding back
val_mean_error_px despite a still-improving median).

Config 6d combines 6a's pos_weight change and 6c's dice/bce ratio change
into one run, rather than waiting to see 6a/6c's individual results before
trying the combination: all four 6-series configs run in parallel on the
cluster regardless, so testing whether the two independently-motivated
changes compound favorably now saves a whole second round-trip later if
both turn out to help individually. Uses 6a's milder pos_weight=75 (not 6b's
100) paired with 6c's dice/bce ratio, to avoid stacking two aggressive
changes in the same run.

Loss:       BCEDiceLoss(pos_weight=75, dice_weight=0.8, bce_weight=0.2) --
            6a's pos_weight + 6c's dice/bce ratio, both moved from config 5's
            50/0.5/0.5 at once.
Augment:    identical to config 5: Aug B with rotation_p=0.3.
Mask:       circular, square_size=5 (radius 2) -- identical to config 5.

Model UNet(kernel_size=5), warm-started from config 5's own trained checkpoint
-- same weights-only loading as config 6a/6b, for the same reason (pos_weight
differs from config 5's own 50, so the usual strict=False warm-start would
silently reload 50 and discard this file's 75 -- see augmented_unet_4b.py /
6a.py for the full explanation). dice_weight/bce_weight don't need this
workaround on their own (see augmented_unet_6c.py), but pos_weight still does,
so the whole checkpoint load uses the weights-only path here.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-5/6a/6b/6c.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "6d"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-6d"
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
    "criterion": BCEDiceLoss(pos_weight=75, dice_weight=0.8, bce_weight=0.2),
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

    # Warm-start UNet weights only from config 5's own trained checkpoint --
    # see module docstring for why the criterion must NOT be loaded from it.
    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-5" / "last.ckpt"
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
