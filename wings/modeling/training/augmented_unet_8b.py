"""
Online augmentation training -- config 8b (see jobs/online/README.md for the
full comparison table). Second of three loss-family follow-ups on top of
config 8's result -- identical to config 8a except `landmark_weight` (50 -> 75),
mapping out the response to this parameter. See augmented_unet_8a.py's
docstring for the full rationale (why WeightedDiceLoss now, why sigmoid=True
is the correct pairing -- there's no proven sigmoid=False+WeightedDiceLoss
historical recipe, that combination in bced_unet.py postdates the runs that
actually got 1.15-1.18px -- and why strict=False is safe here).

Loss:       WeightedDiceLoss(landmark_weight=75, background_weight=1).
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8/8a.
Mask:       circular, square_size=9 (radius 4) -- unchanged from config 6/7/8.

Model UNet(kernel_size=5, sigmoid=True), warm-started from config 8's own
trained checkpoint (not config 8a's -- these three are independent siblings,
all built on config 8 directly, not chained to each other).

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import WeightedDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "8b"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-8b"
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
    "criterion": WeightedDiceLoss(landmark_weight=75, background_weight=1.0),
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
        square_size=9,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=True)
    model.to(DEVICE)

    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8" / "last.ckpt"
    )

    # strict=False: safe here -- WeightedDiceLoss has no criterion buffers at
    # all (see augmented_unet_8a.py's docstring).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
