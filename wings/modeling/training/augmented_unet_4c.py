"""
Online augmentation training -- config 4c (see jobs/online/README.md for the
full comparison table). Tests a specific hypothesis about why config 4/4b's
val_mean_error_px (~2.3-2.4px) is much worse than the pre-online-augmentation
unet-final-k5.ckpt's own (~1.15-1.38px, re-measured with today's evaluation
code) despite configs 1-4/4b having *higher* val_dice: unet-final-k5.ckpt was
originally trained with mask square_size=3, not 5 -- every online-augmentation
config so far (1-4, 4b) uses square_size=5 without that ever having been the
variable under test. A larger mask target lets the model get away with
larger/more diffuse predicted blobs (looks better on Dice, which rewards
covering more of a larger target region) that are both less precisely
centered (worse landmark position accuracy) and more likely to bleed into a
neighboring landmark's blob (worse point-count reliability, i.e. higher
val_wrong_spot_count_pct) than the smaller, sharper blobs a square_size=3
target forces the model to produce.

Config 4c changes ONLY square_size (5 -> 3) relative to config 4 -- same
loss, same pos_weight, same augmentation, same warm-start source -- to
isolate that one variable instead of conflating it with config 4b's
pos_weight change.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- same as config 4/2.
Augment:    same as config 4 ("Aug B"): rotation +/-90 deg, triangle noise 80%
            (40-240 triangles), color jitter 100% (0.5-1.5x)
            (horizontal_flip_p and triangle size left at TrainAugmentConfig's
            own defaults -- not overridden for this sweep)
Mask:       square_size=3 (vs. 5 for every other online-augmentation config;
            matches unet-final-k5.ckpt's own original training).

Model UNet(kernel_size=5), warm-started from models/new_unet/unet-final-k5.ckpt
-- same source as configs 1-4 (not config 4b's own checkpoint): unet-final-k5.ckpt
was itself trained on square_size=3 masks, making it the more natural/fair
starting point for testing square_size=3 again, and its own criterion
(BCEDiceLoss, pos_weight=50 -- confirmed by inspecting its saved state dict)
matches this config's exactly, so the usual strict=False warm-start is safe
here (unlike config 4b's pos_weight change, there's no matching-key/
mismatched-value gotcha to work around -- see augmented_unet_4b.py).

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-4/4b.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "4c"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-4c"
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
        square_size=3,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = MODELS_DIR / "new_unet" / "unet-final-k5.ckpt"

    # strict=False: kept for consistency with configs 1-4 -- unet-final-k5.ckpt's
    # criterion here actually matches exactly (BCEDiceLoss, pos_weight=50), so this
    # is just future-proofing against an unrelated saved key, not masking a real
    # mismatch (contrast with augmented_unet_4b.py's pos_weight=25 case).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
