"""
Online augmentation training -- config 8e (see jobs/online/README.md for the
full comparison table). A follow-up on config 8's checkpoint, not on 8a/8b/8c:
those three (WeightedDiceLoss, landmark_weight 50/75/100) all came back
monotonically *worse* than config 8 on both val_mean_error_px (2.58/2.73/2.90
vs. config 8's 2.17) and val_wrong_spot_count_pct (5.99%/6.79%/8.10% vs.
2.95%) -- worse the higher landmark_weight went. WeightedDiceLoss itself
doesn't beat BCEDiceLoss here, so this returns to BCEDiceLoss (proven
throughout configs 1-8/8d) and tunes its own dice/bce ratio instead.

`dice_weight=0.8` (`bce_weight=0.2`) was planned once before (configs 6c/6d,
on top of config 5b) but never actually launched -- there's no wandb run for
either name, they were dropped once the project pivoted to config 6's
radius-4 mask instead. So this is a genuinely new, untested direction, not a
repeat of an old result. 8e/8f map out two points on the dice-share axis
(0.7, then 0.9 in 8f) the same way 8a/8b/8c mapped out landmark_weight.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3) --
            dice/bce ratio raised from config 8's 0.5/0.5. 8f goes further (0.9/0.1).
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8.
Mask:       circular, square_size=9 (radius 4) -- unchanged from config 6/7/8.

Model UNet(kernel_size=5, sigmoid=False) -- back to the standard pairing
(BCEDiceLoss applies its own sigmoid, see augmented_unet_8a.py's docstring
for why 8a-8c needed sigmoid=True instead). Warm-started from config 8's own
trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-8/last.ckpt).
Its criterion (BCEDiceLoss, pos_weight=50) matches this config's exactly, so
the usual strict=False warm-start is safe.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a-8d.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "8e"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-8e"
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
        square_size=9,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8" / "last.ckpt"
    )

    # strict=False: safe here since config 8's criterion (BCEDiceLoss,
    # pos_weight=50) matches this config's exactly -- see module docstring.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
