"""
Online augmentation training -- config 9a (see jobs/online/README.md for the
full comparison table). First of a five-config sweep on top of config 8d's
checkpoint, not config 8's: config 8d (mask radius 4 -> 3) and configs
8e/8f (BCEDiceLoss dice_weight 0.5 -> 0.7/0.9) each independently beat both
config 8 and the DeepWings paper's 0.943 on notebook 26's positional
precision (8: 0.9463, 8d: 0.9502, 8e: 0.9475, 8f: 0.9479) -- radius 3 gave
the single largest gain. This sweep checks whether radius and dice_weight
*stack* rather than being redundant, by warm-starting from 8d (which already
has the radius-3 adaptation) and changing dice_weight on top, instead of
re-deriving both changes from config 8 at once.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.7, bce_weight=0.3) --
            same dice_weight as config 8e, now combined with 8d's radius 3.
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8/8d.
Mask:       circular, square_size=7 (radius 3) -- matches config 8d, down
            from config 8/8e/8f's radius 4.

Model UNet(kernel_size=5, sigmoid=False), warm-started from config 8d's own
trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-8d/last.ckpt).
Its criterion (BCEDiceLoss, pos_weight=50) matches this config's pos_weight
exactly (only dice_weight/bce_weight differ, and those aren't stored as
criterion buffers), so the usual strict=False warm-start is safe -- unlike
9d/9e below, which also change pos_weight.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a-8f.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "9a"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-9a"
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
        square_size=7,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8d" / "last.ckpt"
    )

    # strict=False: safe here since config 8d's pos_weight (50) matches this
    # config's exactly -- only dice_weight/bce_weight differ, and those are
    # plain Python floats, not criterion buffers. See module docstring.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
