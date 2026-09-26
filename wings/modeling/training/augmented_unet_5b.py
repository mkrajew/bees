"""
Online augmentation training -- config 5b (see jobs/online/README.md for the
full comparison table). Corrects a miscommunication in config 5's own
`rotation_p` value, not a new hypothesis: config 5 was meant to keep *most*
rotation-robustness training while giving the model some exposure to
unrotated images, but `rotation_p=0.3` actually means only 30% of training
accesses get rotated (70% don't) -- the opposite emphasis from what was
intended (~70% rotated was the goal). Confirmed as the cause once config 5
finished: it plateaus at a much better val_wrong_spot_count_pct (~3% vs.
config 4's ~9%+) and its val_mean_error_px monotonic-drift problem is
genuinely fixed (plateaus/oscillates rather than climbing to the end) --
but re-running its checkpoint through notebook 25's rotation-robustness
sweep showed it performing *worse* than config 4b specifically on rotated
input, i.e. it lost real rotation robustness by seeing rotation so rarely
during training.

Config 5b changes only `rotation_p` (0.3 -> 0.8, bumped from an initial 0.7
correction to lean further toward preserving rotation robustness) relative
to config 5 -- same circular mask, same square_size=5, same loss, same
warm-start source (unet-final-k5.ckpt, not config 5's own checkpoint, since
that checkpoint's weights are downstream of the too-low rotation_p this file
is specifically correcting). Configs 6a-6d (loss-tuning follow-ups, see
their own docstrings) are retargeted to warm-start from this config's
checkpoint instead of config 5's once this run finishes.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- same as config 5.
Augment:    same as config 5 ("Aug B") except rotation_p=0.8 (corrected from
            5's 0.3): rotation +/-90 deg applied 80% of the time, triangle
            noise 80% (40-240 triangles), color jitter 100% (0.5-1.5x)
            (horizontal_flip_p and triangle size left at TrainAugmentConfig's
            own defaults -- not overridden for this sweep)
Mask:       circular, square_size=5 (radius 2) -- identical to config 5.

Model UNet(kernel_size=5), warm-started from models/new_unet/unet-final-k5.ckpt
-- same source as config 5 (not config 5's own checkpoint -- see above); its
own criterion (BCEDiceLoss, pos_weight=50) matches this config's exactly, so
the usual strict=False warm-start is safe.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-5.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "5b"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-5b"
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
    rotation_p=0.8,
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

    # strict=False: kept for consistency with configs 1-5 -- unet-final-k5.ckpt's
    # criterion here actually matches exactly (BCEDiceLoss, pos_weight=50), so this
    # is just future-proofing against an unrelated saved key, not masking a real
    # mismatch (contrast with augmented_unet_4b.py's pos_weight=25 case).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
