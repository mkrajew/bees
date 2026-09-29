"""
Online augmentation training -- config 9d (see jobs/online/README.md for the
full comparison table). Tests `pos_weight` (50 -> 100) on top of config 8d's
radius-3 checkpoint, `dice_weight` left at the default 0.5 -- a genuinely
untested axis: `pos_weight` 75/100 were planned once as configs 6a/6b (on
top of config 5b) but never actually launched (no wandb run exists for
either name, same as 6c/6d -- dropped once the project pivoted to config 6's
radius-4 mask). See augmented_unet_9a.py's docstring for the shared
rationale behind warm-starting this whole five-config sweep from 8d.

Loss:       BCEDiceLoss(pos_weight=100, dice_weight=0.5, bce_weight=0.5) --
            pos_weight doubled from every prior config in this series (all
            used 50); dice/bce ratio unchanged from config 8/8d, kept as the
            one changed variable.
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8/8d.
Mask:       circular, square_size=7 (radius 3) -- matches config 8d.

Model UNet(kernel_size=5, sigmoid=False), warm-started from config 8d's own
trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-8d/last.ckpt).

Only the UNet weights are loaded from that checkpoint, manually, rather than
going through train()'s usual LitNet.load_from_checkpoint(path, strict=False)
mechanism: that loads the ENTIRE saved LitNet state, including the criterion's
own buffers, and BCEWithLogitsLoss's pos_weight is a persistent buffer under
the *same* key ("criterion.bce.pos_weight") in both config 8d's criterion
(pos_weight=50) and this one's (pos_weight=100) -- strict=False only ignores
missing/unexpected keys, it does not stop a matching key from being
overwritten by the checkpoint's stored value (see augmented_unet_4b.py,
which hit this exact issue first, and 6a/6b/6d for the same gotcha with a
different warm-start source). Loading just the UNet's own weights and
calling train(..., path=None) avoids this entirely.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a-8f/9a-9c.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "9d"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-9d"
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
    "criterion": BCEDiceLoss(pos_weight=100, dice_weight=0.5, bce_weight=0.5),
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

    # Warm-start UNet weights only from config 8d's own trained checkpoint --
    # see module docstring for why the criterion must NOT be loaded from it.
    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8d" / "last.ckpt"
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
