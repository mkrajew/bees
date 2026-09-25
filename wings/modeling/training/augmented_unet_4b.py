"""
Online augmentation training -- config 4b (see jobs/online/README.md for the
full comparison table). Follow-up to config 4: same loss class and
augmentation, lower pos_weight, warm-started from config 4's OWN trained
checkpoint instead of the original unet-final-k5.ckpt.

Loss:       BCEDiceLoss(pos_weight=25, dice_weight=0.5, bce_weight=0.5) -- half
            of config 4's pos_weight=50.
Augment:    same as config 4 ("Aug B"): rotation +/-90 deg, triangle noise 80%
            (40-240 triangles), color jitter 100% (0.5-1.5x)
            (horizontal_flip_p and triangle size left at TrainAugmentConfig's
            own defaults -- not overridden for this sweep)

Model UNet(kernel_size=5), warm-started from config 4's own trained checkpoint
(models/online/last.ckpt, copied down from the HPC's
wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-4/),
not from unet-final-k5.ckpt like configs 1-4 -- this is a follow-up run on top
of config 4's result, not a fresh comparison point.

Only the UNet weights are loaded from that checkpoint, manually, rather than
going through train()'s usual LitNet.load_from_checkpoint(path, strict=False)
mechanism: that loads the ENTIRE saved LitNet state, including the criterion's
own buffers, and BCEWithLogitsLoss's pos_weight is a persistent buffer under
the *same* key ("criterion.bce.pos_weight") in both this config's criterion
and config 4's -- strict=False only ignores missing/unexpected keys, it does
not stop a matching key from being overwritten by the checkpoint's stored
value. So warm-starting the usual way would silently reload pos_weight=50 from
the checkpoint and discard this file's pos_weight=25 (verified directly:
loading last.ckpt via LitNet.load_from_checkpoint(strict=False) into a freshly
built BCEDiceLoss(pos_weight=25) leaves it holding pos_weight=50 afterwards).
Loading just the UNet's own weights and calling train(..., path=None) avoids
this entirely -- the criterion is then never touched by any checkpoint load.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-4.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "4b"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-4b"
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
    "criterion": BCEDiceLoss(pos_weight=25, dice_weight=0.5, bce_weight=0.5),
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
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)

    # Warm-start UNet weights only from config 4's own trained checkpoint --
    # see module docstring for why the criterion must NOT be loaded from it.
    checkpoint_path = MODELS_DIR / "online" / "last.ckpt"
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
