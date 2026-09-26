"""
Online augmentation training -- config 6a (see jobs/online/README.md for the
full comparison table). First of four loss-tuning follow-ups on top of
config 5b's result (circular mask + rotation_p=0.8, see
augmented_unet_5b.py) -- retargeted from config 5's own checkpoint after
config 5's rotation_p=0.3 turned out to cost real rotation robustness
(a miscommunication: 0.3 meant only 30% of training was rotated, not 70% as
intended). Config 5b keeps config 5's other fixes (circular mask,
square_size=5) while correcting rotation_p to 0.8. On config 5 itself,
val_mean_error_px still plateaued around ~2.1-2.2px, held back by a
persistent ~3-4% val_wrong_spot_count_pct tail (median error kept slowly
improving even as mean stalled, suggesting a genuine subset of
hard-to-detect landmarks rather than general stagnation) -- these four
configs assume the same pattern shows up on config 5b and tune the loss
function to address it there instead.

Config 6a raises pos_weight (50 -> 75) to push harder on recall specifically,
directly targeting that "completely missed landmark" tail -- at some cost to
precision, which handle_coordinates' extra-point handling already copes with
reasonably well. Config 6b tries an even higher value (100) to map out the
shape of this response; 6c/6d vary the dice/bce ratio instead (see their own
docstrings).

Loss:       BCEDiceLoss(pos_weight=75, dice_weight=0.5, bce_weight=0.5) -- pos_weight
            raised from config 5b's 50.
Augment:    identical to config 5b: Aug B with rotation_p=0.8.
Mask:       circular, square_size=5 (radius 2) -- identical to config 5b.

Model UNet(kernel_size=5), warm-started from config 5b's own trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-5b/last.ckpt).

Only the UNet weights are loaded from that checkpoint, manually, rather than
going through train()'s usual LitNet.load_from_checkpoint(path, strict=False)
mechanism: that loads the ENTIRE saved LitNet state, including the criterion's
own buffers, and BCEWithLogitsLoss's pos_weight is a persistent buffer under
the *same* key ("criterion.bce.pos_weight") in both config 5b's criterion and
this one's -- strict=False only ignores missing/unexpected keys, it does not
stop a matching key from being overwritten by the checkpoint's stored value
(see augmented_unet_4b.py, which hit this exact issue first). Loading just the
UNet's own weights and calling train(..., path=None) avoids this entirely.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-5/5b.
"""

import torch
from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "6a"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-6a"
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
    "criterion": BCEDiceLoss(pos_weight=75, dice_weight=0.5, bce_weight=0.5),
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

    # Warm-start UNet weights only from config 5b's own trained checkpoint --
    # see module docstring for why the criterion must NOT be loaded from it.
    checkpoint_path = (
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-5b" / "last.ckpt"
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
