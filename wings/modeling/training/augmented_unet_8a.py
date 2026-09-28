"""
Online augmentation training -- config 8a (see jobs/online/README.md for the
full comparison table). First of three loss-family follow-ups on top of
config 8's result, not a mask/augmentation change: tests `WeightedDiceLoss`
(landmark_weight=50, background_weight=1, matching its own class defaults),
a loss class never tried anywhere in this online-augmentation series (which
has used `BCEDiceLoss` exclusively, configs 1-8).

Initially thought this matched the original pre-online-augmentation baseline
(`wings/modeling/training/bced_unet.py`, ~1.15-1.18px test error) exactly,
`sigmoid=False` included -- but checking that file's git history
(`git log --all -- wings/modeling/training/bced_unet.py`) shows its
`WeightedDiceLoss(landmark_weight=100)` line was introduced in a commit dated
2026-06-27, over a month *after* the wandb runs that actually achieved
1.15-1.18px (created 2026-05-11 to 05-13). Those runs ran the version of the
file live at the time, `BCEDiceLoss(pos_weight=50, dice_weight=0.8,
bce_weight=0.2)` -- which correctly pairs with `sigmoid=False`
(`BCEDiceLoss.forward(logits, targets)` applies sigmoid itself, via
`nn.BCEWithLogitsLoss` for its BCE term and an explicit `torch.sigmoid(logits)`
for its Dice term, so it needs raw logits in). There is no evidence the
`WeightedDiceLoss` line was ever actually trained to completion. So there is
no proven historical `WeightedDiceLoss` + `sigmoid=False` recipe to preserve
faithfully -- `WeightedDiceLoss.forward(y_pred, y_true)` has no sigmoid
anywhere in its own code and uses `y_pred` directly, so it mathematically
expects actual [0, 1] probabilities, and this config uses the correct
pairing, `sigmoid=True`.

Loss:       WeightedDiceLoss(landmark_weight=50, background_weight=1) --
            replaces BCEDiceLoss entirely. 8b/8c vary landmark_weight (75/100).
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8.
Mask:       circular, square_size=9 (radius 4) -- unchanged from config 6/7/8.

Model UNet(kernel_size=5, sigmoid=True) -- see above for why this differs
from every other config in this series (1-8, all sigmoid=False for
BCEDiceLoss). Warm-started from config 8's own trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-8/last.ckpt) --
config 8's own UNet weights transfer regardless of the loss/sigmoid change
(only the model's final activation changes, not its learned weights).
WeightedDiceLoss registers no torch buffers (landmark_weight/background_weight
are plain Python floats, not nn.Module buffers) -- unlike BCEDiceLoss's
pos_weight (see augmented_unet_4b.py's docstring for that gotcha), so there is
no overlapping criterion state to accidentally inherit from config 8's
checkpoint; the standard strict=False warm-start is safe here for a different
reason than usual (no shared keys at all, not matching values at a shared key).

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import WeightedDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "8a"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-8a"
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
    "criterion": WeightedDiceLoss(landmark_weight=50, background_weight=1.0),
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
    # all, so there's nothing for config 8's checkpoint to mismatch or
    # accidentally overwrite (see module docstring).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
