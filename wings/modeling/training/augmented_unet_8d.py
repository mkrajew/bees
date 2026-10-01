"""
Online augmentation training -- config 8d (see jobs/online/README.md for the
full comparison table). A follow-up on config 8's checkpoint, not a new
hypothesis: identical to config 8 in every respect except mask radius
(4 -> 3, `square_size` 9 -> 7).

Motivated by the DeepWings paper's own ablation (Table 2): within their
tested range, radius 3 already reached 88.2% exact-19-landmark accuracy,
radius 4 reached 91.8% -- a real but not huge gap, and their own note that
radius >4 starts merging blobs together applies at some degree to radius 4
itself once masks get dense enough. Config 8's own remaining DeepWings
worst-offenders (jobs/online/README.md's "Follow-up: config 8" section)
include exactly this failure mode -- landmark blobs merging or overlapping
under the densest real debris, needing `mask_to_coords`'s watershed splitting
to recover. A smaller radius reduces how often adjacent landmarks' circles
can touch in the first place (radius 3 leaves roughly twice the gap between
same-spaced landmark pairs that radius 4 does), at the cost of some of the
rotation-robustness/detection-reliability benefit radius 4 brought over the
project's earlier radius-2 configs (5/5b) -- this config tests where radius 3
actually lands on that tradeoff now that everything else (full rotation_p,
denser triangle noise, tuned loss) is already in place, rather than
re-deriving the radius choice in isolation the way config 6 originally did.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- unchanged.
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (60-300 triangles, size
            2-13 px) -- identical to config 8.
Mask:       circular, square_size=7 (radius 3) -- down from config 8's radius 4.

Model UNet(kernel_size=5, sigmoid=False), warm-started from config 8's own
trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-8/last.ckpt).
Mask radius only affects target *data* generation (`generate_circular_landmark_mask`),
not the model architecture or the criterion's own state -- unlike a pos_weight
change (see augmented_unet_4b.py's docstring), there is no buffer/key overlap
to worry about here at all, so the standard strict=False warm-start carries no
gotcha in either direction.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-8/8a-8c.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = "8d"
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-8d"
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
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5),
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
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-8" / "last.ckpt"
    )

    # strict=False: safe here since config 8's criterion (BCEDiceLoss,
    # pos_weight=50) matches this config's exactly -- see module docstring.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
