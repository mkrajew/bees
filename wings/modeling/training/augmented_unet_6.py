"""
Online augmentation training -- config 6 (see jobs/online/README.md for the
full comparison table). A fresh primary config informed by reading the actual
DeepWings paper (Rodrigues et al. 2022, BDCC 6(3):70 -- the source of the
0.943 positional-precision benchmark and the mean_coords/19-landmark scheme
this whole project is built on), not a follow-up on configs 4/5's own
checkpoints.

Two changes from everything tried in this series so far, both aimed at the
same underlying problem investigated across configs 4c/4d/5/5b (val_mean_error_px
drifting worse under training, and -- once mitigated via rotation_p -- a
separate GPA-vs-nearest-neighbor gap re-appearing on config 5b specifically
at large rotation angles):

1. Mask radius 4 (square_size=9 -> generate_circular_landmark_mask's
   radius = square_size // 2 = 4), not smaller. Every mask-size experiment
   in this series so far (4c's square_size=3, 4d/5/5b/6a-6d's circular
   radius 2) moved toward a *smaller* target, on the theory that smaller
   forces sharper, more precisely centered blobs. The paper's own published
   ablation (their Table 2) found the opposite within their tested range:
   radius <3 was "virtually ignored by the network", and accuracy kept
   improving through radius 3 -> 4 (88.2% -> 91.8% exact-19-landmark
   detection), with radius >4 the point where blobs start merging into each
   other. Their radius-4 circle has ~50 px of area -- larger than even our
   original square_size=5 (25 px), and ~4x our circular radius-2 (13 px).
   Re-checking config 5b's checkpoint (radius 2) against notebook 25 showed
   a real detection-quality drop under rotation (not just an ordering issue
   -- visible directly in the nearest-neighbor-matched distances, which
   don't depend on landmark ordering at all), consistent with a
   too-small target being less robust to rotation's resampling blur, the
   same mechanism that made config 4c's square_size=3 fail rotation
   robustness earlier. This config tests whether matching the paper's own
   validated radius fixes that, rather than continuing to shrink the target.
2. rotation_p=1.0 (back to unconditional rotation, like every config before
   5/5b/6a-6d): configs 5/5b lowered rotation_p specifically because full
   rotation training measurably dragged down val_mean_error_px over epochs
   (see config 5's README section) -- but that diagnosis was made entirely
   on square_size=5/circular-radius-2 checkpoints, i.e. potentially
   confounded by the same too-small-target issue this config also
   addresses. Also, wings/gpa.py now has a second, complementary fix
   (pca_prealign, added alongside this config -- see its own docstring)
   for the GPA-ordering-specific gap found on config 5b, which is
   independent of how much the model itself was trained on rotated input.
   With both changes together, full rotation training may no longer need
   to be rationed down to protect val_mean_error_px.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) --
            unchanged; also matches the paper's own reported weighting
            (background weight 1, landmark class weight +50).
Augment:    Aug B, rotation_p=1.0 (rotation +/-90 deg, always applied):
            triangle noise 80% (40-240 triangles), color jitter 100%
            (0.5-1.5x) (horizontal_flip_p and triangle size left at
            TrainAugmentConfig's own defaults -- not overridden here).
Mask:       circular, square_size=9 (radius 4).

Model UNet(kernel_size=5), warm-started from models/new_unet/unet-final-k5.ckpt
-- a fresh start (not any config 4/5-family checkpoint): this changes the
mask target size, which the model has never been trained against at any
point in this series, so there is no more-relevant checkpoint to build on
than the original converged baseline. Its own criterion (BCEDiceLoss,
pos_weight=50) matches this config's exactly, so the usual strict=False
warm-start is safe.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-5/5b/6a-6d.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = 6
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-6"
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
    rotation_p=1.0,
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
        square_size=9,
        train_augment_cfg=TRAIN_AUGMENT_CONFIG,
        mask_fn=generate_circular_landmark_mask,
    )
    logger.info("Built datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = MODELS_DIR / "new_unet" / "unet-final-k5.ckpt"

    # strict=False: kept for consistency with configs 1-4/4c/4d/5/5b --
    # unet-final-k5.ckpt's criterion here actually matches exactly
    # (BCEDiceLoss, pos_weight=50), so this is just future-proofing against
    # an unrelated saved key, not masking a real mismatch.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
