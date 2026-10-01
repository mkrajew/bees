"""
Online augmentation training -- config 7 (see jobs/online/README.md for the
full comparison table). A follow-up on config 6's checkpoint, not a new
hypothesis: config 6 (mask radius 4 + full rotation_p=1.0) essentially solved
this project's rotation-robustness problem (notebook 25: positional precision
~0.99, flat across the entire 0-90 degree sweep) but its notebook 26 result
against the independent DeepWings test set showed mean positional precision
of 0.9127 overall (paper: 0.943) vs. 0.9525 restricted to the ~78% of wings
with a plausible (16-22) predicted point count -- i.e. the gap to the paper's
number is coming almost entirely from outright detection failures on the
noisier ~22% of DeepWings photos, not from positional/shape accuracy where
detection succeeds at all.

This config targets that gap directly: `TrainAugmentConfig`'s triangle noise
has always varied *count* (`n_triangles_range`) but never *size*
(`triangle_min_size`/`triangle_max_size` have been left at their defaults,
2/6, in every config built so far -- rendering specks only ~2-9px on a
400x400 crop, comparable to or smaller than a single landmark). DeepWings'
own photos are frequently described as considerably dirtier/more damaged than
our own collection's, and the paper's own augmentation includes analogous
"dust" noise -- a wider, larger size range should better simulate that.

`triangle_max_size` moves from 6 to 11 (not the initially proposed 20):
notebook 24's "Config 7 -- larger triangle noise" section renders the
augmentation at fixed count/size combinations to check this visually before
training on it, since `TriangleNoise` draws each triangle's size *uniformly*
between `triangle_min_size` and `triangle_max_size` -- raising the max
shifts every triangle's expected size up, not just a rare worst case (half
of all draws land above the new range's midpoint on every single image).
At max_size=20 combined with the existing 240-triangle ceiling, wing venation
was substantially obscured in a large fraction of the crop; max_size=11 at
that same 240-triangle ceiling stayed legible while still meaningfully
bigger than config 6's specks. `n_triangles_range` is deliberately left
unchanged (40-240): the same notebook check showed count and size
contribute comparable amounts of total occlusion (a maxed-count/medium-size
draw looked about as cluttered as a medium-count/maxed-size draw), so with
size itself now moderated, there's no longer a clear case for also cutting
how many triangles appear.

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- unchanged.
Augment:    Aug B, rotation_p=1.0, triangle noise 80% (40-240 triangles,
            size 2-11 px, up from 2-6), color jitter 100% (0.5-1.5x).
Mask:       circular, square_size=9 (radius 4) -- unchanged from config 6.

Model UNet(kernel_size=5), warm-started from config 6's own trained checkpoint
(wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-6/last.ckpt).
Its criterion (BCEDiceLoss, pos_weight=50) matches this config's exactly, so
the usual strict=False warm-start is safe -- unlike configs 4b/6a/6b/6d, which
changed pos_weight relative to their own warm-start source and needed the
manual weights-only loading workaround instead.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-6/6a-6d.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = 7
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-7"
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
    triangle_max_size=11,
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
        TRAINING_DIR / "lightning-checkpoints" / "unet-400-online-augmentation-k5-6" / "last.ckpt"
    )

    # strict=False: safe here since config 6's criterion (BCEDiceLoss,
    # pos_weight=50) matches this config's exactly -- see module docstring.
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
