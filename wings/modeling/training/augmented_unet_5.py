"""
Online augmentation training -- config 5 (see jobs/online/README.md for the
full comparison table). A fresh primary config, not a config-4 follow-up:
combines the two best-evidenced fixes found investigating why config 4/4b/4c/4d
all show val_mean_error_px *drifting worse* over training (not just an
initial dip that hasn't recovered -- confirmed from the full per-epoch wandb
curves, all four monotonically worsen after an early peak, with no sign of
turning around):

1. Circular masks (like config 4d) instead of square_size=5 -- a circle has
   no corners for the model to "cover" cheaply for Dice credit without
   actually sharpening its localization. square_size=3 (config 4c) was tried
   first and initially looked more promising (a fully-converged track record
   from unet-final-k5.ckpt itself, plus a better best-epoch median on clean
   validation images) -- but re-checking config 4c's checkpoint against
   notebook 25's rotation-robustness sweep showed it performing much worse
   than square_size=5 specifically on rotated input: a 3x3 target is simply a
   harder detection problem once the input is degraded by rotation's
   resampling blur, with much less margin for error than a 5x5 one. A circle
   of the same radius as config 4/4b/4d's square (square_size=5 -> radius 2)
   keeps that same effective "reach"/robustness margin -- it's not a *smaller*
   target the way square_size=3 is, just one without corners -- so it
   shouldn't cost the rotation robustness this whole online-augmentation
   effort exists to build in the first place.
2. rotation_p=0.3 (new) -- unlike flip/noise/jitter, TrainAugmentConfig's
   rotation previously had no apply-probability: every single training
   access got rotated by a random angle in (-90, 90), while val/test are
   never rotated at all. That means training supplied essentially zero
   examples resembling what val is scored on, and continuing to train only
   pulls the model further toward "good on rotated/blurred" at the expense
   of "good on the crisp, unrotated images validation actually uses" -- a
   plausible mechanism for the observed monotonic drift, unrelated to
   pos_weight or mask shape/size, which is why 4b/4c/4d didn't fix it either.
   rotation_p=0.3 keeps most of the rotation-robustness training signal
   (notebook 25 confirmed that robustness is real and worth keeping) while
   giving 70% of training accesses a chance to look like what val measures.

Kept as config 4's own proven value otherwise: pos_weight=50 (config 4b's
pos_weight=25 showed *worse* numbers than config 4 at every epoch measured,
not better, so there's no evidence for moving off 50).

Loss:       BCEDiceLoss(pos_weight=50, dice_weight=0.5, bce_weight=0.5) -- same as config 4/2/4c/4d.
Augment:    same as config 4 ("Aug B") except rotation_p=0.3 (new): rotation
            +/-90 deg applied 30% of the time, triangle noise 80% (40-240
            triangles), color jitter 100% (0.5-1.5x)
            (horizontal_flip_p and triangle size left at TrainAugmentConfig's
            own defaults -- not overridden for this sweep)
Mask:       circular, square_size=5 (radius 2) -- like config 4d, unlike
            config 4/4b's square or 4c's smaller square_size=3.

Model UNet(kernel_size=5), warm-started from models/new_unet/unet-final-k5.ckpt
-- same source as configs 1-4/4c/4d (not 4b's own checkpoint): its own
criterion (BCEDiceLoss, pos_weight=50) matches this config's exactly, so the
usual strict=False warm-start is safe.

Reads from the plain YOLO-cropped data/processed/cropped/ folder -- augmentation
happens online, freshly on every training access (see wings/transforms.py).
Logs to the shared "wingai-online-augmentation" wandb project (not the other
training scripts' "wingai" project) so this can be compared directly against
configs 1-4/4b/4c/4d.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet
from wings.transforms import TrainAugmentConfig

run_num = 5
run_name = "online-augmentation-k5"
model_name = "unet-400-online-augmentation-k5-5"
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
    rotation_p=0.3,
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

    # strict=False: kept for consistency with configs 1-4/4c/4d -- unet-final-k5.ckpt's
    # criterion here actually matches exactly (BCEDiceLoss, pos_weight=50), so this
    # is just future-proofing against an unrelated saved key, not masking a real
    # mismatch (contrast with augmented_unet_4b.py's pos_weight=25 case).
    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path, strict=False)
