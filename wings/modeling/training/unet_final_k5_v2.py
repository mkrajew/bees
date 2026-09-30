"""
Redo of unet-final-k5.ckpt's own recipe, warm-started from that checkpoint,
now that wings/dataset.py's coordinate-rounding bug is fixed (commit
8ef4e61).

Recipe confirmed via git archaeology in jobs/online/README.md's "Config
8a-8c" section: unet-final-k5.ckpt's own wandb runs (created 2026-05-11 to
05-13, ~1.15-1.18px test error) predate the WeightedDiceLoss line later
added to wings/modeling/training/bced_unet.py by over a month, so the
*actual* live config at training time was BCEDiceLoss(pos_weight=50,
dice_weight=0.8, bce_weight=0.2), sigmoid=False -- not whatever
bced_unet.py/unet_kernel_5x5.py happen to say today (both have been hand-
edited in place for later, unrelated experiments since). Mask square_size=3
is separately confirmed in the same README's "Config 4c" section (checked
against notebooks/04_save_datasets.ipynb and the commit that produced it).

Warm-started from unet-final-k5.ckpt itself (models/new_unet/unet-final-k5.ckpt,
the same path every online-augmentation config already warm-starts from):
cheaper than from scratch, and the same warm-start-through-a-small-fix
philosophy config 11 uses on the online-augmentation side. Loss class and
pos_weight match the checkpoint's own saved criterion exactly -- directly
verified by loading the checkpoint and inspecting its state dict
(`criterion.bce.pos_weight == 50.0`), not just inferred from the
git-archaeology above -- so there's no warm-start gotcha; the standard
strict=True load is safe as-is.

Prerequisite -- rebuild the static mask datasets first:
    uv run wings/modeling/training/rebuild_baseline_k5_datasets.py
NOT wings/dataset.py's own __main__: that one has since moved to
square_size=5 and writes *_sq5.pth-suffixed files for a different, later
need in this project, so it no longer touches the unsuffixed
square_size=3 files this script's load_datasets() actually reads.
rebuild_baseline_k5_datasets.py targets that original combination
specifically. Skipping this step trains against the *old*, still-truncated
mask targets baked into whatever .pth files already happen to be on disk --
the fix lives in how those files get built, not in this script.
MaskRectangleDataset.split()'s default seed=42 keeps the same
train/val/test membership as before, so rebuilding doesn't change which
images land in the held-out test set.
"""

from loguru import logger

from wings.config import DEVICE, TRAINING_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from wings.dataset import load_datasets
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.train import train
from wings.modeling.unet import UNet

run_num = 1
run_name = "weighted-bce-dice-kernel-fix"
model_name = "unet-400-bce-dice-kernel-fix"
PARAMETERS = {
    "project_name": "wingai",
    "logger_save_dir": TRAINING_DIR,
    "run_name": f"{run_name}-{run_num}",
    "checkpoint_save_dir": TRAINING_DIR / "lightning-checkpoints" / model_name,
    "checkpoint_filename": model_name
    + "-{epoch:02d}-{val_mean_error_px:.4f}-"
    + f"{run_name}-{run_num}",
    "num_epochs": 100,
    "batch_size": 12,
    "num_workers": 8,
    "early_stop_min_delta": 0.01,
    "early_stop_patience": 60,
    "criterion": BCEDiceLoss(pos_weight=50, dice_weight=0.8, bce_weight=0.2),
}

if __name__ == "__main__":
    data_dir = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle-cropped"
    train_val_test_datasets = load_datasets(
        [
            data_dir / "train_mask_dataset_ch1_400.pth",
            data_dir / "val_mask_dataset_ch1_400.pth",
            data_dir / "test_mask_dataset_ch1_400.pth",
        ]
    )
    logger.info("Loaded datasets.")

    model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    model.to(DEVICE)

    checkpoint_path = MODELS_DIR / "new_unet" / "unet-final-k5.ckpt"

    train(model, train_val_test_datasets, PARAMETERS, checkpoint_path)
