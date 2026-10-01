"""Scores a trained checkpoint against DeepWings' own published test set
(test/ + test_masks/, from the Beeapp-landmark-detection repo), reproducing
notebook 26's evaluation loop as a reusable function. Extracted so training
runs (wings/modeling/train.py) can get this genuinely out-of-domain signal
automatically for every saved checkpoint, not only via a manual notebook run
afterward -- this is the one evaluation set neither training nor validation
ever touches, so it isn't subject to the same epoch-to-epoch noise that
motivated smoothing val_mean_error_px (see litnet.py).
"""
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from wings.utils import load_image
from wings.visualizing.image_preprocess import (
    unet_fit_rectangle_preprocess,
    final_coords,
    mask_to_coords,
)
from wings.gpa import FULL_ROTATION_MULTISTART_ANGLES
from wings.metrics import gpa_ordered_metrics_both
from wings.modeling.unet import UNet
from wings.modeling.litnet import LitNet
from wings.modeling.loss import BCEDiceLoss

DEFAULT_LOCAL_DEEPWINGS_DIR = r"C:\Users\X\projects\deepwings"

# Overridable via env var: the hardcoded default only exists on this local
# Windows machine, but wings/modeling/train.py now calls this from SLURM
# training jobs too, which run on the (Linux) HPC cluster -- set
# DEEPWINGS_DIR there (e.g. in the job script, before `uv run ...`) to
# wherever test/ + test_masks/ were copied to on that machine.
DEEPWINGS_DIR = Path(os.environ.get("DEEPWINGS_DIR", DEFAULT_LOCAL_DEEPWINGS_DIR))
TEST_DIR = DEEPWINGS_DIR / "test"
MASK_DIR = DEEPWINGS_DIR / "test_masks"
OUTPUT_SIZE = 400


def _discover_pairs() -> list[tuple[Path, Path]]:
    exts = (".jpg", ".jpeg")
    all_files = list(TEST_DIR.iterdir())
    test_files = sorted(p for p in all_files if p.suffix.lower() in exts)
    mask_files = {p.name: p for p in MASK_DIR.iterdir() if p.suffix.lower() in exts}
    return [(p, mask_files[p.name]) for p in test_files if p.name in mask_files]


def evaluate_checkpoint_on_deepwings(
    checkpoint_path,
    mean_coords,
    sigmoid: bool = False,
    kernel_size: int = 5,
    max_samples: int | None = None,
    device=None,
) -> dict:
    """Loads `checkpoint_path` and scores it against DeepWings' own published
    test set, exactly like notebook 26. `sigmoid` must match how this
    checkpoint's UNet was built (False for the BCEDiceLoss configs this
    whole online-augmentation series uses; see jobs/online/README.md).

    Returns a dict with 'precision_mean'/'precision_median'
    (wing_positional_precision, DeepWings' own "metric 2", directly
    comparable to their paper's 0.943 -- unfiltered over all evaluated
    pairs, per notebook 26's own reasoning: restricting to plausible point
    counts would silently reward outright detection failures instead of
    counting against them), 'reliable_pct' (share of images where the model
    found a plausible 16-22 raw points, diagnostic only), 'n_samples', and
    the raw 'precisions'/'n_pred' lists (same order as _discover_pairs())
    so a caller can fold in more points -- e.g. evaluate_bmp_stragglers_on_
    deepwings()'s 21 .bmp images -- and recompute an exact combined mean/
    median instead of approximating from the rounded summary stats alone.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = partial(unet_fit_rectangle_preprocess, output_size=OUTPUT_SIZE)

    model = UNet(in_channels=1, out_channels=1, kernel_size=kernel_size, sigmoid=sigmoid)
    lit_net = LitNet.load_from_checkpoint(
        checkpoint_path,
        model=model,
        criterion=BCEDiceLoss(),
        num_epochs=1,
        mean_coords=mean_coords,
        strict=False,
    )
    lit_net.eval()
    lit_net.to(device)

    pairs = _discover_pairs()
    if max_samples is not None:
        pairs = pairs[:max_samples]

    def predict_coords(image_path):
        image, x_size, y_size = load_image(image_path, preprocess)
        # DeepWings' own images are pre-flipped relative to ours -- see
        # notebook 26's intro markdown, re-verified there on a sample rather
        # than taken on faith.
        image = torch.flip(image, dims=[-1])
        with torch.no_grad():
            output = lit_net.model(image.unsqueeze(0).to(device))
        probs = output if sigmoid else torch.sigmoid(output)
        mask = (probs > 0.5).float().squeeze().cpu().numpy()
        return final_coords(mask, x_size, y_size)

    def ground_truth_coords(mask_path):
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 1)
        return mask_to_coords(img.astype(np.float32) / 255.0)

    precisions = []
    n_pred_list = []
    for image_path, mask_path in tqdm(
        pairs, desc=f"DeepWings eval: {Path(checkpoint_path).name}"
    ):
        pred_coords = predict_coords(image_path)
        gt_coords = ground_truth_coords(mask_path)
        n_pred_list.append(len(pred_coords))
        _, precision = gpa_ordered_metrics_both(
            pred_coords,
            gt_coords,
            mean_coords,
            allow_reflection=True,
            multistart_angles=FULL_ROTATION_MULTISTART_ANGLES,
        )
        precisions.append(precision)

    precisions = np.array(precisions, dtype=float)
    n_pred_arr = np.array(n_pred_list)
    reliable = (n_pred_arr >= 16) & (n_pred_arr <= 22)

    return {
        "checkpoint": str(checkpoint_path),
        "n_samples": len(pairs),
        "precision_mean": float(np.nanmean(precisions)),
        "precision_median": float(np.nanmedian(precisions)),
        "reliable_pct": float(reliable.mean() * 100.0) if len(reliable) else float("nan"),
        "precisions": precisions.tolist(),
        "n_pred": n_pred_arr.tolist(),
    }


def _discover_bmp_pairs() -> list[tuple[Path, Path]]:
    """The 21 `.bmp` files in DeepWings' test/ that _discover_pairs() skips
    (wings.utils.load_image's torchvision decoder can't read .bmp at all --
    see notebook 26's Setup section). Their ground-truth masks are also
    saved as .bmp, matched by filename exactly like _discover_pairs() does
    for the .jpg majority."""
    all_files = list(TEST_DIR.iterdir())
    bmp_files = sorted(p for p in all_files if p.suffix.lower() == ".bmp")
    mask_files = {p.name: p for p in MASK_DIR.iterdir() if p.suffix.lower() == ".bmp"}
    return [(p, mask_files[p.name]) for p in bmp_files if p.name in mask_files]


def evaluate_bmp_stragglers_on_deepwings(
    checkpoint_path,
    mean_coords,
    sigmoid: bool = False,
    kernel_size: int = 5,
    device=None,
) -> dict:
    """Scores the 21 `.bmp` images evaluate_checkpoint_on_deepwings() has to
    skip. cv2.imread decodes .bmp natively (unlike the torchvision decoder
    wings.utils.load_image uses), so this one-off reads just these 21 that
    way instead of teaching the shared production loader a format used by
    0.6% of one external dataset. Everything downstream -- preprocessing,
    model, the DeepWings-orientation flip, gpa_ordered_metrics_both -- is
    identical to evaluate_checkpoint_on_deepwings, so the two sets of
    precisions are directly comparable/combinable. Returns 'precisions'/
    'n_pred' (raw lists, same order as _discover_bmp_pairs(), same shape as
    evaluate_checkpoint_on_deepwings()'s own) rather than a mean/median of
    just 21 points -- fold them into evaluate_checkpoint_on_deepwings()'s
    own result instead of treating this as a result on its own.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = partial(unet_fit_rectangle_preprocess, output_size=OUTPUT_SIZE)

    model = UNet(in_channels=1, out_channels=1, kernel_size=kernel_size, sigmoid=sigmoid)
    lit_net = LitNet.load_from_checkpoint(
        checkpoint_path,
        model=model,
        criterion=BCEDiceLoss(),
        num_epochs=1,
        mean_coords=mean_coords,
        strict=False,
    )
    lit_net.eval()
    lit_net.to(device)

    pairs = _discover_bmp_pairs()

    def predict_coords(image_path):
        arr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(arr).unsqueeze(0)
        x_size, y_size = image.shape[2], image.shape[1]
        image, _, _ = preprocess(image)
        image = torch.flip(image, dims=[-1])
        with torch.no_grad():
            output = lit_net.model(image.unsqueeze(0).to(device))
        probs = output if sigmoid else torch.sigmoid(output)
        mask = (probs > 0.5).float().squeeze().cpu().numpy()
        return final_coords(mask, x_size, y_size)

    def ground_truth_coords(mask_path):
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 1)
        return mask_to_coords(img.astype(np.float32) / 255.0)

    precisions = []
    n_pred_list = []
    for image_path, mask_path in tqdm(
        pairs, desc=f"DeepWings .bmp stragglers: {Path(checkpoint_path).name}"
    ):
        pred_coords = predict_coords(image_path)
        gt_coords = ground_truth_coords(mask_path)
        n_pred_list.append(len(pred_coords))
        _, precision = gpa_ordered_metrics_both(
            pred_coords,
            gt_coords,
            mean_coords,
            allow_reflection=True,
            multistart_angles=FULL_ROTATION_MULTISTART_ANGLES,
        )
        precisions.append(precision)

    return {
        "checkpoint": str(checkpoint_path),
        "n_samples": len(pairs),
        "precisions": precisions,
        "n_pred": n_pred_list,
    }
