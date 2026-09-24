"""
Online (per-sample, per-epoch) augmentation for wing landmark training.

Replaces the offline pipeline in `wings/detection/augment_dataset.py` (which baked
one fixed rotated+noised copy per image onto disk) with transforms applied fresh on
every dataset access. Geometric transforms (horizontal flip, rotation) and the final
resize/pad step move image and landmark keypoints together so they never drift out
of sync; the two photometric transforms (triangle noise, color jitter) only ever
touch the image.

Keypoints are handled in top-left (x, y) pixel convention throughout this module,
matching the convention used internally by `augment_dataset.py`'s rotation logic.
Callers are responsible for converting to/from the bottom-left convention used by
the on-disk CSVs at the boundary (see `MasksDataset.generate_mask` in
`wings/dataset.py`, which already does this flip).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess


class TriangleNoise(v2.Transform):
    """Draws random small dark triangles onto the image only (keypoints pass through).

    Ports the triangle-drawing logic from `augment_dataset.add_triangle_noise`, but
    draws all random parameters from torch's RNG rather than `numpy.random`, so that
    per-DataLoader-worker randomness is correctly decorrelated without needing a
    custom `worker_init_fn` (PyTorch reseeds each worker's torch RNG automatically;
    it does not touch NumPy's global RNG, which is a well-known DataLoader gotcha).
    """

    def __init__(
        self,
        n_triangles_range: tuple[int, int] = (10, 60),
        min_size: int = 2,
        max_size: int = 6,
    ) -> None:
        super().__init__()
        self.n_triangles_range = n_triangles_range
        self.min_size = min_size
        self.max_size = max_size

    def make_params(self, flat_inputs: list) -> dict:
        low, high = self.n_triangles_range
        n_triangles = int(torch.randint(low, high + 1, (1,)).item())
        return {"n_triangles": n_triangles}

    def transform(self, inpt, params: dict):
        if isinstance(inpt, tv_tensors.KeyPoints) or not isinstance(inpt, torch.Tensor):
            return inpt

        img_np = inpt.permute(1, 2, 0).cpu().numpy().copy()
        h, w = img_np.shape[:2]

        tip_ratio, base_half = 0.9, 0.55
        unit_tri = np.array(
            [[0.0, -tip_ratio], [-base_half, 0.6], [base_half, 0.6]], dtype=np.float32
        )

        for _ in range(params["n_triangles"]):
            cx = float(torch.empty(1).uniform_(0, w).item())
            cy = float(torch.empty(1).uniform_(0, h).item())
            size = float(torch.empty(1).uniform_(self.min_size, self.max_size).item())
            angle = float(torch.empty(1).uniform_(0, 2 * np.pi).item())

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            pts = (rot @ (unit_tri * size).T).T
            pts[:, 0] += cx
            pts[:, 1] += cy

            cv2.fillPoly(img_np, [pts.astype(np.int32)], (0,) * img_np.shape[2])

        out = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().to(inpt.dtype)
        return tv_tensors.wrap(out, like=inpt)


def _resized_dims(h: int, w: int, target_short: int, max_size: int) -> tuple[int, int]:
    """Exactly replicates torchvision's internal resize-with-max_size arithmetic
    (`torchvision.transforms.functional._compute_resized_output_size`) so the
    resize scale `unet_fit_rectangle_preprocess` used can be recovered exactly,
    rather than approximated. Pure integer/float arithmetic, verified to match
    `F.resize`'s actual output shape bit-for-bit across aspect ratios.
    """
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = target_short, int(target_short * long / short)
    if new_long > max_size:
        new_short, new_long = int(max_size * new_short / new_long), max_size
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return new_h, new_w


class KeypointAwareResizePad:
    """Joint image+keypoint version of `unet_fit_rectangle_preprocess`.

    Delegates the image transform entirely to `unet_fit_rectangle_preprocess` (so
    eval-path image output stays pixel-identical to today, which `final_coords`
    and the production inference path depend on), then recovers the *exact*
    resize scale and pad amounts via `_resized_dims` (rather than approximating
    them from the padded output, as `unet_reverse_padding` does) to move the
    keypoints through the identical resize+pad transform with sub-pixel accuracy.
    """

    def __init__(self, output_size: int = 400) -> None:
        self.output_size = output_size

    def __call__(
        self, image: torch.Tensor, keypoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_plain = image.as_subclass(torch.Tensor)
        keypoints_plain = keypoints.as_subclass(torch.Tensor).clone().float()

        _, h, w = image_plain.shape
        processed, _pad_left_unused, _pad_bottom_unused = unet_fit_rectangle_preprocess(
            image_plain, self.output_size
        )

        resized_h, resized_w = _resized_dims(h, w, self.output_size - 1, self.output_size)
        scale_x = resized_w / w
        scale_y = resized_h / h
        pad_left = (self.output_size - resized_w) // 2
        pad_top = (self.output_size - resized_h) // 2

        keypoints_plain[:, 0] = keypoints_plain[:, 0] * scale_x + pad_left
        keypoints_plain[:, 1] = keypoints_plain[:, 1] * scale_y + pad_top

        return processed, keypoints_plain


@dataclass
class TrainAugmentConfig:
    """Reviewable knobs for training-time augmentation severity."""

    rotation_degrees: tuple[float, float] = (-90.0, 90.0)
    horizontal_flip_p: float = 0.5
    triangle_noise_p: float = 0.4
    n_triangles_range: tuple[int, int] = (10, 60)
    triangle_min_size: int = 2
    triangle_max_size: int = 6
    color_jitter_p: float = 0.4
    color_jitter_brightness_range: tuple[float, float] = (0.7, 1.3)
    color_jitter_contrast_range: tuple[float, float] = (0.7, 1.3)


class _TrainTransform:
    """Callable(image, keypoints) -> (image, keypoints) with fresh randomness on
    every call: random horizontal flip, then random rotation, then
    (independently, each 40% of the time by default) triangle noise and color
    jitter, then the deterministic resize+pad.

    A plain module-level class rather than a closure so instances stay picklable
    under Windows' `spawn`-based multiprocessing, which DataLoader workers need
    (a nested closure cannot be pickled by reference the way a class can).
    """

    def __init__(self, output_size: int, cfg: TrainAugmentConfig) -> None:
        self.geometric = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=cfg.horizontal_flip_p),
                v2.RandomRotation(
                    degrees=cfg.rotation_degrees,
                    expand=True,
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
            ]
        )
        self.photometric = v2.Compose(
            [
                v2.RandomApply(
                    [
                        TriangleNoise(
                            n_triangles_range=cfg.n_triangles_range,
                            min_size=cfg.triangle_min_size,
                            max_size=cfg.triangle_max_size,
                        )
                    ],
                    p=cfg.triangle_noise_p,
                ),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=cfg.color_jitter_brightness_range,
                            contrast=cfg.color_jitter_contrast_range,
                        )
                    ],
                    p=cfg.color_jitter_p,
                ),
            ]
        )
        self.resize_pad = KeypointAwareResizePad(output_size)

    def __call__(self, image, keypoints):
        image, keypoints = self.geometric(image, keypoints)
        image, keypoints = self.photometric(image, keypoints)
        image, keypoints = self.resize_pad(image, keypoints)
        return image, keypoints


class _EvalTransform:
    """Deterministic callable(image, keypoints): resize+pad only, no randomness.
    Module-level class for the same picklability reason as `_TrainTransform`.
    """

    def __init__(self, output_size: int) -> None:
        self.resize_pad = KeypointAwareResizePad(output_size)

    def __call__(self, image, keypoints):
        return self.resize_pad(image, keypoints)


def build_train_transform(output_size: int, cfg: TrainAugmentConfig = None) -> _TrainTransform:
    return _TrainTransform(output_size, cfg or TrainAugmentConfig())


def build_eval_transform(output_size: int) -> _EvalTransform:
    return _EvalTransform(output_size)


def seed_worker(worker_id: int) -> None:
    """Defensive DataLoader `worker_init_fn`: reseeds NumPy's global RNG per worker.

    Not required by the transforms in this module (all randomness is drawn from
    torch's RNG, which DataLoader already reseeds correctly per worker), but kept
    as cheap insurance against a future numpy-based transform silently duplicating
    its RNG stream across forked workers.
    """
    np.random.seed(torch.initial_seed() % 2**32)
