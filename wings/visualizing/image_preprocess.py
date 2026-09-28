"""
Image preprocessing utilities for deep learning models.

This module provides functions to resize_preprocess and reverse-resize_preprocess images
in a way that aligns with standard pipelines used in pretrained models (e.g., ResNet, ViT).

Constants:
    mean: Channel-wise mean used for normalization in ImageNet.
    std: Channel-wise standard deviation used for normalization in ImageNet.
"""

import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def resize_preprocess(img: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses an input image for use with ResNet and Vision Transformer (ViT) models.

    This function applies standard preprocessing steps used for models trained on ImageNet.
    Unlike some preprocessing pipelines that apply center cropping, this function
    resizes the image directly to 224x224, preserving more of the original image content.

    Steps:
    1. Resizes the image to 224x224 using bilinear interpolation with antialiasing.
    2. Converts the image to a tensor if it is not already.
    3. Converts the tensor to float with values in the [0, 1] range.
    4. Normalizes the image using ImageNet mean and standard deviation.

    This preprocessing is suitable for PyTorch-based ResNet and ViT (Vision Transformer) models.

    Args:
        img: Input image tensor or PIL image.

    Returns:
        Preprocessed image tensor ready for model input.
    """

    img = F.resize(
        img, [224, 224], interpolation=F.InterpolationMode.BILINEAR, antialias=True
    )
    if not isinstance(img, torch.Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)
    return img


def unet_preprocess(img: torch.Tensor) -> torch.Tensor:
    """
    Resizes image tensor to 256x256 pixels and normalizes z-score per volume.

    Args:
        img: Input image tensor or PIL image.

    Returns:
        Preprocessed image tensor ready for U-Net model input.
    """
    img = F.resize(
        img, [256, 256], interpolation=F.InterpolationMode.BILINEAR, antialias=False
    )
    img = F.convert_image_dtype(img, torch.float)
    m, s = img.mean(dim=(1, 2)), img.std(dim=(1, 2))
    img = F.normalize(img, mean=m, std=s)
    return img


def unet_fit_rectangle_preprocess(
    img: torch.Tensor, output_size: int = 256
) -> tuple[torch.Tensor, int, int]:
    img = F.resize(
        img,
        output_size - 1,
        interpolation=F.InterpolationMode.BILINEAR,
        antialias=True,
        max_size=output_size,
    )
    _, h, w = img.shape
    pad_h = output_size - h
    pad_w = output_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    img = F.pad(
        img, [pad_left, pad_top, pad_right, pad_bottom], padding_mode="constant", fill=0
    )
    img = F.convert_image_dtype(img, torch.float)
    m, s = img.mean(dim=(1, 2)), img.std(dim=(1, 2))
    img = F.normalize(img, mean=m, std=s.clamp(min=1e-6))

    return img, pad_left, pad_bottom


def denormalize(img: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet-style normalization on a tensor image.

    This function is intended to convert a normalized image tensor
    (as used in ResNet or ViT preprocessing) back to its original
    image format for visualization or saving.

    Steps:
    1. Reverses normalization using ImageNet mean and standard deviation.
    2. Converts the tensor to a NumPy array with shape (H, W, C).
    3. Scales the image to [0, 255] and converts it to uint8 format.
    4. Ensures the image is stored in a contiguous array.

    Args:
        img: A normalized image tensor of shape (3, H, W), with float values.

    Returns:
        Denormalized image as a NumPy array in (H, W, C) format with dtype uint8.
    """

    mean_d = torch.tensor(mean).view(3, 1, 1)
    std_d = torch.tensor(std).view(3, 1, 1)
    img = img * std_d + mean_d
    img = img.numpy().transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    return img


def fit_rectangle_preprocess(img: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """
    Resizes and pads an image to fit into a 224x224 square while preserving aspect ratio.

    Steps:
    1. Resizes the image so that the width dimension is 224 pixels, preserving aspect ratio.
       Asserts that the width of the image is bigger than its height.
    2. Applies vertical padding (top and bottom) to reach a final height of 224 pixels.
    3. Converts the image to float and normalizes it using ImageNet mean and standard deviation.

    This preprocessing avoids the distortion caused by direct resizing of rectangular images.

    Args:
        img: Input image tensor of shape (C, H, W), typically with dtype uint8 or float.

    Returns:
        A tuple of:
            - Preprocessed image tensor of shape (3, 224, 224), normalized.
            - Number of pixels padded at the top.
            - Number of pixels padded at the bottom.
    """

    img = F.resize(
        img,
        223,
        interpolation=F.InterpolationMode.BILINEAR,
        antialias=True,
        max_size=224,
    )
    _, h, w = img.shape
    if w >= h:
        pad_left = 0
        pad_top = (224 - h) // 2
        pad_right = 0
        pad_bottom = 224 - h - pad_top
    else:
        pad_left = (224 - w) // 2
        pad_top = 0
        pad_right = 224 - w - pad_left
        pad_bottom = 0
    img = F.pad(
        img, [pad_left, pad_top, pad_right, pad_bottom], padding_mode="constant", fill=0
    )
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)

    return img, pad_left, pad_bottom


def _split_blob_watershed(roi_binary):
    """roi_binary: (h, w) uint8, 0/255, a single (possibly merged) blob,
    already isolated from any other blob's pixels. Returns a list of (x, y)
    centroids, one per detected peak, or None if fewer than 2 peaks are
    found (i.e. this blob isn't actually a merge)."""
    dist = cv2.distanceTransform(roi_binary, cv2.DIST_L2, 5)
    dist_max = dist.max()
    if dist_max == 0:
        return None

    # Local maxima: a pixel equals its own value after a small dilation
    # (no neighbor within the kernel is strictly larger), restricted to a
    # fraction of the peak distance to suppress flat/noisy low-lying maxima.
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(dist, kernel)
    local_max = (dist == dilated) & (dist > 0.5 * dist_max)

    n_labels, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if n_labels <= 2:  # background (0) + at most one peak region -> not a merge
        return None

    markers = markers + 1  # cv2.watershed reserves 0 for "unknown"
    markers[roi_binary == 0] = 0

    roi_bgr = cv2.cvtColor(roi_binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(roi_bgr, markers)

    centroids = []
    for label in range(2, n_labels + 1):
        ys, xs = np.where(markers == label)
        if len(xs) == 0:
            continue
        # float(...): xs.mean()/ys.mean() are numpy.float64 scalars, unlike
        # cv2.moments()'s plain-Python-float centroids elsewhere in
        # mask_to_coords. torch.tensor() infers float64 from even one
        # numpy.float64 element in an otherwise plain-float list, silently
        # promoting the *entire* coordinate list to double -- which then
        # mismatches mean_coords' float32 in procrustes_align's matmul
        # (RuntimeError: expected m1 and m2 to have the same dtype).
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


def mask_to_coords(mask, area_ratio=1.5, pad=5):
    binary = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Two landmarks close enough together in mean_coords (e.g. landmarks
    # 1/2, the closest pair of all 19) can have their circular mask targets
    # touch or overlap, especially at larger mask radii (see
    # jobs/online/README.md's config 6/7 sections -- radius 4, up from 2,
    # improved rotation robustness and DeepWings generalization but made
    # this more likely). cv2.findContours then merges them into one blob
    # and returns a single centroid roughly between the two true positions,
    # instead of two. Contours anomalously larger than the image's own
    # median (area_ratio) are treated as suspected merges: a distance
    # transform of just that blob finds one local maximum per original
    # circle's center, and cv2.watershed splits the blob along the "valley"
    # between them, giving one centroid per peak instead of one for the
    # whole merged region. Normal, isolated blobs (the vast majority) never
    # hit this path at all, so their centroids are exactly the plain
    # contour-moment calculation, unchanged. See
    # notebooks/27_mask_blob_merge_fix.ipynb for why this was chosen over a
    # simpler unconditional erosion (matches erosion's fix rate and speed,
    # without erosion's ~0.25% chance of erasing an already-correct,
    # separate small blob entirely).
    areas = np.array([cv2.contourArea(c) for c in contours])
    median_area = np.median(areas) if len(areas) >= 3 else None

    img_y_size = mask.shape[0]
    coordinates = []
    for cnt, area in zip(contours, areas):
        if median_area is not None and area > area_ratio * median_area:
            x, y, w, h = cv2.boundingRect(cnt)
            y0, y1 = max(0, y - pad), min(binary.shape[0], y + h + pad)
            x0, x1 = max(0, x - pad), min(binary.shape[1], x + w + pad)
            roi = binary[y0:y1, x0:x1].copy()
            roi_mask = np.zeros_like(roi)
            cv2.drawContours(roi_mask, [cnt - [x0, y0]], -1, 255, thickness=cv2.FILLED)
            roi = cv2.bitwise_and(roi, roi_mask)

            split = _split_blob_watershed(roi)
            if split:
                for px, py in split:
                    coordinates.append((px + x0, img_y_size - (py + y0) - 1))
                continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]  # x coordinate of centroid
            cy = M["m01"] / M["m00"]  # y coordinate of centroid
            cy = img_y_size - cy - 1
            coordinates.append((cx, cy))

    return coordinates


def unet_reverse_padding(
    padded_img: torch.Tensor, w_orig: int, h_orig: int
) -> tuple[int, int, int, int]:
    """
    Reverses the padding added during unet_fit_rectangle_preprocess.
    Returns (pad_left, pad_top, pad_right, pad_bottom).
    Assumes the padded image is 256x256 and resizing used max_size=256 with max dim 255.
    """
    padded_h, padded_w = padded_img.shape
    # assert padded_h == 256 and padded_w == 256, "Expected padded image to be 256x256"

    # Recompute resize scale from original dimensions
    if w_orig >= h_orig:
        scale = padded_h / w_orig
    else:
        scale = padded_h / h_orig

    resized_w = round(w_orig * scale)
    resized_h = round(h_orig * scale)

    pad_w_total = padded_h - resized_w
    pad_h_total = padded_h - resized_h

    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top

    return pad_left, pad_top, pad_right, pad_bottom


def final_coords(mask, orig_width, orig_height):
    mask_coords = mask_to_coords(mask)

    mask_height, mask_width = mask.shape

    pad_left, pad_top, pad_right, pad_bottom = unet_reverse_padding(
        mask, orig_width, orig_height
    )

    mask_coords = [(x - pad_left, y - pad_bottom) for x, y in mask_coords]

    scale_x = orig_width / (mask_width - pad_right - pad_left)
    scale_y = orig_height / (mask_height - pad_top - pad_bottom)
    mask_coords = [(x * scale_x, y * scale_y) for x, y in mask_coords]

    return mask_coords
