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

    img = F.resize(img, [224, 224], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
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
    img = F.resize(img, [256, 256], interpolation=F.InterpolationMode.BILINEAR, antialias=False)
    img = F.convert_image_dtype(img, torch.float)
    m, s = img.mean(dim=(1, 2)), img.std(dim=(1, 2))
    img = F.normalize(img, mean=m, std=s)
    return img


def unet_fit_rectangle_preprocess(img: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    img = F.resize(img, 255, interpolation=F.InterpolationMode.BILINEAR, antialias=True, max_size=256)
    _, h, w = img.shape
    if w >= h:
        pad_left = 0
        pad_top = (256 - h) // 2
        pad_right = 0
        pad_bottom = 256 - h - pad_top
    else:
        pad_left = (256 - w) // 2
        pad_top = 0
        pad_right = 256 - w - pad_left
        pad_bottom = 0
    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], padding_mode='constant', fill=0)
    img = F.convert_image_dtype(img, torch.float)
    m, s = img.mean(dim=(1, 2)), img.std(dim=(1, 2))
    img = F.normalize(img, mean=m, std=s)

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

    img = F.resize(img, 223, interpolation=F.InterpolationMode.BILINEAR, antialias=True, max_size=224)
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
    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], padding_mode='constant', fill=0)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)

    return img, pad_left, pad_bottom


def mask_to_coords(mask, max_iter=1):
    # mask = torch.round(mask).squeeze().detach().cpu().numpy()
    binary = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)

    # Find contours of the white spots
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get center coordinates of each contour
    coordinates = []
    iteration = 1
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img_y_size = mask.shape[0]

    while iteration <= max_iter:
        coordinates = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])  # x coordinate of centroid
                cy = int(M['m01'] / M['m00'])  # y coordinate of centroid
                cy = img_y_size - cy - 1
                coordinates.append((cx, cy))

        if len(coordinates) == 19:
            return coordinates
        binary = cv2.erode(binary, kernel, iterations=1)
        _, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iteration += 1

    raise Exception(f"Found {len(coordinates)} spots in mask.")
