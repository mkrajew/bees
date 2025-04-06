import numpy as np
import torch
import torchvision.transforms.functional as F

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def preprocess(img: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses an input image for use with ResNet and Vision Transformer (ViT) models.

    This function applies standard preprocessing steps used for models trained on ImageNet.
    Note: Unlike some preprocessing pipelines that apply center cropping, this function
    resizes the image directly to 224x224, preserving more of the original image content.

    Steps:
    1. Resizes the image to 224x224 using bilinear interpolation with antialiasing.
    2. Converts the image to a tensor if it is not already.
    3. Converts the tensor to float with values in the [0, 1] range.
    4. Normalizes the image using ImageNet mean and standard deviation.

    This preprocessing is suitable for PyTorch-based ResNet and ViT (Vision Transformer) models.

    Args:
        img (torch.Tensor): Input image tensor or PIL image.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """

    img = F.resize(img, [224, 224], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    if not isinstance(img, torch.Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)
    return img


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
        img (torch.Tensor): A normalized image tensor of shape (3, H, W), with float values.

    Returns:
        np.ndarray: Denormalized image as a NumPy array in (H, W, C) format with dtype uint8.
    """

    mean_d = torch.tensor(mean).view(3, 1, 1)
    std_d = torch.tensor(std).view(3, 1, 1)
    img = img * std_d + mean_d
    img = img.numpy().transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    return img
