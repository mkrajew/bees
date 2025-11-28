from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchvision.io import decode_image, ImageReadMode


def order_coords(predicted, original):
    predicted = np.array(predicted)
    original = np.array(original)

    original = np.array(list(zip(original[::2], original[1::2])))
    distance_matrix = cdist(original, predicted)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Does not preserve the original coords order
    matched_predicted = predicted[col_ind]
    matched_original = original[row_ind]

    # Preserves the original coords order
    # matched_predicted = np.zeros_like(original)
    # for orig_idx, pred_idx in zip(row_ind, col_ind):
    #     matched_predicted[orig_idx] = predicted[pred_idx]

    return matched_predicted, matched_original


def load_image(filepath: Path, preprocess_func: Callable[[torch.Tensor], Any]) -> tuple[torch.Tensor, int, int]:
    """
    Loads and preprocesses an image tensor.

    Args:
        filepath: Path of the image file to load.
        preprocess_func: function to preprocess image tensors.

    Returns:
        The image tensor.
    """

    image = decode_image(str(filepath), mode=ImageReadMode.RGB, apply_exif_orientation=True)
    x_size, y_size = image.shape[2], image.shape[1]
    # if image.shape[0] == 1:
    #     image = image.repeat(3, 1, 1)
    image, _, _ = preprocess_func(image)
    return image, x_size, y_size
