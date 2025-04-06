from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from wings.config import RAW_DATA_DIR, IMG_FOLDER_SUFX, COORDS_SUFX


def plt_imshow(img: np.ndarray, img_title: Optional[str] = None) -> None:
    """
    Displays an image using matplotlib with optional title.
    The image is shown in grayscale with intensity range fixed to [0, 255].

    Args:
        img (np.ndarray): The image to display.
        img_title (str, optional): Optional title for the image window.
    """

    plt.figure()
    plt.title(img_title)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


def visualize_from_file(filename: str, data_folder: Path | str = RAW_DATA_DIR) -> None:
    """
    Loads an image and its corresponding coordinate annotations from files,
    then visualizes the annotated coordinates on the image using function visualize_coords.

    Args:
        filename (str): Name of the image file.
        data_folder (Path or str): Root folder containing the image and coordinate subfolders.
    """

    country = filename.split('-', 1)[0]
    imgpath = data_folder / f"{country}{IMG_FOLDER_SUFX}" / filename
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

    df = pd.read_csv(data_folder / f"{country}{COORDS_SUFX}")
    row = df[df['file'] == filename].iloc[0]

    targets = pd.to_numeric(row.iloc[1:].values)
    targets = torch.tensor(targets, dtype=torch.float32)

    visualize_coords(img, targets, filename=filename)


def visualize_coords(img: np.ndarray, targets: torch.Tensor, *, filename: Optional[str] = None, spot_size: int = 6,
                     show: bool = True, save_path: Optional[Path | str] = None) -> None:
    """
    Draws target coordinates on an image as green circles and optionally displays or saves it.

    Args:
        img (np.ndarray): The image on which to draw.
        targets (torch.Tensor): A 1D tensor of alternating x and y coordinates.
        filename (str, optional): Optional title used when displaying the image.
        spot_size (int): Radius of the circle drawn at each coordinate point.
        show (bool): If True, displays the image using matplotlib.
        save_path (str or Path, optional): If provided, saves the annotated image to this path.
    """

    x_size, y_size = img.shape[1], img.shape[0]
    x_coords, y_coords = targets[::2], targets[1::2]
    y_coords = y_size - y_coords

    for x, y in zip(x_coords, y_coords):
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), spot_size, (0, 255, 0), -1)

    if show:
        plt_imshow(img, filename)

    if save_path:
        cv2.imwrite(save_path, img)
