"""
This module provides Dataset classes and utils for loading and preprocessing
wing images with their keypoint coordinate labels.

Each image is associated with a CSV entry containing 19 (x, y) coordinate pairs (total 38 values) used as labels.
The datasets support per-image preprocessing, on-the-fly normalization of coordinates based on image size, and
dataset splitting into training, validation, and test subsets.


Usage Example:
    dataset = WingsDataset(['NG', 'BR'], Path("/data/wings"), preprocess_func)
    train_set, val_set, test_set = dataset.split()
"""

from pathlib import Path
from typing import override, Callable, Any

import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.io import decode_image
from tqdm import tqdm

from wings.config import COORDS_SUFX, IMG_FOLDER_SUFX

tqdm.pandas()


class WingsDataset(data.Dataset):
    """
    A PyTorch Dataset for loading and preprocessing wing keypoint data from image files
    and corresponding coordinate CSVs.

    This dataset is designed to load wing images associated with labeled keypoints, apply preprocessing transformations
    to the images, and normalize the coordinates. It supports data loading from multiple countries and provides a method
    to split the dataset into training, validation, and testing sets.

    Attributes:
        coords_df: A Pandas Dataframe containing the filenames and corresponding coordinates
            with information if they were already normalized.
        preprocess_func: Function used to preprocess the images.
    """

    preprocess_func: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, countries: list[str], data_folder: Path, preprocess_func: Callable[[torch.Tensor], Any]) -> None:
        """
        Initializes the dataset by loading filenames with their coordinates and preparing the dataframe.

        Args:
            countries: List of country codes used to locate data files.
            data_folder: Base path containing coordinate CSVs and image folders.
            preprocess_func: function to preprocess image tensors.
        """

        super(WingsDataset, self).__init__()

        self.data_folder = data_folder
        self.preprocess_func = preprocess_func

        self.coords_df = pd.DataFrame()
        for country in countries:
            coords_file = data_folder / f"{country}{COORDS_SUFX}"
            df = pd.read_csv(coords_file)
            self.coords_df = pd.concat([self.coords_df, df], ignore_index=True)
        self.coords_df['label'] = self.coords_df.iloc[:, 1:].progress_apply(
            lambda row: torch.tensor(row.values, dtype=torch.float32), axis=1
        )
        self.coords_df = self.coords_df[['file', 'label']]
        self.coords_df['normalized'] = False

    def load_image(self, filename: str) -> tuple[torch.Tensor, int, int]:
        """
        Loads and preprocesses an image tensor.

        Args:
            filename: Name of the image file to load.

        Returns:
            Tuple of the image tensor, with width (x_size) and height (y_size) of the original image.
        """

        country = filename.split('-', 1)[0]
        image = decode_image(str(self.data_folder / f"{country}{IMG_FOLDER_SUFX}" / filename))
        x_size, y_size = image.shape[2], image.shape[1]
        image = image.repeat(3, 1, 1)
        image = self.preprocess_func(image)
        return image, x_size, y_size

    def __len__(self) -> int:
        return len(self.coords_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the image and the coordinates from file corresponding to
        the self.coords_df index and then preprocesses the coordinates at the first load.

        Args:
            index: Index of the data sample in self.coords_df to retrieve.

        Returns:
            Tuple containing the image tensor and label tensor of coordinates - 38 numbers
            representing 19 (x, y) coordinates pairs.
        """

        filename = self.coords_df.loc[index, 'file']
        image, orig_x_size, orig_y_size = self.load_image(filename)
        x_size, y_size = image.shape[2], image.shape[1]
        if not self.coords_df.loc[index, 'normalized']:
            self.coords_df.loc[index, 'label'][::2] = (
                        self.coords_df.loc[index, 'label'][::2] * x_size / orig_x_size).int()
            self.coords_df.loc[index, 'label'][1::2] = (
                        self.coords_df.loc[index, 'label'][1::2] * y_size / orig_y_size).int()
            self.coords_df.loc[index, 'normalized'] = True
        labels = self.coords_df.loc[index, 'label']
        return image, labels

    def split(self, val_percentage: float = 0.2, test_percentage: float = 0.1) -> tuple[Dataset, Dataset, Dataset]:
        """
        Splits the dataset into training, validation, and testing subsets.

        Args:
            val_percentage: Fraction of data to use for validation.
            test_percentage: Fraction of data to use for testing.

        Returns:
            Datasets for training, validation, and testing.
        """

        val_size = int(len(self) * val_percentage)
        test_size = int(len(self) * test_percentage)
        train_size = len(self) - val_size - test_size

        seed = torch.Generator().manual_seed(42)
        train_set, valid_set, test_set = data.random_split(self, [train_size, val_size, test_size], generator=seed)

        return train_set, valid_set, test_set


class WingsDatasetRectangleImages(WingsDataset):
    """
    Extends WingsDataset enabling supporting images with rectangular padding during preprocessing.

    This variant of the dataset class is designed to handle images that are padded to maintain
    aspect ratio.
    """

    preprocess_func: Callable[[torch.Tensor], tuple[torch.Tensor, int, int]]

    def load_image(self, filename: str) -> tuple[torch.Tensor, int, int, int, int]:
        """
        Loads and preprocesses an image tensor, additionally returning padding sizes.
        """
        tup, x_size, y_size = super(WingsDatasetRectangleImages, self).load_image(filename)
        image, pad_top, pad_bottom = tup
        print(x_size, y_size)
        return image, x_size, y_size, pad_top, pad_bottom

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.coords_df.loc[index, 'file']
        image, orig_x_size, _, _, pad_bottom = self.load_image(filename)
        x_size = image.shape[2]
        if not self.coords_df.loc[index, 'normalized']:
            self.coords_df.loc[index, 'label'][::2] = (self.coords_df.loc[index, 'label'][::2] * x_size / orig_x_size).int()
            self.coords_df.loc[index, 'label'][1::2] = (self.coords_df.loc[index, 'label'][
                                                        1::2] * x_size / orig_x_size).int() + pad_bottom
            self.coords_df.loc[index, 'normalized'] = True
        labels = self.coords_df.loc[index, 'label']

        return image, labels


class MasksDataset(data.Dataset):
    def __init__(self):
        super(MasksDataset, self).__init__()

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, labels = super(MasksDataset, self).__getitem__(index)
        mask = torch.Tensor([0.])
        return image, mask


def load_datasets(files: list[Path]) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads pre-saved PyTorch datasets from the specified file paths.

    This utility function expects three file path corresponding to the training, validation, and test sets.
    It returns these datasets as PyTorch `Dataset` objects, which can be used directly with DataLoaders.

    Args:
        files: A list of three Path objects pointing to the training, validation,
               and testing dataset files in that order.

    Returns:
        A tuple containing the loaded training, validation, and test datasets.
    """

    train_dataset = torch.load(files[0], weights_only=False)
    val_dataset = torch.load(files[1], weights_only=False)
    test_dataset = torch.load(files[2], weights_only=False)

    return train_dataset, val_dataset, test_dataset
