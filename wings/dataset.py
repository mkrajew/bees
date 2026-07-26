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

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image, ImageReadMode
from tqdm import tqdm

from wings.config import COORDS_SUFX, IMG_FOLDER_SUFX
from wings.transforms import TrainAugmentConfig, build_eval_transform, build_train_transform

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
        countries: List of country names where the images in the dataset come from.
    """

    preprocess_func: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        countries: list[str],
        data_folder: Path,
        preprocess_func: Callable[[torch.Tensor], Any],
    ) -> None:
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
        self.countries = countries

        self.coords_df = pd.DataFrame()
        for country in countries:
            coords_file = data_folder / f"{country}{COORDS_SUFX}"
            df = pd.read_csv(coords_file)
            self.coords_df = pd.concat([self.coords_df, df], ignore_index=True)
        self.coords_df["orig_label"] = self.coords_df.iloc[:, 1:].progress_apply(
            lambda row: torch.tensor(row.values, dtype=torch.float32), axis=1
        )
        self.coords_df = self.coords_df[["file", "orig_label"]]
        self.coords_df["normalized"] = False
        self.coords_df["orig_size"] = None
        self.coords_df["label"] = None

    def load_image(self, filename: str) -> tuple[torch.Tensor, int, int]:
        """
        Loads and preprocesses an image tensor.

        Args:
            filename: Name of the image file to load.

        Returns:
            Tuple of the image tensor, with width (x_size) and height (y_size) of the original image.
        """

        country = filename.split("-", 1)[0]
        image = decode_image(
            str(self.data_folder / f"{country}{IMG_FOLDER_SUFX}" / filename),
            mode=ImageReadMode.GRAY,
        )
        x_size, y_size = image.shape[2], image.shape[1]
        # image = image.repeat(3, 1, 1)
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

        filename = self.coords_df.loc[index, "file"]
        image, orig_x_size, orig_y_size = self.load_image(filename)
        self.coords_df.at[index, "orig_size"] = (orig_x_size, orig_y_size)
        x_size, y_size = image.shape[2], image.shape[1]
        if not self.coords_df.loc[index, "normalized"]:
            self.coords_df.at[index, "label"] = torch.zeros_like(
                self.coords_df.at[index, "orig_label"]
            )
            self.coords_df.loc[index, "label"][::2] = (
                self.coords_df.loc[index, "orig_label"][::2] * x_size / orig_x_size
            ).int()
            self.coords_df.loc[index, "label"][1::2] = (
                self.coords_df.loc[index, "orig_label"][1::2] * y_size / orig_y_size
            ).int()
            self.coords_df.loc[index, "normalized"] = True

        labels = self.coords_df.loc[index, "label"]

        return image, labels

    def split(
        self,
        val_percentage: float = 0.2,
        test_percentage: float = 0.1,
        seed: int = 42,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """
        Splits the dataset into training, validation, and testing subsets.

        Args:
            val_percentage: Fraction of data to use for validation.
            test_percentage: Fraction of data to use for testing.
            seed: Seed for the split's random generator, for reproducibility.

        Returns:
            Datasets for training, validation, and testing.
        """

        val_size = int(len(self) * val_percentage)
        test_size = int(len(self) * test_percentage)
        train_size = len(self) - val_size - test_size

        generator = torch.Generator().manual_seed(seed)
        train_set, valid_set, test_set = data.random_split(
            self, [train_size, val_size, test_size], generator=generator
        )

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
        tup, x_size, y_size = super(WingsDatasetRectangleImages, self).load_image(
            filename
        )
        image, pad_left, pad_bottom = tup
        return image, x_size, y_size, pad_left, pad_bottom

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.coords_df.loc[index, "file"]
        image, orig_x_size, orig_y_size, pad_left, pad_bottom = self.load_image(
            filename
        )
        self.coords_df.at[index, "orig_size"] = (orig_x_size, orig_y_size)
        x_size, y_size = image.shape[2], image.shape[1]
        factor = (
            x_size / orig_x_size if orig_x_size >= orig_y_size else y_size / orig_y_size
        )
        if not self.coords_df.loc[index, "normalized"]:
            self.coords_df.at[index, "label"] = torch.zeros_like(
                self.coords_df.at[index, "orig_label"]
            )
            self.coords_df.loc[index, "label"][::2] = (
                self.coords_df.loc[index, "orig_label"][::2] * factor
            ).int() + pad_left
            self.coords_df.loc[index, "label"][1::2] = (
                self.coords_df.loc[index, "orig_label"][1::2] * factor
            ).int() + pad_bottom
            self.coords_df.loc[index, "normalized"] = True

        labels = self.coords_df.loc[index, "label"]

        return image, labels


def generate_landmark_mask(
    image: torch.Tensor, labels: torch.Tensor, square_size: int
) -> torch.Tensor:
    """Paints a `square_size` x `square_size` block of 1s around each landmark.

    `labels` must be flat (x0, y0, x1, y1, ...) in bottom-left convention, matching
    the on-disk CSVs; `image` is used only for its (square) spatial size.
    """
    x_coords, y_coords = labels[::2].int(), labels[1::2].int()
    x_size, y_size = image.shape[2], image.shape[1]
    assert x_size == y_size, f"Expected square image, got {x_size=} {y_size=}"
    img_size = x_size

    y_coords = y_size - y_coords - 1

    mask = np.zeros((img_size, img_size), dtype=np.float32)
    square_half = square_size // 2
    for x, y in zip(x_coords, y_coords):
        x_start = max(0, x - square_half)
        x_end = min(img_size, x + square_half + 1)
        y_start = max(0, y - square_half)
        y_end = min(img_size, y + square_half + 1)
        mask[y_start:y_end, x_start:x_end] = 1

    return torch.from_numpy(mask).unsqueeze(0)


class MasksDataset(WingsDataset):
    def __init__(
        self,
        countries: list[str],
        data_folder: Path,
        preprocess_func: Callable[[torch.Tensor], Any],
        square_size: int = 5,
    ) -> None:
        super(MasksDataset, self).__init__(countries, data_folder, preprocess_func)
        self.square_size = square_size

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        image, labels = super(MasksDataset, self).__getitem__(index)
        mask = self.generate_mask(image, labels)
        orig_size = self.coords_df.loc[index, "orig_size"]
        orig_labels = self.coords_df.loc[index, "orig_label"]
        return image, mask, orig_labels, orig_size

    def generate_mask(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return generate_landmark_mask(image, labels, self.square_size)

    def generate_circular_mask(
        self, image: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        x_coords, y_coords = labels[::2].int(), labels[1::2].int()
        x_size, y_size = image.shape[2], image.shape[1]
        assert x_size == y_size, f"Expected square image, got {x_size=} {y_size=}"
        img_size = x_size

        y_coords = y_size - y_coords - 1

        mask = np.zeros((img_size, img_size), dtype=np.float32)
        radius = self.square_size // 2
        radius_sq = radius**2
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        for x, y in zip(x_coords.tolist(), y_coords.tolist()):
            circle = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius_sq
            mask[circle] = 1

        return torch.from_numpy(mask).unsqueeze(0)


class MaskRectangleDataset(MasksDataset, WingsDatasetRectangleImages):
    preprocess_func: Callable[[torch.Tensor], tuple[torch.Tensor, int, int]]

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        image, labels = WingsDatasetRectangleImages.__getitem__(self, index)
        mask = self.generate_mask(image, labels)
        orig_size = self.coords_df.loc[index, "orig_size"]
        orig_labels = self.coords_df.loc[index, "orig_label"]
        return image, mask, orig_labels, orig_size


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


def _identity(image: torch.Tensor) -> torch.Tensor:
    """Picklable no-op preprocess_func for WingsRawDataset (a lambda here would
    break Windows DataLoader workers, which pickle the whole dataset via spawn)."""
    return image


class WingsRawDataset(WingsDataset):
    """Loads only raw image bytes + raw (bottom-left, original-pixel-space) label
    tensor + original size -- no preprocessing, no per-access caching.

    This is the raw source for `TransformedMaskDataset`, which applies a transform
    (random for training, deterministic for eval) after this dataset has already
    been split into train/val/test -- letting each split get independently
    configured behavior despite sharing one underlying image/label source. Unlike
    `WingsDataset.__getitem__`, nothing here mutates `coords_df`, since a cached
    "normalized" label would only be valid for one particular random transform draw.
    """

    def __init__(self, countries: list[str], data_folder: Path) -> None:
        super().__init__(countries, data_folder, preprocess_func=_identity)

    @override
    def load_image(self, filename: str) -> torch.Tensor:
        country = filename.split("-", 1)[0]
        return decode_image(
            str(self.data_folder / f"{country}{IMG_FOLDER_SUFX}" / filename),
            mode=ImageReadMode.GRAY,
        )

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        filename = self.coords_df.loc[index, "file"]
        image = self.load_image(filename)
        orig_label = self.coords_df.loc[index, "orig_label"]
        orig_size = (image.shape[2], image.shape[1])
        return image, orig_label, orig_size


class TransformedMaskDataset(Dataset):
    """Wraps a `WingsRawDataset` (or a `Subset` of one) with an injected joint
    image+keypoint transform and mask generation.

    The transform is the only thing that differs between train/val/test instances
    built from the same underlying raw dataset (see `build_mask_datasets`): pass a
    random `wings.transforms.build_train_transform(...)` for training or the
    deterministic `wings.transforms.build_eval_transform(...)` for val/test.
    """

    def __init__(
        self,
        base: Dataset,
        transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        square_size: int = 5,
    ) -> None:
        self.base = base
        self.transform = transform
        self.square_size = square_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        image, orig_label, orig_size = self.base[index]
        orig_width, orig_height = orig_size

        x_coords = orig_label[::2].clone()
        y_coords_top = orig_height - orig_label[1::2].clone() - 1
        keypoints = torch.stack([x_coords, y_coords_top], dim=-1)

        image = tv_tensors.Image(image)
        keypoints = tv_tensors.KeyPoints(keypoints, canvas_size=(orig_height, orig_width))

        transformed_image, transformed_keypoints = self.transform(image, keypoints)
        transformed_image = transformed_image.as_subclass(torch.Tensor)
        transformed_keypoints = transformed_keypoints.as_subclass(torch.Tensor)

        final_size = transformed_image.shape[-1]
        final_labels = torch.empty_like(orig_label)
        final_labels[::2] = transformed_keypoints[:, 0]
        final_labels[1::2] = final_size - transformed_keypoints[:, 1] - 1

        mask = generate_landmark_mask(transformed_image, final_labels, self.square_size)

        return transformed_image, mask, orig_label, orig_size


def build_mask_datasets(
    countries: list[str],
    data_folder: Path,
    output_size: int = 400,
    square_size: int = 5,
    val_percentage: float = 0.2,
    test_percentage: float = 0.1,
    split_seed: int = 42,
    train_augment_cfg: TrainAugmentConfig | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Builds train/val/test mask datasets that share one raw image/label source
    but each apply a different transform: fresh random augmentation for train, and
    the deterministic `unet_fit_rectangle_preprocess`-equivalent transform for
    val/test. Replaces `load_datasets(...)` for training runs that want online
    augmentation instead of loading pre-augmented, pre-pickled dataset files.
    """
    raw = WingsRawDataset(countries, data_folder)
    train_idx, val_idx, test_idx = raw.split(val_percentage, test_percentage, seed=split_seed)

    train_transform = build_train_transform(output_size, train_augment_cfg)
    eval_transform = build_eval_transform(output_size)

    train_set = TransformedMaskDataset(train_idx, train_transform, square_size)
    val_set = TransformedMaskDataset(val_idx, eval_transform, square_size)
    test_set = TransformedMaskDataset(test_idx, eval_transform, square_size)

    return train_set, val_set, test_set


if __name__ == "__main__":
    from functools import partial
    from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess
    from wings.config import COUNTRIES, PROCESSED_DATA_DIR

    square_size = 5
    preprocess = partial(unet_fit_rectangle_preprocess, output_size=400)

    mask_dataset = MaskRectangleDataset(
        COUNTRIES, PROCESSED_DATA_DIR / "cropped", preprocess, square_size=square_size
    )

    train_mask_dataset, val_mask_dataset, test_mask_dataset = mask_dataset.split(
        0.2, 0.1
    )

    folder = PROCESSED_DATA_DIR / "mask_datasets" / "rectangle-cropped"
    folder.mkdir(parents=True, exist_ok=True)

    torch.save(train_mask_dataset, folder / "train_mask_dataset_ch1_400_sq5.pth")
    torch.save(val_mask_dataset, folder / "val_mask_dataset_ch1_400_sq5.pth")
    torch.save(test_mask_dataset, folder / "test_mask_dataset_ch1_400_sq5.pth")
