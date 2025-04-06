from pathlib import Path

import pandas as pd
import torch
import torch.utils.data as data
from jedi.inference.gradual.typing import Callable
from torch.utils.data import Dataset
from torchvision.io import decode_image
from tqdm import tqdm

from wings.config import COORDS_SUFX, IMG_FOLDER_SUFX

tqdm.pandas()


class Dataset(data.Dataset):
    def __init__(self, countries: list[str], data_folder: Path,
                 preprocess_func: Callable[torch.Tensor] | None = None) -> None:
        """
        Initializes the dataset by loading coordinates and preparing the dataframe.

        Args:
            countries: List of country codes used to locate data files.
            data_folder: Base path containing coordinate CSVs and image folders.
            preprocess_func: Optional function to preprocess image tensors.
        """

        super(Dataset, self).__init__()

        self.data_folder = data_folder
        self.preprocess_func = preprocess_func

        self.coords_df = pd.DataFrame()
        for country in countries:
            coords_file = data_folder / f"{country}{COORDS_SUFX}"
            df = pd.read_csv(coords_file)
            self.coords_df = pd.concat([self.coords_df, df], ignore_index=True)
        self.coords_df['label'] = self.coords_df.iloc[:, 1:].progress_apply(
            lambda row: torch.tensor(row.values, dtype=torch.float32), axis=1)
        self.coords_df = self.coords_df[['file', 'label']]
        self.coords_df['normalized'] = False

    def load_image(self, filename: str) -> tuple[torch.Tensor, int, int]:
        """
        Loads and optionally preprocesses an image tensor.

        Args:
            filename: Name of the image file to load.

        Returns:
            Image tensor, with width (x_size) and height (y_size).
        """

        country = filename.split('-', 1)[0]
        image = decode_image(self.data_folder / f"{country}{IMG_FOLDER_SUFX}" / filename)
        x_size, y_size = image.shape[2], image.shape[1]
        image = image.repeat(3, 1, 1)
        if self.preprocess_func is not None:
            image = self.preprocess_func(image)
        return image, x_size, y_size

    def __len__(self) -> int:
        return len(self.coords_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the image and the coordinates from file corresponding to the self.coords_df index and the preprocesses
        the coordinates at the first load.

        Args:
            index: Index of the data sample in self.coords_df to retrieve.

        Returns:
            Tuple containing the image tensor and label tensor of coordinates - 38 numbers
            representing 19 (x, y) coordinates pairs.
        """

        filename = self.coords_df.loc[index, 'file']
        image, x_size, y_size = self.load_image(filename)
        labels = self.coords_df.loc[index, 'label']
        if not self.coords_df.loc[index, 'normalized']:
            labels[::2] = (labels[::2] * 224 / x_size).int()
            labels[1::2] = (labels[1::2] * 224 / y_size).int()
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
