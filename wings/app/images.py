""" Wing Images toolkit for a web application """
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, PngImagePlugin

from wings.gpa import handle_coordinates
from wings.utils import load_image
from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess, final_coords


class WingImage:
    """ Represents a bee wing image """

    def __init__(self, filepath: str, model: torch.nn.Module, mean_coords: torch.tensor, coord_labels):
        self._coordinates = None
        self._filepath = Path(filepath)
        self._orig_filepath = Path(filepath)
        self.model = model
        self.mean_coords = mean_coords
        self.coord_labels = coord_labels

        self._calc_coordinates()
        self._calc_sections()

        self._image = cv2.imread(str(self._filepath), cv2.IMREAD_COLOR)

    def _calc_coordinates(self):
        """Calculates the coordinates of the bee wing image landmarks"""
        image_tensor, x_size, y_size = load_image(self._filepath, unet_fit_rectangle_preprocess)
        self.image_tensor = image_tensor
        self._size = (x_size, y_size)

        output = self.model(image_tensor.cuda().unsqueeze(0))
        mask = torch.round(output).squeeze().detach().cpu().numpy()

        mask_coords = final_coords(mask, x_size, y_size)
        mask_coords = torch.tensor(mask_coords)
        # if random.random() < 0.5:
        #     mask_coords = mask_coords[:18]
        self._check_carefully = len(mask_coords) < 19 or len(mask_coords) > 22
        self._coordinates = handle_coordinates(mask_coords, self.mean_coords)

    def _calc_sections(self):
        """Calculates sections used for displaying gradio AnnotatedImage sections"""
        sections_arr = []
        W, H = self._size
        Y, X = np.ogrid[:H, :W]
        r = 3
        R = 12

        for idx, (x, y) in enumerate(self._coordinates.cpu().numpy()):
            y_img = H - y - 1
            mask_small = ((X - x) ** 2 + (Y - y_img) ** 2) < r ** 2
            mask_small = mask_small.astype(float)

            mask_large = ((X - x) ** 2 + (Y - y_img) ** 2) < R ** 2
            mask_large = mask_large.astype(float)
            mask_large[mask_small > 0] = 0
            mask_large *= 0.3

            combined_mask = mask_small + mask_large

            sections_arr.append((combined_mask, self.coord_labels[idx]))

        self._sections = sections_arr

    @property
    def sections(self):
        return self._sections

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords):
        self._coordinates = coords
        self._calc_sections()

    @property
    def filename(self):
        return self._filepath.name

    @filename.setter
    def filename(self, value):
        self._filepath = self._filepath.with_name(value).with_suffix('.png')

    @property
    def check_carefully(self):
        return self._check_carefully

    @property
    def size(self):
        return self._size

    def __eq__(self, other):
        return self._filepath == other._filepath

    def generate_image_with_meta_landmarks(self):
        img = Image.open(self._orig_filepath)
        img.load()

        y_size = self.size[1]
        coords_np = self._coordinates.detach().cpu().numpy().copy()
        coords_np[:, 1] = y_size - coords_np[:, 1] - 1
        labels_str = " ".join(str(int(x)) for x in coords_np.flatten())
        meta_str = f"landmarks:{labels_str};"
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("IdentiFly", meta_str)
        img.save(self._filepath, pnginfo=png_info)

        return self._filepath

    @property
    def image(self):
        return cv2.cvtColor(self._image.copy(), cv2.COLOR_BGR2RGB)