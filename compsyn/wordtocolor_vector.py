from __future__ import annotations
import os
import json
import csv
from pathlib import Path

import PIL
import numpy as np
import matplotlib.pyplot as plt

from .datahelper import ImageData, rgb_array_to_jzazbz_array
from .analysis import ImageAnalysis
from .vector import Vector


class WordToColorVector(Vector):
    # Generated from a set of images' average color
    def __init__(self, **kwargs) -> None:
        super(**kwargs)
        self._attributes_available: bool = False

    @property
    def raw_images_path(self) -> Path:
        return (
            Path(f"{self.trial.experiment_name}/raw-images")
            .joinpath(self.label)
            .joinpath(self.trial.trial_id)
            .joinpath(self.trial.hostname)
            .joinpath(self.trial.trial_timestamp)
        )

    def create_embedding(self) -> WordToColorVector:

        img_object = ImageData()
        img_object.load_image_dict_from_folder(self.raw_images_path, **kwargs)
        img_analysis = ImageAnalysis(img_object)
        img_analysis.compute_color_distributions(self.label, ["jzazbz", "rgb"])
        img_analysis.get_composite_image()

        self.jzazbz_vector = np.mean(img_analysis.jzazbz_dict[self.label], axis=0)
        self.jzazbz_composite_dists = img_analysis.jzazbz_dist_dict[self.label]
        self.jzazbz_dist = np.mean(self.jzazbz_composite_dists, axis=0)

        self.jzazbz_dist_std = np.std(self.jzazbz_composite_dists, axis=0)

        self.rgb_vector = np.mean(img_analysis.rgb_dict[self.label], axis=0)
        self.rgb_dist = np.mean(img_analysis.rgb_dist_dict[self.label], axis=0)
        self.rgb_ratio = np.mean(img_analysis.rgb_ratio_dict[self.label], axis=0)

        self.colorgram_vector = img_analysis.compressed_img_dict[self.label]

        self.colorgram = PIL.Image.fromarray(self.colorgram_vector.astype(np.uint8))

        return self

    def print_word_color(self, size: int = 30, color_magnitude: float = 1.65) -> None:

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        plt.text(
            0.35, 0.5, self.label, color=color_magnitude * self.rgb_ratio, fontsize=size
        )
        ax.set_axis_off()
        plt.show()

    def run_image_capture(
        self,
        driver_browser: str,
        driver_path: str,
        driver_options: Optional[List[str]] = None,
    ) -> List[str]:
        """ Gather images from Google Images """
        if driver_options is None:
            driver_options = ["--headless"]

        urls = search_and_download(
            search_term=self.label,
            driver_browser=driver_browser,
            driver_executable_path=driver_path,
            driver_options=driver_options,
            target_path=self.raw_images_path,
            number_images=number_images,
        )

        return urls

    def push(self, include_raw_data: bool = False, **kwargs) -> None:
        super().push(**kwargs)
        if include_raw_data:
            # push raw images
            pass

    def pull(self, include_raw_data: bool = False, **kwargs) -> None:
        super().pull(**kwargs)
        if include_raw_data:
            # pull raw images
            pass
