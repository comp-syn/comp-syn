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
from .browser import get_browser_args
from .helperfunctions import search_and_download
from .logger import get_logger


class WordToColorVector(Vector):
    # Generated from a set of images' average color
    def __init__(self, number_of_images: Optional[int] = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self._attributes_available: bool = False
        self.number_of_images = number_of_images
        self.log = get_logger(self.__class__.__name__)
        self.log.info(f"local downloads: {self.raw_images_path}")

    @property
    def raw_images_path(self) -> Path:
        return (
            Path(self.trial.work_dir)
            .joinpath(f"{self.trial.experiment_name}/raw-images")
            .joinpath(self.label)
            .joinpath(self.trial.trial_id)
            .joinpath(self.trial.hostname)
            .joinpath(self.trial.trial_timestamp)
        )

    def validate_analysis(self) -> None:
        """
            Check that the vector has the correct attributes after running analysis
        """
        import IPython; IPython.embed()


    def run_analysis(self, **kwargs) -> None:

        img_object = ImageData()
        img_object.load_image_dict_from_folder(self.raw_images_path, **kwargs)
        self.img_analysis = ImageAnalysis(img_object)
        self.img_analysis.compute_color_distributions(self.label, ["jzazbz", "rgb"])
        self.img_analysis.get_composite_image()

        self.validate_analysis()

        self.jzazbz_vector = np.mean(self.img_analysis.jzazbz_dict[self.label], axis=0)
        self.jzazbz_composite_dists = self.img_analysis.jzazbz_dict[self.label]
        self.jzazbz_dist = np.mean(self.jzazbz_composite_dists, axis=0)

        self.jzazbz_dist_std = np.std(self.jzazbz_composite_dists, axis=0)

        self.rgb_vector = np.mean(self.img_analysis.rgb_dict[self.label], axis=0)
        self.rgb_dist = np.mean(self.img_analysis.rgb_dist_dict[self.label], axis=0)
        self.rgb_ratio = np.mean(self.img_analysis.rgb_ratio_dict[self.label], axis=0)

        self.colorgram_vector = self.img_analysis.compressed_img_dict[self.label]

        self.colorgram = PIL.Image.fromarray(self.colorgram_vector.astype(np.uint8))


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
        driver_options: Optional[List[str]] = None,
    ) -> List[str]:
        """ Gather images from Google Images """
        if driver_options is None:
            driver_options = ["--headless"]

        browser_args, unknown = get_browser_args().parse_known_args()

        urls = search_and_download(
            search_term=self.label,
            driver_browser=browser_args.driver_browser,
            driver_executable_path=browser_args.driver_path,
            driver_options=driver_options,
            target_path=self.raw_images_path,
            number_images=100,
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
