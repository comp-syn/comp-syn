from __future__ import annotations

import copy
import csv
import json
import os
import time
from functools import partial
from pathlib import Path
from multiprocessing.pool import ThreadPool

import PIL
import numpy as np
import matplotlib.pyplot as plt

from .datahelper import ImageData, rgb_array_to_jzazbz_array
from .analysis import ImageAnalysis
from .vector import Vector
from .browser import get_browser_args
from .helperfunctions import search_and_download
from .logger import get_logger
from .s3 import upload_file_to_s3, download_file_from_s3, list_object_paths_in_s3
from .utils import compress_image


class WordToColorVector(Vector):
    # Generated from a set of images' average color
    def __init__(self, number_of_images: Optional[int] = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self._attributes_available: bool = False
        self.number_of_images = number_of_images
        self.log = get_logger(self.__class__.__name__)
        self.log.info(f"local downloads: {self.raw_images_path}")
        self._local_raw_images_path = self.trial.work_dir.joinpath(self.raw_images_path)

    @property
    def raw_images_path(self) -> Path:
        return (
            Path(f"{self.trial.experiment_name}/raw-images")
            .joinpath(self.label)
            .joinpath(self.trial.trial_id)
            .joinpath(self.trial.hostname)
            .joinpath(self.trial.trial_timestamp)
        )

    def run_analysis(self, **kwargs) -> None:

        self.img_data = ImageData()
        self.img_data.load_image_dict_from_folder(
            path=self._local_raw_images_path, label=self.label, **kwargs
        )

        self.img_analysis = ImageAnalysis(self.img_data)
        self.img_analysis.compute_color_distributions(self.label, ["jzazbz", "rgb"])
        self.img_analysis.get_composite_image()

        self.jzazbz_vector = np.mean(self.img_analysis.jzazbz_dict[self.label], axis=0)
        self.jzazbz_composite_dists = self.img_analysis.jzazbz_dist_dict[self.label]
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
        self, driver_options: Optional[List[str]] = None,
    ) -> List[str]:
        """ Gather images from Google Images """

        # check if there are already raw images available already
        try:
            raw_images_available = len(list(self._local_raw_images_path.iterdir()))
        except FileNotFoundError:
            raw_images_available = 0

        if raw_images_available >= 0.80 * self.number_of_images:
            self.log.info(
                f"there are already {raw_images_available} raw images available, skipping image capture"
            )
            return

        if driver_options is None:
            driver_options = ["--headless"]

        browser_args, unknown = get_browser_args().parse_known_args()

        urls = search_and_download(
            search_term=self.label,
            driver_browser=browser_args.driver_browser,
            driver_executable_path=browser_args.driver_path,
            driver_options=driver_options,
            target_path=self._local_raw_images_path,
            number_images=100,
        )

        return urls

    def save(self) -> None:
        # clear some of the bulkier analysis data, raw data is still available
        to_be_saved = copy.deepcopy(self)
        did_clean = False
        for del_attr in [
            "img_analysis",
            "img_data",
            "jzazbz_composite_dists",
            "jzazbz_vector",
            "rgb_vector",
            "colorgram_vector",
        ]:
            if hasattr(to_be_saved, del_attr):
                delattr(to_be_saved, del_attr)
                did_clean = True
        if did_clean:
            to_be_saved.save()
        else:
            super().save()

    def _threaded_compressed_s3_upload(
        self, local_image_path: Path, overwrite: bool = False
    ) -> None:
        compressed_image_path = compress_image(local_image_path)
        upload_file_to_s3(
            local_path=compressed_image_path,
            s3_path=self.raw_images_path.joinpath(local_image_path.name),
            overwrite=overwrite,
        )
        compressed_image_path.unlink()

    def push(
        self, include_raw_images: bool = False, overwrite: bool = False, **kwargs
    ) -> None:
        super().push(**kwargs)
        if include_raw_images:
            # push raw images
            self.log.info(f"pushing raw images (ovewrite={overwrite})...")
            local_paths = list(self._local_raw_images_path.iterdir())
            start = time.time()
            func = partial(self._threaded_compressed_s3_upload, overwrite=overwrite)
            # Performance on 12 core machine:
            # 1 process     = 96 seconds
            # 4 processes   = 30 seconds
            # 5 processes   = 26 seconds
            # 10 processes  = 21 seconds
            # 100 processes = 19 seconds
            with ThreadPool(processes=os.getenv("COMPSYN_THREAD_POOL_SIZE", 4)) as pool:
                pool.map(func, local_paths)
            self.log.info(
                f"pushed {len(local_paths)} raw images to remote in {int(time.time()-start)} seconds"
            )

    def _threaded_s3_download(self, s3_path: Path, overwrite: bool = False) -> None:
        download_file_from_s3(
            local_path=self._local_raw_images_path.joinpath(s3_path.name),
            s3_path=s3_path,
            overwrite=overwrite,
        )

    def pull(
        self, include_raw_images: bool = False, overwrite: bool = False, **kwargs
    ) -> None:
        super().pull(**kwargs)
        if include_raw_images:
            # pull raw images
            self.log.info(f"pushing raw images (ovewrite={overwrite})...")
            s3_paths = list(list_object_paths_in_s3(s3_prefix=self.raw_images_path))
            start = time.time()
            func = partial(self._threaded_s3_download, overwrite=overwrite)
            with ThreadPool(processes=os.getenv("COMPSYN_THREAD_POOL_SIZE", 4)) as pool:
                pool.map(func, s3_paths)
            self.log.info(
                f"pushed {len(s3_paths)} raw images from remote in {int(time.time()-start)} seconds"
            )
