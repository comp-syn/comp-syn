from __future__ import annotations

import copy
import csv
import json
import os
import time
import shutil
from functools import partial
from pathlib import Path
from multiprocessing.pool import ThreadPool

import PIL
import numpy as np
import matplotlib.pyplot as plt
import qloader

from .config import CompsynConfig
from .datahelper import ImageData, rgb_array_to_jzazbz_array
from .analysis import ImageAnalysis
from .vector import Vector
from .logger import get_logger
from .s3 import upload_file_to_s3, download_file_from_s3, list_object_paths_in_s3
from .utils import compress_image
from .texture import get_wavelet_embedding


class WordToColorVector(Vector):
    # Generated from a set of images' average color
    def __init__(self, number_of_images: Optional[int] = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.image_analysis: Union[None, ImageAnalysis] = None
        self.number_of_images = number_of_images
        self.log = get_logger(self.__class__.__name__ + f".{self.label}")
        self.raw_image_urls = None
        self.log.debug(f"local downloads: {self._local_raw_images_path}")

        if "language" not in self.metadata:
            self.metadata["language"] = "en"
        compsyn_config = CompsynConfig().config
        if "browser" not in self.metadata:
            self.metadata["browser"] = compsyn_config["browser"]
        if "driver_path" not in self.metadata:
            self.metadata["driver_path"] = (
                compsyn_config["driver_path"]
                if "driver_path" in compsyn_config
                else None
            )

    def __repr__(self) -> str:
        """ Nice looking representation """
        output = super().__repr__()
        output += "\n\tgenerated data:"
        if not self._local_raw_images_available:
            output += "\n\t\t(no local raw data)"
        else:
            output += f"\n\t\t(raw images available)"

        if self.raw_image_urls is not None:
            output += f"\n\t\t{'raw_image_urls':16s} = {len(self.raw_image_urls)}"

        try:
            rounded_rgb_values = [f"{val:.2e}" for val in self.rgb_dist.tolist()]
            output += f"\n\t\t{'rgb_dist':26s} = {json.dumps([round(val, 6) for val in self.rgb_dist.tolist()])}"
            output += f"\n\t\t{'rgb_dist_std':26s} = {json.dumps([round(val, 6) for val in self.rgb_dist_std.tolist()])}"
            output += f"\n\t\t{'jzazbz_dist':26s} = {json.dumps([round(val, 3) for val in self.jzazbz_dist.tolist()])}"
            output += f"\n\t\t{'jzazbz_dist_std':26s} = {json.dumps([round(val, 3) for val in self.jzazbz_dist_std.tolist()])}"
        except AttributeError:
            pass

        try:
            output += f"\n\t\t{'jzazbz_wavelet_embedding':26s} = {json.dumps([round(val, 3) for val in self.jzazbz_wavelet_embedding.tolist()])}"
        except AttributeError:
            pass
        try:
            output += f"\n\t\t{'rgb_wavelet_embedding':26s} = {json.dumps([round(val, 3) for val in self.rgb_wavelet_embedding.tolist()])}"
        except AttributeError:
            pass
        try:
            output += f"\n\t\t{'grey_wavelet_embedding':26s} = {json.dumps([round(val, 3) for val in self.grey_wavelet_embedding.tolist()])}"
        except AttributeError:
            pass

        return output

    @property
    def _local_raw_images_path(self) -> Path:
        return Path(CompsynConfig().config["work_dir"]).joinpath(self.raw_images_path)

    @property
    def _local_raw_images_available(self) -> bool:
        return not (
            not self._local_raw_images_path.is_dir()
            or len(list(self._local_raw_images_path.iterdir())) == 0
        )

    @property
    def raw_images_path(self) -> Path:
        return (
            Path(f"{self.trial.experiment_name}/raw-images")
            .joinpath(self.label.replace(" ", "_"))
            .joinpath(self.trial.trial_id)
            .joinpath(self.trial.hostname)
            .joinpath(self.trial.trial_timestamp)
        )

    def delete_local_images(self) -> None:
        for img_path in self._local_raw_images_path.iterdir():
            img_path.unlink()

    def run_image_capture(
        self,
        max_items: int = 100,
        extra_query_params: Optional[Dict[str, str]] = None,
        include_related: bool = False,
        overwrite: bool = False,
    ) -> None:
        """ Gather images from Google Images sets the attribute `self.raw_image_urls`"""

        # check if there are already raw images available already
        try:
            raw_images_available = len(list(self._local_raw_images_path.iterdir()))
        except FileNotFoundError:
            raw_images_available = 0

        if raw_images_available > 0 and overwrite:
            for p in self._local_raw_images_path.iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            raw_images_available = 0

        # allow a small failure rate, as a small percentage of downloads will fail
        if raw_images_available >= 0.90 * self.number_of_images:
            self.log.info(f"{raw_images_available} raw images already downloaded")
            if self.raw_image_urls is None:
                self.log.debug(
                    f"raw images are present on disk, but no URLs are known. Perhaps there is a saved object to load urls from"
                )
            return

        self.raw_images_metadata = qloader.run(
            endpoint="google-images",
            query_terms=self.label,
            output_path=self._local_raw_images_path,
            max_items=max_items,
            metadata=self.metadata,
            language=self.metadata["language"],
            browser=self.metadata["browser"],
            driver_path=self.metadata["driver_path"],
            extra_query_params=extra_query_params,
            track_related=include_related,
        )

        if include_related:
            # move related images to the same folder as primary results
            self.log.info(f"flattening related images to main image directory")
            related_img_dir = self._local_raw_images_path.joinpath("related")
            for related_img_path in related_img_dir.iterdir():
                related_img_path.rename(
                    related_img_path.parents[1].joinpath(related_img_path.name)
                )
            related_img_dir.rmdir()

    def load_data(self, compress_dim: int = 300, **kwargs) -> None:
        try:
            self.image_data = ImageData(compress_dims=(compress_dim, compress_dim))
            self.image_data.load_image_dict_from_folder(
                path=self._local_raw_images_path, label=self.label, **kwargs
            )
            self.image_analysis = ImageAnalysis(self.image_data)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No data found to load, run an image capture with run_image_capture"
            )

    def run_analysis(
        self,
        wavelet_modes: Optional[List[str]] = None,
        num_images: Optional[int] = None,
        compress_dim: int = 300,
    ) -> None:

        self.load_data(compress_dim)

        self.image_analysis.compute_color_distributions(self.label, ["jzazbz", "rgb"])
        self.image_analysis.get_composite_image(compress_dim=compress_dim)

        self.jzazbz_vector = np.mean(
            self.image_analysis.jzazbz_dict[self.label], axis=0
        )
        self.jzazbz_composite_dists = self.image_analysis.jzazbz_dist_dict[self.label]
        self.jzazbz_dist = np.mean(self.jzazbz_composite_dists, axis=0)
        self.jzazbz_dist_std = np.std(self.jzazbz_composite_dists, axis=0)

        self.rgb_vector = np.mean(self.image_analysis.rgb_dict[self.label], axis=0)
        self.rgb_composite_dists = self.image_analysis.rgb_dist_dict[self.label]
        self.rgb_dist = np.mean(self.rgb_composite_dists, axis=0)
        self.rgb_dist_std = np.std(self.rgb_composite_dists, axis=0)

        self.rgb_ratio = np.mean(self.image_analysis.rgb_ratio_dict[self.label], axis=0)

        self.colorgram_vector = self.image_analysis.compressed_img_dict[self.label]

        self.colorgram = PIL.Image.fromarray(self.colorgram_vector.astype(np.uint8))

        if wavelet_modes is None:
            wavelet_modes = ["JzAzBz", "RGB", "Grey"]

        for wavelet_mode in wavelet_modes:
            setattr(
                self,
                f"{wavelet_mode.lower()}_wavelet_embedding",
                get_wavelet_embedding(im=self.colorgram, mode=wavelet_mode),
            )

    def print_word_color(self, size: int = 30, color_magnitude: float = 1.65) -> None:

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        plt.text(
            0.35, 0.5, self.label, color=color_magnitude * self.rgb_ratio, fontsize=size
        )
        ax.set_axis_off()
        plt.show()

    def save(self) -> None:
        # clear some of the bulkier analysis data, raw data is still available
        to_be_saved = copy.deepcopy(self)
        did_clean = False
        for del_attr in [
            "image_analysis",
            "image_data",
            "jzazbz_composite_dists",
            "jzazbz_vector",
            "rgb_vector",
            "colorgram_vector",
            "jzazbz_wavelet_embedding",
            "rgb_wavelet_embedding",
            "grey_wavelet_embedding",
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
        try:
            compressed_image_path = compress_image(local_image_path)
        except IsADirectoryError:
            # may have directories in the raw images folder, ignore them.
            return
        upload_file_to_s3(
            local_path=compressed_image_path,
            s3_path=self.raw_images_path.joinpath(local_image_path.name),
            overwrite=overwrite,
        )
        compressed_image_path.unlink()
        compressed_image_path.parent.rmdir()

    def push(
        self, include_raw_images: bool = False, overwrite: bool = False, **kwargs
    ) -> None:
        super().push(**kwargs)
        if include_raw_images:
            # push raw images
            self.log.debug(f"pushing raw images (ovewrite={overwrite})...")
            local_paths = list(self._local_raw_images_path.iterdir())
            start = time.time()
            func = partial(self._threaded_compressed_s3_upload, overwrite=overwrite)
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
            self._local_raw_images_path.mkdir(exist_ok=True, parents=True)
            self.log.debug(f"pulling raw images (ovewrite={overwrite})...")
            s3_paths = list(list_object_paths_in_s3(s3_prefix=self.raw_images_path))
            start = time.time()
            func = partial(self._threaded_s3_download, overwrite=overwrite)
            with ThreadPool(processes=os.getenv("COMPSYN_THREAD_POOL_SIZE", 4)) as pool:
                pool.map(func, s3_paths)
            self.log.info(
                f"pulled {len(s3_paths)} raw images from remote in {int(time.time()-start)} seconds"
            )
