# data helper code
from __future__ import annotations

import os
from collections import defaultdict

import PIL
import numpy as np
from PIL import Image
from numba import jit

from .color import rgb_array_to_jzazbz_array, ColorSpaceConversionError
from .logger import get_logger


class ImageLoadingError(Exception):
    pass


class ImageData:
    def __init__(self, compress_dims: Tuple[int] = (300, 300), **kwargs):
        self.rgb_dict = defaultdict(None)
        self.jzazbz_dict = defaultdict(None)
        self.labels_list = []
        self.compress_dims = compress_dims
        self.log = get_logger(__class__.__name__)

    def load_image_dict_from_subfolders(self, path, label=None):
        assert os.path.isdir(path)
        path = os.path.realpath(path)
        folders = os.listdir(path)
        if len(folders) == 0:
            self.log.error(f"No subfolders found {folders}")
        for folder in folders:
            fp = os.path.join(path, folder)
            self.log.info(f"loading from folder {fp}")
            assert os.path.isdir(fp)
            self.load_image_dict_from_folder(
                fp, label=label, compute_jazabz=compute_jzazbz
            )
        self.labels_list = list(self.rgb_dict.keys())

    def load_image_dict_from_folder(self, path, label=None, compute_jzazbz=True):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"the data directory {path} does not exist")
        path = os.path.realpath(path)
        if label is None:
            label = path.split("/")[-1]
        files = os.listdir(path)
        imglist = []
        arraylist = []
        for f in files:
            fp = os.path.join(path, f)
            img = None
            try:
                img = self.load_rgb_image(fp)
            except ValueError as exc:
                self.log.error(f"{exc} error loading rgb image from {fp}")
            if img is not None:
                imglist.append(img)

        self.log.debug(f'loaded {len(imglist)} images for "{label}"')
        self.rgb_dict[label] = imglist
        if compute_jzazbz:
            self.store_jzazbz_from_rgb(label)
        self.labels_list = list(self.rgb_dict.keys())

    def load_image_continuum_from_folder(
        self,
        path: str,
        continuum_files: List[str],
        idx: int = 0,
        window: int = 100,
        label: Optional[str] = None,
        compute_jzazbz: bool = True,
    ) -> None:
        assert os.path.isdir(path)
        path = os.path.realpath(path)
        label = label or path.split("/")[-1]
        imglist = []
        arraylist = []

        files_in_window = continuum_files[idx : idx + window]

        for f in files_in_window:
            fp = os.path.join(path, f)
            img = None
            try:
                img = self.load_rgb_image(fp)
            except ValueError as exc:
                self.log.error(f"{exc} failed to load image {fp}")
            if img is not None:
                imglist.append(img)

        if compute_jzazbz:
            self.store_jzazbz_from_rgb(label)

        self.rgb_dict[label] = imglist
        self.labels_list = list(self.rgb_dict.keys())

    def load_rgb_image(self, path: Union[Path, str]) -> np.ndarray:
        fmts = [".jpg", ".jpeg", ".png", ".bmp"]
        path = str(path)
        if os.path.isfile(path) and any([fmt in path.lower() for fmt in fmts]):
            try:
                img_raw = PIL.Image.open(path)
                if self.compress_dims:
                    assert len(self.compress_dims) == 2
                    img_raw = img_raw.resize(self.compress_dims, PIL.Image.ANTIALIAS)
                img_array = np.array(img_raw)[:, :, :3]

                assert len(img_array.shape) == 3 and img_array.shape[-1] == 3
                return img_array
            except Exception as exc:
                raise ImageLoadingError(f"while loading {path}") from exc

    def store_jzazbz_from_rgb(
        self, labels: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Creates the jzazbz array from the rgb array
        """
        if labels is not None:
            labels = labels if isinstance(labels, list) else [labels]
        else:
            labels = list(self.rgb_dict.keys())
        self.log.debug(f"creating jzazbz arrays from rgb arrays for {labels}")
        for label in labels:
            try:
                self.jzazbz_dict[label] = [
                    rgb_array_to_jzazbz_array(rgb) for rgb in self.rgb_dict[label]
                ]
            except ColorSpaceConversionError as exc:
                raise ColorSpaceConversionError(
                    f"While converting {labels} to jzazbz colorspace"
                )

    def print_labels(self):
        self.labels_list = list(self.rgb_dict.keys())
        print(self.labels_list)
