#!/usr/bin/env python3
import time
import warnings

import numpy as np
from PIL import Image
from kymatio.numpy import Scattering2D

from .datahelper import rgb_array_to_jzazbz_array
from .logger import get_logger


def get_coefficents(L: int, J: int) -> np.ndarray:
    jj1 = []
    index = []
    jj2 = []
    ll1 = []
    ll2 = []
    for j1 in range(J - 1):
        for j2 in range(j1 + 1, J):
            for l1 in range(L):
                for l2 in range(L):
                    coeff_index = (
                        l1 * L * (J - j1 - 1)
                        + l2
                        + L * (j2 - j1 - 1)
                        + (L ** 2) * (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                    )
                    index.append(coeff_index)
                    jj1.append(j1)
                    jj2.append(j2)
                    ll1.append(l1)
                    ll2.append(l2)
    return np.array(index), np.array(jj1), np.array(jj2), np.array(ll1), np.array(ll2)


def perform_scattering_transform(im: np.ndarray, L: int = 4, J: int = 5) -> np.ndarray:
    """Performs scattering transform on single channel  image array
	In: im (numpy ndarray (128x128))
	Out: scattering coefficient (list)"""
    S2D = Scattering2D(J=J, shape=im.shape, L=L, max_order=2)
    scattering_coefficients = S2D.scattering(im.astype(np.float32) / 255.0)
    scattering_coefficients = np.array(scattering_coefficients)
    sc = []
    scattering_coefficients = (
        np.sum(scattering_coefficients, axis=(1, 2)) / scattering_coefficients.shape[1]
    )
    sc.append(scattering_coefficients[0])
    sc1 = scattering_coefficients[1 : J * L + 1]
    sc2 = scattering_coefficients[J * L + 1 :]
    index, j1, j2, l1, l2 = get_coefficents(L, J)
    a = []
    b = []
    c = []
    sc_temp = []
    for i in range(int(len(sc1) / L)):
        sc.append(np.mean(sc1[L * i : L * (i + 1)]))
    for i in range(int(len(sc2) / L)):
        sc_temp.append(np.mean(sc2[L * i : L * (i + 1)]))
        a.extend(np.unique(l1[index][i * L : (i + 1) * L]))
        b.extend(np.unique(j1[index][i * L : (i + 1) * L]))
        c.extend(np.unique(j2[index][i * L : (i + 1) * L]))
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    sc_temp = np.array(sc_temp)
    for i in np.unique(b):
        for j in np.unique(c):
            mask1 = b == i
            mask2 = c == j
            mask = mask1 & mask2
            if len(sc_temp[mask]) > 0:
                sc.append(np.mean(sc_temp[mask]))
    return sc


def get_wavelet_embedding(im: Image, mode: str = "JzAzBz") -> np.ndarray:
    """Generates wavelet image embedding 
	In: im (PIL Image)
	    mode (str) = "JzAzBz", "RGB", "Grey" 
	Out: 
	    wavelet vector (numpy ndarray)"""

    with warnings.catch_warnings():  # kymatio / torch is raising deprecation warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        log = get_logger("get_wavelet_embedding")
        log.debug("resizing image to 128x128")
        im = im.resize((128, 128))

        start_time = time.time()
        sc = list()
        if mode.lower() == "grey":
            im = im.convert("L")
            im = np.array(im)
            sc = perform_scattering_transform(im)
        elif mode.lower() == "jzazbz":
            im = np.array(im)[:, :, :3]
            im = rgb_array_to_jzazbz_array(im)
            for channel in [0, 1, 2]:
                im_channel = im[:, :, channel]
                sc.extend(perform_scattering_transform(im_channel))
        elif mode.upper() == "RGB":
            sc = list()
            im = np.array(im)[:, :, :3]
            for channel in [0, 1, 2]:
                im_channel = im[:, :, channel]
                sc.extend(perform_scattering_transform(im_channel))
        else:
            raise ValueError(f"unknown mode '{mode}'")

        log.debug(
            f"computed {mode} wavelet embedding in {round(time.time() - start_time, 2)} seconds"
        )
        return np.array(sc)
