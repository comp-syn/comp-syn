from __future__ import annotations
import os

import numpy as np
import scipy.stats
import matplotlib.colors as mplcolors
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

from .logger import get_logger
from .jzazbz import get_jzazbz_array


def kl_divergence(dist1, dist2, symmetrized=True):
    """
    Calculates Kullback-Leibler (KL) divergence between two distributions, with an option for symmetrization

    Args:
        dist1 (array): first distribution
        dist2 (array): second distribution
        symmetrized (Boolean): flag that defaults to symmetrized KL divergence, and returns non-symmetrized version if False

    Returns:
        kl (float): (symmetrized) KL divergence
    """
    if symmetrized == True:
        kl = (
            scipy.stats.entropy(dist1, dist2) + scipy.stats.entropy(dist2, dist1)
        ) / 2.0
        return kl
    else:
        kl = scipy.stats.entropy(dist1, dist2)
        return kl


def js_divergence(dist1, dist2):
    """
    Calculates Jensen-Shannon (JS) divergence between two distributions

    Args:
        dist1 (array): first distribution
        dist2 (array): second distribution

    Returns:
        js (float): JS divergence
    """
    mean_dist = (dist1 + dist2) / 2.0
    js = (
        scipy.stats.entropy(dist1, mean_dist) + scipy.stats.entropy(dist2, mean_dist)
    ) / 2.0
    return js


class ColorSpaceConversionError(Exception):
    pass


def rgb_array_to_jzazbz_array(rgb_array: np.ndarray) -> np.ndarray:
    """
    Converts rgb pixel values to JzAzBz pixel values

    Args:
        rgb_array (array): matrix of rgb pixel values

    Returns:
        jzazbz_array (array): matrix of JzAzBz pixel values
    """

    r = rgb_array[:, :, 0].reshape([-1])
    g = rgb_array[:, :, 1].reshape([-1])
    b = rgb_array[:, :, 2].reshape([-1])

    jzazbz_array_npy = get_jzazbz_array()
    try:
        jzazbz_vals = jzazbz_array_npy[r, g, b]
    except IndexError as exc:
        raise ColorSpaceConversionError(
            f"Input shape: {rgb_array.shape}. r:{r}, g:{g}, b:{b}. Input: {rgb_array}"
        ) from exc

    jzazbz_array = jzazbz_vals.reshape(list(rgb_array.shape[:3])).transpose([0, 1, 2])
    return jzazbz_array


def bin_img(img, num_bins, x_min, x_max, y_min, y_max, z_min, z_max, num_channels=3):
    """
    Calculates the distribution of rgb or JzAzBz pixel values in an image

    Args:
        img (array): matrix of rgb or jzazbz pixel values
        num_bins (int): total number of colorspace subvolumes
        {xyz}_min, {xyz}_max (floats): minimum and maximum coordinates of each colorspace dimension
        num_channels (int): number of color channels

    Returns:
        dist (array): distribution of of rgb JzAzBz pixel values in specified bins
    """
    dist = np.ravel(
        np.histogramdd(
            np.reshape(img[:, :, :], (img.shape[0] * img.shape[1], num_channels,),),
            bins=(
                np.linspace(x_min, x_max, 1 + int(num_bins ** (1.0 / num_channels)),),
                np.linspace(y_min, y_max, 1 + int(num_bins ** (1.0 / num_channels)),),
                np.linspace(z_min, z_max, 1 + int(num_bins ** (1.0 / num_channels)),),
            ),
            density=True,
        )[0]
    )
    return dist


def bin_hsv(img_hsv, spacing, h_max=360):
    """
    Calculates the distribution of hue values in an image
    ​
    Args:
        img_hsv (array): array of hue values
        spacing (int): degrees of rotation subtended by each hsv bin
        h_max (int): maximum hue rotation angle
    ​
    Returns:
        dist (array): distribution of of rgb JzAzBz pixel values in specified bins
    """
    dist = np.histogram(
        1.0 * h_max * np.ravel(img_hsv[:, :, 0]),
        bins=np.arange(0, h_max + spacing, spacing),
        density=True,
    )[0]
    return dist


class UnknownColorSpaceError(Exception):
    pass


class MissingArgumentError(Exception):
    pass


def color_distribution(
    img_rgb: np.ndarray,
    colorspace: str,
    spacing: Optional[int] = None,
    num_bins: Optional[int] = None,
    num_channels: Optional[int] = None,
    Jz_min: Optional[float] = None,
    Jz_max: Optional[float] = None,
    Az_min: Optional[float] = None,
    Az_max: Optional[float] = None,
    Bz_min: Optional[float] = None,
    Bz_max: Optional[float] = None,
    h_max: Optional[int] = None,
    rgb_max: Optional[int] = None,
) -> np.ndarray:
    """
    Calculates color distributions for each word in a dictionary
        
    Args:
        img_rgb (array): RGB image pixel values as loaded from PIL and compressed to (n,n,3)
        colorspace (string): colorspace to calculate distributions in; "jzazbz", "hsv", or "rgb"
        spacing(int): hue spacing for HSV distribution (in degrees)
        num_bins(int): number of bins to calculate 3D distributions in
        num_channels(int): number of color channels
        *z_min (*z_max) (float): minimum (maximum) of JzAzBz coordinates
        h_max (int): maximum hue (in degrees)
        rgb_max (int): maximum value in RGB
        
    Returns:
        {}_dist (array): distribution of values (either jzazbz, hsv, or rgb)
    """

    def _check_required_args(required_args: List[Any]) -> None:
        for i, required in enumerate(required_args):
            try:
                assert required is not None, f"{i}: {required} must be set"
            except AssertionError:
                raise MissingArgumentError(
                    f"Missing one or more required arguments for computing {colorspace} distribution"
                )

    if colorspace == "jzazbz":
        _check_required_args(
            [num_bins, Jz_min, Jz_max, Az_min, Az_max, Bz_min, Bz_max, num_channels]
        )
        img_jzazbz = rgb_array_to_jzazbz_array(img_rgb)
        jzazbz_dist = bin_img(
            img_jzazbz,
            num_bins,
            Jz_min,
            Jz_max,
            Az_min,
            Az_max,
            Bz_min,
            Bz_max,
            num_channels,
        )
        return jzazbz_dist

    elif colorspace == "hsv":
        _check_required_args([rgb_max, h_max, spacing])
        img_hsv = mplcolors.rgb_to_hsv(img_rgb / (1.0 * rgb_max))
        hsv_dist = bin_hsv(img_hsv, spacing, h_max)
        return hsv_dist

    elif colorspace == "rgb":
        _check_required_args([rgb_max, num_bins, num_channels])
        rgb_dist = bin_img(
            img_rgb, num_bins, 0, rgb_max, 0, rgb_max, 0, rgb_max, num_channels
        )
        return rgb_dist

    else:
        raise UnknownColorSpaceError(f"no colorspace '{colorspace}' known")


def avg_rgb(img):
    """
    Calculates the average rgb coordinates of an image
        
    Args:
        img (array): RGB image pixel values as loaded from PIL and compressed to (n,n,3)
        
    Returns:
        avg_rgb (array): 3x1 array containing [average r, average g, average b]
    """

    r = np.sum(np.ravel(img[:, :, 0]))
    g = np.sum(np.ravel(img[:, :, 1]))
    b = np.sum(np.ravel(img[:, :, 2]))
    tot = 1.0 * r + g + b
    return np.array([r / tot, g / tot, b / tot])


def avg_hsv(img):
    """
    Returns the average hue, saturation, value for an image
        
    Args:
        img (array): RGB image pixel values as loaded from PIL and compressed to (n,n,3)
        
    Returns:
        h, s, v (floats): average hue, saturation, value
    """

    img_hsv = mplcolors.rgb_to_hsv(img / (1.0 * rgb_max))
    h = np.mean(np.ravel(img_hsv[:, :, 0]))
    s = np.mean(np.ravel(img_hsv[:, :, 1]))
    v = np.mean(np.ravel(img_hsv[:, :, 2]))
    return h, s, v


# helper function that converts RGB to HEX string
def RGB2HEX(color):
    """In: RGB color array
    Out: HEX string"""

    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# function that selects dominant color from image using kmeans
def get_color(im):
    """In: PIL Image object
    
    Out: Image color
        (string): HEX code 
        (list): RGB value"""
    # selects center portion of image
    im = im.crop((100, 100, 200, 200))

    im = np.array(im)

    modified_image = im.reshape(im.shape[0] * im.shape[1], 3)

    # OPTIONAL: can be modified here to select more than one color
    clf = KMeans(n_clusters=1)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return hex_colors[0], rgb_colors[0]
