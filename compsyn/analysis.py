# analysis code

import time
import os
import random

import PIL
import numpy as np
import scipy.stats
import matplotlib.colors as mplcolors
from numba import jit

from .logger import get_logger


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


class ImageAnalysis:
    def __init__(self, image_data):
        # assert isinstance(image_data, compsyn.ImageData)
        self.image_data = image_data
        self.jzazbz_dict = image_data.jzazbz_dict
        self.rgb_dict = image_data.rgb_dict
        self.labels_list = image_data.labels_list
        self.log = get_logger(__class__.__name__)

    def compute_color_distributions(
        self,
        labels="default",
        color_rep=["jzazbz", "hsv", "rgb"],
        spacing=36,
        num_bins=8,
        num_channels=3,
        Jz_min=0.0,
        Jz_max=0.167,
        Az_min=-0.1,
        Az_max=0.11,
        Bz_min=-0.156,
        Bz_max=0.115,
        h_max=360,
        rgb_max=255,
    ):
        """
        Calculates color distributions for each word in a dictionary
        
        Args:
            self (class instance): ImageAnalysis class instance
            labels (string): if "default" grabs dictionary keys as labels
            color_rep(array): colorspaces to calculate distributions in
            spacing(int): hue spacing for HSV distribution (in degrees)
            num_bins(int): number of bins to calculate 3D distributions in
            num_channels(int): number of color channels
            *z_min (*z_max) (float): minimum (maximum) of JzAzBz coordinates
            h_max (int): maximum hue (in degrees)
            rgb_max (int): maximum value in RGB
        
        Returns:
            self (class instace): ImageAnalysis class instance containing JzAzBz, HSV, and RGB distributions for each word
        """
        dims = self.image_data.dims
        if labels == "default":
            labels = self.labels_list
        labels = labels if isinstance(labels, list) else [labels]
        self.jzazbz_dist_dict, self.hsv_dist_dict = {}, {}
        self.rgb_ratio_dict, self.rgb_dist_dict = {}, {}
        color_rep = [i.lower() for i in color_rep]

        if "jzazbz" in color_rep:
            self.jzazbz_dist_dict = {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    self.log.warning(f"label {key} does not exist")
                    continue
                if key not in self.image_data.jzazbz_dict.keys():
                    self.image_data.store_jzazbz_from_rgb(key)
                jzazbz, dist_array = [], []
                imageset = self.jzazbz_dict[key]
                for i in range(len(imageset)):
                    jzazbz.append(imageset[i])
                    dist = np.ravel(
                        np.histogramdd(
                            np.reshape(
                                imageset[i][:, :, :], (dims[0] * dims[1], num_channels)
                            ),
                            bins=(
                                np.linspace(
                                    Jz_min,
                                    Jz_max,
                                    1 + int(num_bins ** (1.0 / num_channels)),
                                ),
                                np.linspace(
                                    Az_min,
                                    Az_max,
                                    1 + int(num_bins ** (1.0 / num_channels)),
                                ),
                                np.linspace(
                                    Bz_min,
                                    Bz_max,
                                    1 + int(num_bins ** (1.0 / num_channels)),
                                ),
                            ),
                            density=True,
                        )[0]
                    )
                    if True in np.isnan(dist):
                        # Drop any dists that contain NaN
                        continue
                    dist_array.append(dist)
                self.jzazbz_dist_dict[key] = dist_array

        if "hsv" in color_rep:
            self.h_dict, self.s_dict, self.v_dict = {}, {}, {}
            self.hsv_dist_dict = {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    self.log.warning(f"label {key} does not exist")
                    continue
                imageset = self.rgb_ratio_dict[key]
                dist_array, h, s, v = [], [], [], []
                for i in range(len(imageset)):
                    hsv_array = mplcolors.rgb_to_hsv(imageset[i] / (1.0 * rgb_max))
                    dist = np.histogram(
                        1.0 * h_max * np.ravel(hsv_array[:, :, 0]),
                        bins=np.arange(0, h_max + spacing, spacing),
                        density=True,
                    )[0]
                    dist_array.append(dist)
                    h.append(np.mean(np.ravel(hsv_array[:, :, 0])))
                    s.append(np.mean(np.ravel(hsv_array[:, :, 1])))
                    v.append(np.mean(np.ravel(hsv_array[:, :, 2])))
                self.hsv_dist_dict[key] = dist_array
                self.h_dict[key], self.s_dict[key], self.v_dict[key] = h, s, v

        if "rgb" in color_rep:
            self.rgb_ratio_dict, self.rgb_dist_dict = {}, {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    self.log.warning(f"label {key} does not exist")
                    continue
                imageset = self.rgb_dict[key]
                rgb = []
                dist_array = []
                for i in range(len(imageset)):
                    r = np.sum(np.ravel(imageset[i][:, :, 0]))
                    g = np.sum(np.ravel(imageset[i][:, :, 1]))
                    b = np.sum(np.ravel(imageset[i][:, :, 2]))
                    tot = 1.0 * r + g + b
                    rgb.append([r / tot, g / tot, b / tot])
                    dist = np.ravel(
                        np.histogramdd(
                            np.reshape(imageset[i], (dims[0] * dims[1], num_channels)),
                            bins=(
                                np.linspace(
                                    0,
                                    rgb_max,
                                    1 + int(num_bins ** (1.0 / num_channels)),
                                ),
                                np.linspace(
                                    0,
                                    rgb_max,
                                    1 + int(num_bins ** (1.0 / num_channels)),
                                ),
                                np.linspace(
                                    0,
                                    rgb_max,
                                    1 + int(num_bins ** (1.0 / num_channels)),
                                ),
                            ),
                            density=True,
                        )[0]
                    )
                    dist_array.append(dist)
                self.rgb_ratio_dict[key] = rgb
                self.rgb_dist_dict[key] = dist_array

    def entropy_computations(
        self,
        between_labels=True,
        between_images=True,
        between_all_images=False,
        symmetrized=True,
    ):
        """
        Performs KL and JS divergence computations between aggregate color distributions for label pairs,
        between color distributions for image pairs (for a given label), and between color distributions
        for all image pairs (between all labels)
    
        Args:
            self (class instance): ImageAnalysis class instance
            between_labels (Boolean): Whether to calculate cross entropy between labels (default True)
            between_images (Boolean): Whether to calculate cross entropy between images for each label (default True)
            between_all_images (Boolean): Whether to calculate cross entropy between images for all labels (default False)
    
        Returns:
            (Default)
            self.cross_entropy_between_labels_dict (dictionary): dictionary of KL divergence values for each pair of words in JzAzBz
            self.cross_entropy_between_labels_matrix (array of arrays): matrix of KL divergence values for each pair of words in JzAzBz
            self.cross_entropy_between_labels_dict_js (dictionary): dictionary of JS divergence values for each pair of words in JzAzBz
            self.cross_entropy_between_labels_matrix_js (arraay of arrays): matrix of JS divergence values for each pair of words in JzAzBz
            self.cross_entropy_between_images_dict (dictionary): dictionary of KL divergence values between images for each label in JzAzBz
            self.cross_entropy_between_images_dict_js (dictionary): dictionary of JS divergence values between images for each label in JzAzBz
            (Optional)
            self.cross_entropy_between_all_images_dict (dictionary): dictionary of JS divergence values between all images for all labels in JzAzBz
            self.cross_entropy_between_all_images_matrix (arraay of arrays): : matrix of JS divergence values between all images for all labels in JzAzBz
        """
        jzazbz_dist_dict = self.jzazbz_dist_dict

        if between_labels:
            words = self.labels_list
            labels_entropy_dict = {}
            labels_entropy_dict_js = {}
            color_sym_matrix = []
            color_sym_matrix_js = []

            for word1 in words:
                row = []
                row_js = []
                for word2 in words:
                    entropy_js = js_divergence(
                        np.mean(np.array(jzazbz_dist_dict[word1]), axis=0),
                        np.mean(np.array(jzazbz_dist_dict[word2]), axis=0),
                    )
                    entropy = kl_divergence(
                        np.mean(np.array(jzazbz_dist_dict[word1]), axis=0),
                        np.mean(np.array(jzazbz_dist_dict[word1]), axis=0),
                        symmetrized,
                    )
                    row.append(entropy)
                    row_js.append(entropy_js)
                    # these lines are for convenience; if strings are correctly synced across all data they are not needed
                    if word1 == "computer science":
                        labels_entropy_dict["computer_science" + "_" + word2] = entropy
                        labels_entropy_dict_js[
                            "computer_science" + "_" + word2
                        ] = entropy_js
                    elif word2 == "computer science":
                        labels_entropy_dict[word1 + "_" + "computer_science"] = entropy
                        labels_entropy_dict_js[
                            word1 + "_" + "computer_science"
                        ] = entropy_js
                    else:
                        labels_entropy_dict[word1 + "_" + word2] = entropy
                        labels_entropy_dict_js[word1 + "_" + word2] = entropy_js
                color_sym_matrix.append(row)
                color_sym_matrix_js.append(row_js)

            self.cross_entropy_between_labels_dict = labels_entropy_dict
            self.cross_entropy_between_labels_matrix = color_sym_matrix
            self.cross_entropy_between_labels_dict_js = labels_entropy_dict_js
            self.cross_entropy_between_labels_matrix_js = color_sym_matrix_js

        if between_images:
            entropy_dict = {}
            entropy_dict_js = {}
            for key in jzazbz_dist_dict:
                entropy_array = []
                entropy_array_js = []
                for i in range(len(jzazbz_dist_dict[key])):
                    for j in range(len(jzazbz_dist_dict[key])):
                        entropy_array_js.append(
                            js_divergence(
                                jzazbz_dist_dict[key][i], jzazbz_dist_dict[key][j]
                            )
                        )
                        entropy_array.append(
                            kl_divergence(
                                jzazbz_dist_dict[key][i],
                                jzazbz_dist_dict[key][j],
                                symmetrized,
                            )
                        )
                entropy_dict[key] = entropy_array
                entropy_dict_js[key] = entropy_array_js

            self.cross_entropy_between_images_dict = entropy_dict
            self.cross_entropy_between_images_dict_js = entropy_dict_js

        if between_all_images:
            entropy_dict_all = {}
            color_sym_matrix_js_all = []

            for word1 in words:
                row_js_all = []
                for word2 in words:
                    entropy_js_all = []
                    for i in range(len(jzazbz_dist_dict[word1])):
                        for j in range(len(jzazbz_dist_dict[word2])):
                            try:
                                entropy_js_all.append(
                                    js_divergence(
                                        jzazbz_dist_dict[word1][i],
                                        jzazbz_dist_dict[word2][j],
                                    )
                                )
                            except Exception as exc:
                                self.log.error(exc)
                                entropy_js_all.append(np.mean(entropy_js))
                    entropy_dict_all[word1 + "_" + word2] = entropy_js_all
                    row_js_all.append(np.mean(entropy_js_all))
                color_sym_matrix_js_all.append(row_js_all)

            self.cross_entropy_between_all_images_dict = entropy_dict_all
            self.cross_entropy_between_all_images_matrix = color_sym_matrix_js_all

    def compress_color_data(self):
        """
        Saves mean rgb and jzazbz values
        """
        avg_rgb_vals_dict = {}  # dictionary of average color coordinates
        for label in self.labels_list:
            try:
                avg_rgb = np.mean(
                    np.mean(np.mean(self.jzazbz_dict[label], axis=0), axis=0), axis=0
                )
                avg_rgb_vals_dict[label] = avg_rgb
            except Exception as exc:
                self.log.error(exc)
                self.log.error(label + " failed")
        self.avg_rgb_vals_dict = avg_rgb_vals_dict

        jzazbz_dict_simp = {}
        for label in self.labels_list:
            avg_jzazbz = np.mean(self.jzazbz_dist_dict[label], axis=0)
            jzazbz_dict_simp[label] = avg_jzazbz
        self.jzazbz_dict_simp = jzazbz_dict_simp

    def get_composite_image(
        self,
        labels=None,
        compress_dim=300,
        num_channels=3,
        num_of_images="all",
        sample=False,
        reverse=False,
    ):
        """
        Returns colorgrams for a list of words
    
        Args:
            labels (array): list of words
    
        Returns:
            compressed_img_dict (dictionary of arrays): dictionary of mean rgb values for each pixel for a given word (dictionary keys are words)
        """
        compressed_img_dict = {}
        img_data = self.image_data.rgb_dict
        if not labels:
            labels = img_data.keys()
        for label in labels:
            self.log.info(label + " is being compressed.")
            total_images = len(img_data[label])
            if num_of_images == "all":
                vectors = img_data[label]
            elif type(num_of_images) == int:
                vectors = img_data[label]
                if sample:
                    vectors = random.sample(vectors, num_of_images)
                if reverse:
                    vectors.reverse()
                vectors = vectors[0:num_of_images]

            compressed_img_dict[label] = np.zeros(
                (compress_dim, compress_dim, num_channels)
            )
            compressed_img_dict[label] = np.sum(vectors, axis=0) / (1.0 * len(vectors))

        self.compressed_img_dict = compressed_img_dict
        return compressed_img_dict

    def save_colorgram_to_disk(self):
        if not os.path.exists("colorgrams"):
            os.makedirs("colorgrams")

        if len(self.compressed_img_dict) > 0:
            for img in self.compressed_img_dict:
                colorgram = PIL.Image.fromarray(
                    self.compressed_img_dict[img].astype(np.uint8)
                )
                colorgram.save(os.path.join("colorgrams", img + "_colorgram.png"))
