from __future__ import annotations

import time
import os
import random
import tempfile
from pathlib import Path

import PIL
import numpy as np
import scipy.stats
import matplotlib.colors as mplcolors
from numba import jit

from .color import kl_divergence, js_divergence, color_distribution, avg_rgb, avg_hsv
from .logger import get_logger
from .datahelper import ImageData


class ImageAnalysis:
    def __init__(self, image_data: ImageData):
        assert isinstance(image_data, ImageData)
        self.image_data = image_data
        self.jzazbz_dict = image_data.jzazbz_dict
        self.rgb_dict = image_data.rgb_dict
        self.labels_list = image_data.labels_list
        self.log = get_logger(__class__.__name__)
        self.default_color_params = {
            "spacing": 36,
            "num_bins": 8,
            "num_channels": 3,
            "Jz_min": 0.0,
            "Jz_max": 0.167,
            "Az_min": -0.1,
            "Az_max": 0.11,
            "Bz_min": -0.156,
            "Bz_max": 0.115,
            "h_max": 360,
            "rgb_max": 255,
        }

    def compute_color_distributions(
        self,
        labels=None,
        color_rep=["jzazbz", "hsv", "rgb"],
        spacing=None,
        num_bins=None,
        num_channels=None,
        Jz_min=None,
        Jz_max=None,
        Az_min=None,
        Az_max=None,
        Bz_min=None,
        Bz_max=None,
        h_max=None,
        rgb_max=None,
    ):
        """
        Calculates color distributions for each word in a dictionary
        
        Args:
            self (class instance): ImageAnalysis class instance
            labels (string): if None grabs dictionary keys as labels
            color_rep(array): colorspaces to calculate distributions in

            For the following args, if no value is provided, the instance-shared defaults are used:
            spacing(int): hue spacing for HSV distribution (in degrees)
            num_bins(int): number of bins to calculate 3D distributions in
            num_channels(int): number of color channels
            *z_min (*z_max) (float): minimum (maximum) of JzAzBz coordinates
            h_max (int): maximum hue (in degrees)
            rgb_max (int): maximum value in RGB
        
        Returns:
            self (class instace): ImageAnalysis class instance containing JzAzBz, HSV, and RGB distributions for each word
        """
        assert (
            self.image_data.compress_dims is not None
        ), "Must set compress_dims on ImageData to carry out analysis"
        if labels is None:
            labels = self.labels_list
        labels = labels if isinstance(labels, list) else [labels]
        self.log.debug(f"compute_color_distributions for {labels}")
        self.jzazbz_dist_dict, self.hsv_dist_dict = {}, {}
        self.rgb_ratio_dict, self.rgb_dist_dict = {}, {}
        color_rep = [i.lower() for i in color_rep]

        if "jzazbz" in color_rep:
            self.jzazbz_dist_dict = {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    self.log.warning(f"label {key} does not exist")
                if key not in self.image_data.jzazbz_dict.keys():
                    self.image_data.store_jzazbz_from_rgb(key)
                dist_array = list()
                for i, img_rgb in enumerate(self.rgb_dict[key]):
                    try:
                        jzazbz_dist = color_distribution(
                            img_rgb=img_rgb,
                            colorspace="jzazbz",
                            num_bins=num_bins
                            if num_bins
                            else self.default_color_params["num_bins"],
                            Jz_min=Jz_min
                            if Jz_min
                            else self.default_color_params["Jz_min"],
                            Jz_max=Jz_max
                            if Jz_max
                            else self.default_color_params["Jz_max"],
                            Az_min=Az_min
                            if Az_min
                            else self.default_color_params["Az_min"],
                            Az_max=Az_max
                            if Az_max
                            else self.default_color_params["Az_max"],
                            Bz_min=Bz_min
                            if Bz_min
                            else self.default_color_params["Bz_min"],
                            Bz_max=Bz_max
                            if Bz_max
                            else self.default_color_params["Bz_max"],
                            num_channels=num_channels
                            if num_channels
                            else self.default_color_params["num_channels"],
                        )
                    except RuntimeWarning as exc:
                        failed_image_path = Path(tempfile.NamedTemporaryFile().name)
                        PIL.Image.fromarray(img_rgb, "RGB").save(
                            str(failed_image_path), "png"
                        )
                        self.log.warning(
                            f"{exc}, could not compute jzazbz color distribution for image saved to {failed_image_path}, skipping image {i}/{len(self.rgb_dict[key])}"
                        )
                        continue
                    if True in np.isnan(jzazbz_dist):
                        self.log.warning(f"Dropping jzazbz_dist with NaN for {key}")
                        continue
                    dist_array.append(jzazbz_dist)
                self.jzazbz_dist_dict[key] = dist_array

        if "hsv" in color_rep:
            self.h_dict, self.s_dict, self.v_dict = {}, {}, {}
            self.hsv_dist_dict = {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    self.log.warning(f"label {key} does not exist")
                dist_array, h, s, v = list(), list(), list(), list()
                for img_rgb in self.rgb_dict[key]:
                    dist = color_distribution(
                        img_rgb=img_rgb,
                        colorspace="hsv",
                        rgb_max=rgb_max
                        if rgb_max
                        else self.default_color_params["rgb_max"],
                        spacing=spacing
                        if spacing
                        else self.default_color_params["spacing"],
                        h_max=h_max if h_max else self.default_color_params["h_max"],
                    )
                    dist_array.append(dist)
                    h_temp, s_temp, v_temp = avg_hsv(hsv_img)
                    h.append(h_temp)
                    s.append(s_temp)
                    v.append(v_temp)
                self.hsv_dist_dict[key] = dist_array
                self.h_dict[key], self.s_dict[key], self.v_dict[key] = h, s, v

        if "rgb" in color_rep:
            self.rgb_ratio_dict, self.rgb_dist_dict = {}, {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    self.log.warning(f"label {key} does not exist")
                    continue
                rgb = list()
                dist_array = list()
                for i, img_rgb in enumerate(self.rgb_dict[key]):
                    try:
                        rgb_tuple = avg_rgb(img_rgb)
                        if True in np.isnan(rgb_tuple):
                            self.log.warning(f"Dropping rgb_tuple with NaN for {key}")
                        else:
                            rgb.append(rgb_tuple)
                    except RuntimeWarning as exc:
                        failed_image_path = Path(tempfile.NamedTemporaryFile().name)
                        PIL.Image.fromarray(img_rgb, "RGB").save(
                            str(failed_image_path), "png"
                        )
                        self.log.warning(
                            f"{exc}, could not compute rgb average for image saved to {failed_image_path}, skipping image {i}/{len(self.rgb_dict[key])}"
                        )
                    try:
                        dist = color_distribution(
                            img_rgb=img_rgb,
                            colorspace="rgb",
                            rgb_max=rgb_max
                            if rgb_max
                            else self.default_color_params["rgb_max"],
                            num_bins=num_bins
                            if num_bins
                            else self.default_color_params["num_bins"],
                            num_channels=num_channels
                            if num_channels
                            else self.default_color_params["num_channels"],
                        )
                    except RuntimeWarning as exc:
                        failed_image_path = Path(tempfile.NamedTemporaryFile().name)
                        PIL.Image.fromarray(img_rgb, "RGB").save(
                            str(failed_image_path), "png"
                        )
                        self.log.warning(
                            f"{exc}, could not compute rgb color distribution for image saved to {failed_image_path}, skipping image {i}/{len(self.rgb_dict[key])}"
                        )
                        continue
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
        self.log.info("performing entropy calculations")

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
                        np.mean(np.array(jzazbz_dist_dict[word2]), axis=0),
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
        if labels is None:
            labels = self.labels_list
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


def merge_vectors_to_image_analysis(vectors: List[WordToColorVector]) -> ImageAnalysis:
    """ Take a list of WordToColorVector objects and return an image analysis object combining each of their data """

    log = get_logger("merge_image_analysis")

    initial_vector = vectors[0]

    merged_image_data = initial_vector.image_data

    for vector in vectors[1:]:
        label = vector.label
        merged_image_data.labels_list.append(label)
        merged_image_data.rgb_dict[label] = vector.image_data.rgb_dict[label]
        merged_image_data.jzazbz_dict[label] = vector.image_data.jzazbz_dict[label]

    log.info(f"merged ImageData from {len(vectors)} WordToColorVector objects")

    image_analysis = ImageAnalysis(merged_image_data)

    image_analysis.compute_color_distributions(color_rep=["jzazbz", "rgb"])
    image_analysis.get_composite_image()

    return image_analysis
