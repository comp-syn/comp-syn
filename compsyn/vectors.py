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


class Vector:
    def __init__(self, word: str) -> None:
        self.word = word

    def load_from_folder(self, path: Path, **kwargs) -> Vector:

        img_object = ImageData()
        self.path = Path(path).joinpath(self.word)
        img_object.load_image_dict_from_folder(self.path, **kwargs)
        img_analysis = ImageAnalysis(img_object)
        img_analysis.compute_color_distributions(self.word, ["jzazbz", "rgb"])
        img_analysis.get_composite_image()

        self.jzazbz_vector = np.mean(img_analysis.jzazbz_dict[self.word], axis=0)
        self.jzazbz_composite_dists = img_analysis.jzazbz_dist_dict[self.word]
        self.jzazbz_dist = np.mean(self.jzazbz_composite_dists, axis=0)

        self.jzazbz_dist_std = np.std(self.jzazbz_composite_dists, axis=0)

        self.rgb_vector = np.mean(img_analysis.rgb_dict[self.word], axis=0)
        self.rgb_dist = np.mean(img_analysis.rgb_dist_dict[self.word], axis=0)
        self.rgb_ratio = np.mean(img_analysis.rgb_ratio_dict[self.word], axis=0)

        self.colorgram_vector = img_analysis.compressed_img_dict[self.word]

        self.colorgram = PIL.Image.fromarray(self.colorgram_vector.astype(np.uint8))

        return self

    def print_word_color(self, size: int = 30, color_magnitude: float = 1.65) -> None:

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        plt.text(
            0.35, 0.5, self.word, color=color_magnitude * self.rgb_ratio, fontsize=size
        )
        ax.set_axis_off()
        plt.show()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "query": self.word,
            "jzazbz_vector": self.jzazbz_vector.tolist(),
            "jzazbz_dist": self.jzazbz_dist.tolist(),
            "jzazbz_dist_std": self.jzazbz_dist_std.tolist(),
            "jzazbz_composite_dists": [
                dist.tolist() for dist in self.jzazbz_composite_dists
            ],
            "rgb_vector": self.rgb_vector.tolist(),
            "rgb_dist": self.rgb_dist.tolist(),
            "rgb_ratio": self.rgb_ratio,
            "colorgram_vector": self.colorgram_vector,
        }


class LoadVectorsFromDisk:
    def __init__(self, path: Union[str, Path], default: bool = True) -> None:

        self.vectors = {}
        self.path = Path(path)
        if default:
            self.load_distributions()

    def load_distributions(self, distributions="concreteness-color-embeddings.json") -> None:

        for vector_data in json.loads(self.path.joinpath(distributions).read_text()):
            word = vector_data["query"]

            word_vector = Vector(word)
            word_vector.rgb_dist = vector_data["rgb_dist"]
            word_vector.jzazbz_dist = vector_data["jzazbz_dist"]
            word_vector.jzazbz_dist_std = vector_data["jzazbz_dist_std"]

            self.vectors[word] = word_vector

    def load_info_from_all_colorgrams(
        self, compress_dims: Tuple[int, int] = (300, 300)
    ) -> None:

        for filename in os.listdir(self.path.joinpath("colorgrams")):
            if filename.endswith(".png"):
                q, word = filename.split("=")
                word, png = word.split(".")
                img_raw = PIL.Image.open(
                    self.path.joinpath(f"colorgrams/query={word}.png")
                )
                img_raw = img_raw.resize(
                    (compress_dims[0], compress_dims[1]), PIL.Image.ANTIALIAS
                )
                img_array = np.array(img_raw)[:, :, :3]

                self.vectors[word].rgb_vector = img_array
                self.vectors[word].jzazbz_vector = rgb_array_to_jzazbz_array(img_array)

    def load_info_from_colorgram(
        self, word: str, compress_dims: Tuple[int, int] = (300, 300)
    ) -> None:

        img_raw = PIL.Image.open(self.path.joinpath(f"colorgrams/query={word}.png"))
        img_raw = img_raw.resize(
            (compress_dims[0], compress_dims[1]), PIL.Image.ANTIALIAS
        )
        img_array = np.array(img_raw)[:, :, :3]

        self.vectors[word].rgb_vector = img_array
        self.vectors[word].jzazbz_vector = rgb_array_to_jzazbz_array(img_array)

    def load_colorgram(
        self, word: str, compress_dims: Tuple[int, int] = (300, 300)
    ) -> None:

        img_raw = PIL.Image.open(self.path.joinpath(f"colorgrams/query={word}.png"))
        img_raw = img_raw.resize(
            (compress_dims[0], compress_dims[1]), PIL.Image.ANTIALIAS
        )
        img_array = np.array(img_raw)[:, :, :3]

        self.vectors[word].colorgram_vector = img_array
        self.vectors[word].colorgram = img_raw

    def load_all_colorgrams(self, compress_dims: Tuple[int, int] = (300, 300)) -> None:

        for filename in os.listdir(self.path.joinpath("colorgrams")):
            if filename.endswith(".png"):
                q, word = filename.split("=")
                word, png = word.split(".")
                img_raw = PIL.Image.open(
                    self.path.joinpath(f"colorgrams/query={word}.png")
                )
                img_raw = img_raw.resize(
                    (compress_dims[0], compress_dims[1]), PIL.Image.ANTIALIAS
                )
                img_array = np.array(img_raw)[:, :, :3]

                self.vectors[word].colorgram_vector = img_array
                self.vectors[word].colorgram = img_raw

    def load_concreteness_data(self) -> None:

        with open(
            self.path.joinpath("Concreteness_ratings.csv")
        ) as csvfile:
            ratings = csv.reader(csvfile)
            for row in ratings:
                (
                    region,
                    search_term,
                    Bigram,
                    Conc_M,
                    Conc_SD,
                    Unknown,
                    Total,
                    Percent_known,
                    SUBTLEX,
                    Dom_Pos,
                ) = row
                try:
                    self.vectors[search_term].concreteness_mean = float(Conc_M)
                    self.vectors[search_term].concreteness_sd = float(Conc_SD)
                except (KeyError, ValueError) as e:
                    continue
