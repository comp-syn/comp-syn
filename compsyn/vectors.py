from __future__ import annotations
from compsyn import datahelper, analysis, visualisation, vectors
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path


class Vector:
    def __init__(self, word: str) -> None:
        self.word = word

    def load_from_folder(self, path: Path) -> None:

        img_object = datahelper.ImageData()
        img_object.load_image_dict_from_folder(path.joinpath(self.word))
        img_analysis = analysis.ImageAnalysis(img_object)
        img_analysis.compute_color_distributions(self.word, ["jzazbz", "rgb"])
        img_analysis.get_composite_image()

        self.jzazbz_vector = np.mean(img_analysis.jzazbz_dict[self.word], axis=0)
        self.jzazbz_dist = np.mean(img_analysis.jzazbz_dist_dict[self.word], axis=0)

        self.rgb_vector = np.mean(img_analysis.rgb_dict[self.word], axis=0)
        self.rgb_dist = np.mean(img_analysis.rgb_dist_dict[self.word], axis=0)

        self.rgb_ratio = np.mean(img_analysis.rgb_ratio_dict[self.word], axis=0)
        self.colorgram_vector = img_analysis.compressed_img_dict[self.word]

        self.colorgram = PIL.Image.fromarray(self.colorgram_vector.astype(np.uint8))

    def print_word_color(self, size: int = 30, color_magnitude: float = 1.65) -> None:

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        plt.text(
            0.35, 0.5, self.word, color=color_magnitude * self.rgb_ratio, fontsize=size
        )
        ax.set_axis_off()
        plt.show()

    def save_vector_to_disk(self) -> None:

        vector_properties = {}
        vector_properties["jzazbz_vector"] = self.jzazbz_vector
        vector_properties["jzazbz_dist"] = self.jzazbz_dist
        vector_properties["rgb_vector"] = self.rgb_vector
        vector_properties["rgb_dist"] = self.rgb_dist
        vector_properties["rgb_ratio"] = self.rgb_ratio
        vector_properties["colorgram_vector"] = self.colorgram_vector

        with open(self.word + "_vector_properties", "w") as fp:
            json.dump(vector_properties, fp)


class LoadVectorsFromDisk:
    def __init__(self, path: Union[str, Path], default: bool = True) -> None:

        self.vectors = {}
        self.path = Path(path)
        if default:
            self.load_distributions()

    def load_distributions(self) -> None:

        with open(self.path.joinpath("vectors.json")) as vectors_json:

            vectors_dist = json.load(vectors_json)

            for info in vectors_dist:
                word = info["query"]
                rgb_dist = info["rgb_dist"]
                jzazbz_dist = info["jzazbz_dist"]

                word_vector = vectors.Vector(word)
                word_vector.rgb_dist = rgb_dist
                word_vector.jzazbz_dist = jzazbz_dist

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
                self.vectors[word].jzazbz_vector = datahelper.rgb_array_to_jzazbz_array(
                    img_array
                )

    def load_info_from_colorgram(
        self, word: str, compress_dims: Tuple[int, int] = (300, 300)
    ) -> None:

        img_raw = PIL.Image.open(self.path.joinpath("colorgrams/query={word}.png"))
        img_raw = img_raw.resize(
            (compress_dims[0], compress_dims[1]), PIL.Image.ANTIALIAS
        )
        img_array = np.array(img_raw)[:, :, :3]

        self.vectors[word].rgb_vector = img_array
        self.vectors[word].jzazbz_vector = datahelper.rgb_array_to_jzazbz_array(
            img_array
        )

    def load_colorgram(
        self, word: str, compress_dims: Tuple[int, int] = (300, 300)
    ) -> None:

        img_raw = PIL.Image.open(self.path.joinpath("colorgrams/query={word}.png"))
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
                    self.path.joinpath("colorgrams/query={word}.png")
                )
                img_raw = img_raw.resize(
                    (compress_dims[0], compress_dims[1]), PIL.Image.ANTIALIAS
                )
                img_array = np.array(img_raw)[:, :, :3]

                self.vectors[word].colorgram_vector = img_array
                self.vectors[word].colorgram = img_raw

    def load_concreteness_data(self) -> None:

        with open(
            self.path.joinpath("Concreteness_ratings_Brysbaert_et_al_BRM.csv")
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
                    self.vectors[search_term].conreteness_mean = float(Conc_M)
                    self.vectors[search_term].conreteness_sd = float(Conc_SD)
                except (KeyError, ValueError) as e:
                    continue
