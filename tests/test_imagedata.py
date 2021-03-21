from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from compsyn.datahelper import ImageData, rgb_array_to_jzazbz_array

TEST_IMAGES = (
    Path(__file__)
    .parent.joinpath(
        "test-assets/test-downloads/raw-images/atlantis/known-dist/pytester/testoclock/"
    )
    .resolve()
)


@pytest.mark.unit
def test_load_rgb_and_jzazbz_arrays() -> None:
    image_data = ImageData()
    rgb_array = image_data.load_rgb_image(path=TEST_IMAGES.joinpath("3bd3eec514.jpg"))
    jzazbz_array = rgb_array_to_jzazbz_array(rgb_array)
    assert isinstance(rgb_array, np.ndarray)
    assert isinstance(jzazbz_array, np.ndarray)
    assert rgb_array.shape == (300, 300, 3)
    assert jzazbz_array.shape == (300, 300, 3)

    image_data = ImageData(compress_dims=None)
    rgb_array = image_data.load_rgb_image(path=TEST_IMAGES.joinpath("3bd3eec514.jpg"))
    jzazbz_array = rgb_array_to_jzazbz_array(rgb_array)
    assert isinstance(rgb_array, np.ndarray)
    assert isinstance(jzazbz_array, np.ndarray)
    assert rgb_array.shape == (1080, 1920, 3)
    assert jzazbz_array.shape == (1080, 1920, 3)

    image_data = ImageData(compress_dims=(500, 500))
    rgb_array = image_data.load_rgb_image(path=TEST_IMAGES.joinpath("3bd3eec514.jpg"))
    jzazbz_array = rgb_array_to_jzazbz_array(rgb_array)
    assert isinstance(rgb_array, np.ndarray)
    assert isinstance(jzazbz_array, np.ndarray)
    assert rgb_array.shape == (500, 500, 3)
    assert jzazbz_array.shape == (500, 500, 3)


@pytest.mark.unit
def test_load_image_dict_from_folder() -> None:

    image_data = ImageData()

    image_data.load_image_dict_from_folder(TEST_IMAGES, label="atlantis")

    assert image_data.compress_dims == (300, 300)  # this is the default

    assert len(list(image_data.jzazbz_dict.keys())) > 0
    assert len(list(image_data.rgb_dict.keys())) > 0
    assert len(list(image_data.labels_list)) > 0

    count = 0
    for rgb_vector, jzazbz_vector in zip(
        image_data.rgb_dict["atlantis"], image_data.jzazbz_dict["atlantis"]
    ):
        count += 1
        assert jzazbz_vector.shape == (300, 300, 3)
        assert rgb_vector.shape == (300, 300, 3)
    assert count == 95

    assert image_data.labels_list[0] == "atlantis"

    image_data = ImageData(compress_dims=(500, 500))
    image_data.load_image_dict_from_folder(
        TEST_IMAGES, label="atlantis",
    )

    assert image_data.compress_dims == (500, 500)
    assert len(list(image_data.jzazbz_dict.keys())) > 0
    assert len(list(image_data.rgb_dict.keys())) > 0
    assert len(list(image_data.labels_list)) > 0

    count = 0
    for rgb_vector, jzazbz_vector in zip(
        image_data.rgb_dict["atlantis"], image_data.jzazbz_dict["atlantis"]
    ):
        count += 1
        assert jzazbz_vector.shape == (500, 500, 3)
        assert rgb_vector.shape == (500, 500, 3)
    assert count == 95

    assert image_data.labels_list[0] == "atlantis"


# TODO
# @pytest.mark.unit
# def test_load_image_continuum_from_folder() -> None
#    pass

# TODO
# @pytest.mark.unit
# def test_load_image_dict_from_subfolders() -> None
#    pass
