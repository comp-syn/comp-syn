import os
from pathlib import Path

import pytest
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

from compsyn.utils import compress_image, set_env_var


@pytest.mark.unit
def test_compress_image() -> None:
    raw_image_paths = (
        Path(__file__)
        .parent.joinpath(
            "test-assets/test-downloads/raw-images/atlantis/known-dist/pytester/testoclock/"
        )
        .resolve()
    )

    compressed_image_paths = list()
    for raw_image_path in raw_image_paths.iterdir():
        compressed_image_paths.append(compress_image(raw_image_path))

    profile_plot_path = Path(__file__).parent.joinpath(
        "test-assets/compress-image-ratio.png"
    )
    x = [raw_image_path.stat().st_size for raw_image_path in raw_image_paths.iterdir()]
    y = [
        compressed_image_path.stat().st_size / raw_image_path.stat().st_size
        for compressed_image_path, raw_image_path in zip(
            compressed_image_paths, raw_image_paths.iterdir()
        )
    ]
    plt.plot(x, y, "o")
    plt.xlabel("Raw Image Size (bytes)")
    plt.ylabel("Compressed Image Size (percent of original)")
    plt.ylim([0, 1])

    plt.savefig(profile_plot_path)

    assert sum(y) / len(y) < 0.5

    # Clean up
    for cip in compressed_image_paths:
        cip.unlink()


@pytest.mark.unit
def test_set_env_var() -> None:
    env_var_key_a = "COMPSYN_PYTEST_ENV_VAR_A"
    env_var_key_a_code = "pytest_env_var_a"
    env_var_key_b = "COMPSYN_PYTEST_ENV_VAR_B"
    env_var_val_1 = "value-1"
    env_var_val_2 = "value-2"
    existing_env_var = os.getenv(env_var_key_a)

    assert existing_env_var is None, "unset \"{env_var_key_a}\" before running tests"

    set_env_var(env_var_key_a, env_var_val_1)

    assert os.getenv(env_var_key_a) == env_var_val_1

    set_env_var(env_var_key_a, env_var_val_2)
    set_env_var(env_var_key_b, env_var_val_1)

    assert os.getenv(env_var_key_a) == env_var_val_2
    assert os.getenv(env_var_key_b) == env_var_val_1
