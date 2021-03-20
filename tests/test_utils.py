from pathlib import Path

import pytest
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from compsyn.utils import compress_image

@pytest.mark.unit
def test_compress_image() -> None:
    raw_image_paths = (
        Path(__file__).parent.joinpath("test-assets/test-downloads/raw-images/atlantis/known-dist/pytester/testoclock/").resolve()
    )

    compressed_image_paths = list()
    for raw_image_path in raw_image_paths.iterdir():
        compressed_image_paths.append(compress_image(raw_image_path))

    profile_plot_path = Path(__file__).parent.joinpath("test-assets/compress-image-ratio.png")
    x = [raw_image_path.stat().st_size for raw_image_path in raw_image_paths.iterdir()]
    y = [compressed_image_path.stat().st_size/raw_image_path.stat().st_size for compressed_image_path, raw_image_path in zip(compressed_image_paths, raw_image_paths.iterdir())]
    plt.plot(x, y, "o")
    plt.xlabel("Raw Image Size (bytes)")
    plt.ylabel("Compressed Image Size (percent of original)")
    plt.ylim([0, 1])

    plt.savefig(profile_plot_path)

    assert sum(y)/len(y) < 0.5

    # Clean up
    for cip in compressed_image_paths:
        cip.unlink()
