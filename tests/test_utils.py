from pathlib import Path

import pytest

from compsyn.utils import compress_image


@pytest.mark.unit
def test_compress_image() -> None:
    raw_image_path = (
        Path(__file__).parent.joinpath("test-assets/37f0960b6e.jpg").resolve()
    )
    compressed_image_path = compress_image(raw_image_path)

    assert raw_image_path.stat().st_size > compressed_image_path.stat().st_size
    # could assert some desired minimum compression ratio?

    compressed_image_path.unlink()
