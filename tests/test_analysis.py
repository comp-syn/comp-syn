from __future__ import annotations

from pathlib import Path

import pytest

from compsyn.analysis import ImageAnalysis, merge_vectors_to_image_analysis
from compsyn.config import CompsynConfig
from compsyn.vectors import WordToColorVector
from compsyn.trial import Trial


@pytest.mark.integration
def test_merge_vectors_to_image_analysis() -> None:

    CompsynConfig(
        work_dir=Path(__file__).parent.joinpath("test-assets"),
    )
    trial = Trial(
        experiment_name="test-downloads",
        trial_id="known-dist",
        hostname="pytester",
        trial_timestamp="testoclock",
    )
    vectors = list()
    for label in ["ice", "fire", "earth", "wind", "atlantis"]:
        w2cv = WordToColorVector(label=label, trial=trial)
        w2cv.load_data()
        vectors.append(w2cv)

    image_analysis = merge_vectors_to_image_analysis(vectors)

    assert len(image_analysis.labels_list) == 5
    assert len(image_analysis.jzazbz_dict) == 5
    assert len(image_analysis.rgb_dict) == 5

    # TODO: validate resulting ImageAnalysis object more
