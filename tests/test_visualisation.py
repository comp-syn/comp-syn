from __future__ import annotations

from pathlib import Path

import pytest

from compsyn.analysis import ImageAnalysis, merge_vectors_to_image_analysis
from compsyn.config import CompsynConfig
from compsyn.vectors import WordToColorVector
from compsyn.trial import Trial
from compsyn.visualisation import Visualisation


@pytest.mark.integration
def test_Visualisation() -> None:

    CompsynConfig(work_dir=Path(__file__).parent.joinpath("test-assets"),)
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
    image_analysis.compress_color_data()
    image_analysis.entropy_computations()

    visualization = Visualisation(image_analysis)

    visualization.plot_labels_in_space()

    # TODO: validate resulting ImageAnalysis object more
