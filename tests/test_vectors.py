import os
from pathlib import Path

import pytest
from pytest import approx

from compsyn.s3 import get_s3_client
from compsyn.wordtocolor_vector import WordToColorVector
from compsyn.trial import Trial


@pytest.mark.integration
def test_w2cv_produce_known_analysis_results():
    """
	creates vector object for the saved love image set and tests distributions and ratios.
	"""

    trial = Trial(
        experiment_name="test-downloads",
        trial_id="known-dist",
        hostname="pytester",
        trial_timestamp="testoclock",
        work_dir=Path(__file__).parent.joinpath("test-assets")
    )
    w2cv = WordToColorVector(label="atlantis", trial=trial)
    w2cv.run_analysis()


    expected_rgb_dist = [1.89736450e-07, 6.40508963e-08, 4.61866761e-09, 8.04098130e-08,
 1.60031875e-08, 1.87863849e-09, 1.86990891e-08, 1.07072293e-07]
    expected_jzazbz_dist = [243.92783389, 172.05688373,   3.19593217,  54.89909299, 151.00108506,
 148.22235232,   3.8768293,   64.57348934]
    expected_rgb_ratio = [0.24078636, 0.35787702, 0.40133662]

    for i, v in enumerate(expected_rgb_dist):
        assert v == approx(w2cv.rgb_dist[i], rel=1e-6, abs=1e-12)
    for i, v in enumerate(expected_jzazbz_dist):
        assert v == approx(w2cv.jzazbz_dist[i], rel=1e-6, abs=1e-12)
    for i, v in enumerate(expected_rgb_ratio):
        assert v == approx(w2cv.rgb_ratio[i], rel=1e-6, abs=1e-12)


@pytest.mark.integration
def test_w2cv_fresh_run():
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.run_image_capture()
    w2cv.run_analysis()
    w2cv.save()


@pytest.mark.integration
def test_w2cv_s3_push():
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.load()
    w2cv.push(include_raw_images=True, overwrite=True)


@pytest.mark.integration
def test_w2cv_s3_pull():
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.pull(include_raw_images=True, overwrite=True)
    w2cv.run_analysis()
