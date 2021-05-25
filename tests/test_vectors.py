from __future__ import annotations
import os
import time
from pathlib import Path

import pytest
from pytest import approx

from compsyn.config import CompsynConfig
from compsyn.s3 import get_s3_client
from compsyn.wordtocolor_vector import WordToColorVector
from compsyn.trial import Trial


def validate_w2cv(
    w2cv: WordToColorVector,
    expected_rgb_dist: List[float],
    expected_jzazbz_dist: List[float],
    expected_rgb_ratio: List[float],
    rel: float = 1e-6,
) -> None:
    for i, v in enumerate(expected_rgb_dist):
        assert v == approx(w2cv.rgb_dist[i], rel=rel, abs=1e-10)
    for i, v in enumerate(expected_jzazbz_dist):
        assert v == approx(w2cv.jzazbz_dist[i], rel=rel, abs=1e-10)
    for i, v in enumerate(expected_rgb_ratio):
        assert v == approx(w2cv.rgb_ratio[i], rel=rel, abs=1e-10)


@pytest.mark.integration
def test_w2cv_produce_known_analysis_results():
    """
	creates vector object for the saved love image set and tests distributions and ratios.
	"""

    CompsynConfig(work_dir=Path(__file__).parent.joinpath("test-assets"),)
    trial = Trial(
        experiment_name="test-downloads",
        trial_id="known-dist",
        hostname="pytester",
        trial_timestamp="testoclock",
    )
    w2cv = WordToColorVector(label="atlantis", trial=trial)
    w2cv.run_analysis()

    validate_w2cv(
        w2cv=w2cv,
        expected_rgb_dist=[
            1.89736450e-07,
            6.40508963e-08,
            4.61866761e-09,
            8.04098130e-08,
            1.60031875e-08,
            1.87863849e-09,
            1.86990891e-08,
            1.07072293e-07,
        ],
        expected_jzazbz_dist=[
            243.92783389,
            172.05688373,
            3.19593217,
            54.89909299,
            151.00108506,
            148.22235232,
            3.8768293,
            64.57348934,
        ],
        expected_rgb_ratio=[0.24078636, 0.35787702, 0.40133662],
    )


@pytest.mark.online
def test_w2cv_fresh_run():
    start = time.time()
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.run_image_capture()
    w2cv.run_analysis()

    # set results for the next two tests to use, better way to do this?
    global DOG_RGB_DIST
    DOG_RGB_DIST = w2cv.rgb_dist
    global DOG_JZAZBZ_DIST
    DOG_JZAZBZ_DIST = w2cv.jzazbz_dist
    global DOG_RGB_RATIO
    DOG_RGB_RATIO = w2cv.rgb_ratio

    w2cv.save()
    print("full run completed in", round(time.time() - start, 2), "seconds")


@pytest.mark.credentials
@pytest.mark.depends(on=["test_w2cv_fresh_run"])
def test_w2cv_s3_push():
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.load()
    validate_w2cv(
        w2cv=w2cv,
        expected_rgb_dist=DOG_RGB_DIST,
        expected_jzazbz_dist=DOG_JZAZBZ_DIST,
        expected_rgb_ratio=DOG_RGB_RATIO,
    )
    w2cv.push(include_raw_images=True, overwrite=True)


@pytest.mark.credentials
@pytest.mark.depends(on=["test_w2cv_fresh_run", "test_w2cv_s3_push"])
def test_w2cv_s3_pull():
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.delete_local_images()
    w2cv.pull(include_raw_images=True, overwrite=True)
    w2cv.run_analysis()
    validate_w2cv(
        w2cv=w2cv,
        expected_rgb_dist=DOG_RGB_DIST,
        expected_jzazbz_dist=DOG_JZAZBZ_DIST,
        expected_rgb_ratio=DOG_RGB_RATIO,
        rel=1,  # re-calculating from lower quality images after storing in S3 compressed TODO: discuss
    )
