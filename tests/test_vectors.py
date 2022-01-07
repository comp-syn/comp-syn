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
    print("computed rgb dist:", w2cv.rgb_dist)
    print("computed jzazbz dist:", w2cv.jzazbz_dist)
    print("computed rgb ratio:", w2cv.rgb_ratio)
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

    CompsynConfig(
        work_dir=Path(__file__).parent.joinpath("test-assets"),
    )
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
            1.90415631e-07,
            6.38808753e-08,
            4.48634131e-09,
            8.02620816e-08,
            1.56456526e-08,
            1.76758597e-09,
            1.85108979e-08,
            1.07499970e-07,
        ],
        expected_jzazbz_dist=[
            244.12465444,
            173.21539068,
            3.04959916,
            54.23655512,
            150.68078842,
            149.05550707,
            3.70381827,
            63.68718565,
        ],
        expected_rgb_ratio=[0.24074674, 0.35789798, 0.40135529],
    )


@pytest.mark.online
def test_w2cv_fresh_run():
    start = time.time()
    w2cv = WordToColorVector(label="dog", revision="raw-test")
    w2cv.run_image_capture(max_items=10)
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


@pytest.mark.online
def test_w2cv_fresh_run_with_related():
    start = time.time()
    w2cv = WordToColorVector(label="poodle")
    w2cv.run_image_capture(max_items=10, include_related=True)
    w2cv.run_analysis()

    print(
        "full run (with related) completed in", round(time.time() - start, 2), "seconds"
    )


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
