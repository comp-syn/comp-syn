import os

import pytest
from pytest import approx

from compsyn.s3 import get_s3_client
from compsyn.wordtocolor_vector import WordToColorVector
from compsyn.trial import Trial


@pytest.mark.unit
def test_w2cv_produce_known_analysis_results():
    """
	creates vector object for the saved love image set and tests distributions and ratios.
	"""

    path = os.getcwd() + "/downloads/paper_categories"
    w2cv = WordToColorVector(label="love")

    expected_rgb_dist = [
        1.40895633e-07,
        7.41199147e-09,
        8.63679137e-11,
        5.33250120e-09,
        1.23785039e-07,
        1.61645538e-08,
        2.79873735e-08,
        1.60805576e-07,
    ]
    expected_jzazbz_dist = [
        43.70086125,
        82.88902835,
        20.36144642,
        207.55729541,
        11.45945804,
        131.10485973,
        10.49319587,
        334.18735373,
    ]
    expected_rgb_ratio = [0.43899919, 0.27914204, 0.28185877]

    assert expected_rgb_dist[0] == approx(w2cv.rgb_dist[0], rel=1e-6, abs=1e-12)
    assert expected_jzazbz_dist[0] == approx(w2cv.jzazbz_dist[0], rel=1e-6, abs=1e-12)
    assert expected_rgb_ratio[0] == approx(w2cv.rgb_ratio[0], rel=1e-6, abs=1e-12)


class InvalidWordToColorVectorError(Exception):
    pass


def validate_w2cv(w2cv: WordToColorVector) -> None:
    # TODO analysis results exist and are the right shape?
    pass


@pytest.mark.integration
def test_w2cv_fresh_run():
    w2cv = WordToColorVector(label="dog")
    w2cv.run_image_capture()
    w2cv.create_embedding()

    # TODO fresh raw images were downloaded?

    validate_w2cv(w2cv)


@pytest.mark.integration
def test_w2cv_s3_integration():
    try:
        get_s3_client()
    except Exception as exc:
        print(exc)
        print(
            "s3 error, configure S3 environment variables according to compsyn.s3.get_s3_args to test this feature"
        )

    # first test push integration
    trial = Trial(
        experiment_name="compsyn-integration-test",
        trial_id="test-s3-integration",
        hostname="compsyn",
    )

    w2cv = WordToColorVector(label="seaborne", revision="test", trial=trial)
    w2cv.run_image_capture()
    w2cv.create_embedding()

    w2cv.push(overwrite=True)

    # then, test pulling what we just pushed

    del w2cv

    w2cv = WordToColorVector(label="seaborne", revision="test", trial=trial)
    w2cv.pull()

    validate_w2cv(w2cv)
