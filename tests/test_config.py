import os

import pytest
from pathlib import Path

from compsyn.config import CompsynConfig
from compsyn.trial import get_trial_from_env


@pytest.mark.unit
def test_CompsynConfig() -> None:

    # capture original environment COMPSYN_ variables so we can put things back after messin' about
    original_values = {
        key: val for key, val in os.environ.items() if key.startswith("COMPSYN_")
    }

    config = CompsynConfig(
        experiment_name="test-patterns", trial_id="phase-0", hostname="pytester",
    )

    assert os.getenv("COMPSYN_EXPERIMENT_NAME") == "test-patterns"
    assert os.getenv("COMPSYN_TRIAL_ID") == "phase-0"
    assert os.getenv("COMPSYN_HOSTNAME") == "pytester"

    trial = get_trial_from_env()

    assert trial.experiment_name == "test-patterns"
    assert trial.trial_id == "phase-0"
    assert trial.hostname == "pytester"

    config = CompsynConfig(
        experiment_name="test-patterns", trial_id="phase-1", hostname="pytester",
    )

    assert trial.experiment_name == "test-patterns"
    assert trial.trial_id == "phase-0"
    assert trial.hostname == "pytester"

    trial = get_trial_from_env()

    assert trial.experiment_name == "test-patterns"
    assert trial.trial_id == "phase-1"
    assert trial.hostname == "pytester"

    # reset original environment values
    for key, val in original_values.items():
        os.environ[key] = val
