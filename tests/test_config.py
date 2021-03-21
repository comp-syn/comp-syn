import os

import pytest

from compsyn.config import CompsynConfig
from compsyn.trial import get_trial

@pytest.mark.unit
def test_CompsynConfig() -> None:

    # capture original environment COMPSYN_ variables so we can put things back after messin' about
    original_values = {key: val for key, val in os.environ.items() if key.startswith("COMPSYN_")}

    config = CompsynConfig(experiment_name="test-patterns", trial_id="phase-0", hostname="pytester", work_dir=None)

    assert os.getenv("COMPSYN_EXPERIMENT_NAME") == "test-patterns"
    assert os.getenv("COMPSYN_TRIAL_ID") == "phase-0"
    assert os.getenv("COMPSYN_HOSTNAME") == "pytester"

    trial = get_trial()

    assert trial.experiment_name == "test-patterns"
    assert trial.trial_id == "phase-0"
    assert trial.hostname == "pytester"

    config = CompsynConfig(experiment_name="test-patterns", trial_id="phase-1", hostname="pytester", work_dir=None)

    original_work_dir = trial.work_dir

    assert trial.experiment_name == "test-patterns"
    assert trial.trial_id == "phase-0"
    assert trial.hostname == "pytester"

    trial = get_trial()

    assert trial.experiment_name == "test-patterns"
    assert trial.trial_id == "phase-1"
    assert trial.hostname == "pytester"
    assert trial.work_dir != original_work_dir # re-creating Trial should pick a new tmp dir in this case

    # reset original environment values
    for key, val in original_values.items():
        os.environ[key] = val
