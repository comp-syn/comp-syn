from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from .logger import get_logger


class Trial:
    """
    Trial subclasses can be used to standardize metadata for a set of experiments.
    
    Standardizing on a set of metadata means future Vector subclasses can be integrated
    into existing experiments seamlessly by re-existing Trial attributes.

    This also facilitates multiple hosts contributing to a shared effort and timeseries
    experiments.
    """
    def __init__(self, experiment_name: str, trial_id: str, hostname: Optional[str], trial_timestamp: Optional[str], work_dir: Optional[Path]) -> None:

        #: An over-arching experiment_name can be used to facilitate multi-trial data collection efforts
        self.experiment_name = experiment_name
        #: The more specfic trial_id can be used to logically partition a dataset
        self.trial_id = trial_id
        #: The (optional) hostname attribute can be used to provide context on the machine the data was initially gathered on
        if hostname is None:
            host_name = "unknown-host"
        self.hostname = hostname
        #: The trial_timestamp can record when the initial data capture took place
        if trial_timestamp is None:
            trial_timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        self.trial_timestamp = trial_timestamp
        #: The local root working directory
        if work_dir is None:
            work_dir = tempfile.TemporaryDirectory().name
            get_logger(self.__class__.__name__).warning(
                f"no work_dir passed, using a temporary directory: {work_dir}"
            )
        self.work_dir = Path(work_dir)

        self.log = get_logger(self.__class__.__name__)
        self.log.info(f"work_dir: {self.work_dir}")
        self.log.info(f"experiment: {self.experiment_name}")
        self.log.info(f"trial_id: {self.trial_id}")
        self.log.info(f"hostname: {self.hostname}")


def get_trial_from_env() -> Trial:
    """
    This method achieves two things:
      1. the Trial object can be configured from the environment,
      2. a default Trial object can be provided for less structured uses of Compsyn
    """

    return Trial(
        experiment_name=os.getenv("COMPSYN_EXPERIMENT_NAME", "default-experiment"),
        trial_id=os.getenv("COMPSYN_TRIAL_ID", "local-trial"),
        hostname=os.getenv("COMPSYN_HOSTNAME", None),
        trial_timestamp=os.getenv("COMPSYN_TRIAL_TIMESTAMP", None),
        work_dir=os.getenv("COMPSYN_WORK_DIR", None),
    )
