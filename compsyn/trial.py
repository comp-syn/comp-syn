from collections import UserDict
from dataclasses import dataclass, InitVar
from datetime import datetime
from pathlib import Path


@dataclass
class Trial(UserDict):
    """
    Trial subclasses can be used to standardize metadata for a set of experiments.
    
    Standardizing on a set of metadata means future Vector subclasses can be integrated
    into existing experiments seamlessly by re-existing Trial attributes.

    This also facilitates multiple hosts contributing to a shared effort and timeseries
    experiments.
    """

    #: An over-arching experiment_name can be used to facilitate multi-trial data collection efforts
    experiment_name: str
    #: The more specfic trial_id can be used to logically partition a dataset
    trial_id: str
    #: The (optional) hostname attribute can be used to provide context on the machine the data was initially gathered on
    hostname: Optional[str] = None
    #: The trial_timestamp can record when the initial data capture took place
    trial_timestamp: InitVar[Optional[str]] = None
    #: The local root working directory
    work_dir: InitVar[Optional[Path]] = None

    def __post_init__(
        self, trial_timestamp: Optional[str], work_dir: Optional[Path]
    ) -> None:
        """ set defaults """

        if trial_timestamp is None:
            self.trial_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        if work_dir is None:
            self.work_dir = tempfile.TemporaryDirectory()
            get_logger(self.__class__.__name__).warning(
                f"no work_dir passed, using a temporary directory: {self.work_dir}"
            )


def get_trial_from_env() -> Trial:
    """
    This method achieves two things:
      1. the Trial object can be configured from the environment,
      2. a default Trial object can be provided for less structured uses of Compsyn
    """

    return Trial(
        experiment_name=os.getenv("COMPSYN_EXPERIMENT_NAME", "no-experiment-name"),
        trial_id=os.getenv("COMPSYN_TRIAL_ID", "local-trial"),
        hostname=os.getenv("COMPSYN_HOSTNAME", "default"),
        trial_timestamp=os.getenv("COMPSYN_TRIAL_TIMESTAMP", None),
        work_dir=os.getenv("COMPSYN_WORK_DIR", None),
    )
