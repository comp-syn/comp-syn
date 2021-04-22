from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .logger import get_logger
from .utils import env_default


def get_trial_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:

    if parser is None:
        parser = argparse.ArgumentParser()

    trial_parser = parser.add_argument_group("trial")
    trial_parser.add_argument(
        "--experiment-name",
        type=str,
        action=env_default("COMPSYN_EXPERIMENT_NAME"),
        default="default-experiment",
        help="An over-arching experiment_name can be used to facilitate multi-trial data collection efforts",
    )

    trial_parser.add_argument(
        "--trial-id",
        type=str,
        action=env_default("COMPSYN_TRIAL_ID"),
        default="default-trial",
        help="",
    )
    trial_parser.add_argument(
        "--hostname",
        type=str,
        action=env_default("COMPSYN_HOSTNAME"),
        default="default-hostname",
        help="Can be used to identify the hostname the data collection or analysis was run on",
    )
    trial_parser.add_argument(
        "--trial-timestamp",
        type=str,
        action=env_default("COMPSYN_TRIAL_TIMESTAMP"),
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="Usually this should be unset, as the code will record the current time.",
    )
    return parser


class Trial:
    """
    Trial subclasses can be used to standardize metadata for a set of experiments.
    
    Standardizing on a set of metadata means future Vector subclasses can be integrated
    into existing experiments seamlessly by re-existing Trial attributes.

    This also facilitates multiple hosts contributing to a shared effort and timeseries
    experiments.
    """

    def __init__(
        self,
        experiment_name: str,
        trial_id: str,
        hostname: Optional[str],
        trial_timestamp: Optional[str] = None,
    ) -> None:

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

        self.log = get_logger(self.__class__.__name__)
        self.log.info(f"experiment: {self.experiment_name}")
        self.log.info(f"trial_id: {self.trial_id}")
        self.log.info(f"hostname: {self.hostname}")

    def __repr__(self) -> str:
        representation = f"{self.__class__.__name__}"
        representation += f"\n\texperiment_name = {self.experiment_name}"
        representation += f"\n\ttrial_id        = {self.trial_id}"
        representation += f"\n\thostname        = {self.hostname}"
        representation += f"\n\ttrial_timestamp = {self.trial_timestamp}"
        return representation


def get_trial_from_env() -> Trial:
    """
    This method achieves two things:
      1. the Trial object can be configured from the environment,
      2. a default Trial object can be provided for less structured uses of Compsyn
    """
    trial_args, unknown = get_trial_args().parse_known_args()

    return Trial(
        experiment_name=trial_args.experiment_name,
        trial_id=trial_args.trial_id,
        hostname=trial_args.hostname,
        trial_timestamp=trial_args.trial_timestamp,
    )
