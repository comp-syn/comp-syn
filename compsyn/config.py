from __future__ import annotations

import argparse
import os
from pathlib import Path

from .logger import get_logger
from .helperfunctions import get_browser_args, get_google_application_args
from .s3 import get_s3_args
from .jzazbz import get_jzazbz_args
from .trial import get_trial_args
from .utils import set_env_var, get_logger_args


class CompsynConfig:
    """ 
    Convenient interface for setting compsyn config values in environment through code.
    The possible configuration values are gathered from argparse.
    """

    def __init__(
        self, show_secret_values: bool = False, **kwargs: Dict[str, str]
    ) -> None:
        self.show_secret_values = show_secret_values
        self.config = dict()

        # fill argument values according to argparse config
        for key, val in self.args.items():
            set_env_var(key, val)
            self.config[key] = val

        # overwrite argparse values with those called in code
        for key, val in kwargs.items():
            set_env_var(key, val)  # set vars in os environ
            self.config[key] = val  # record on config object for convenience

        for required_path in ["jzazbz_array"]:
            if not Path(self.config["jzazbz_array"]).is_file():
                raise FileNotFoundError(
                    f"{self.config['jzazbz_array']} does not exist!"
                )

    def __repr__(self) -> str:
        """ A nice human readable representation """
        representation = f"{self.__class__.__name__}"
        for key, val in self.config.items():
            if (
                key in self.secret_attrs
                and not self.show_secret_values
                and val is not None
            ):
                val = "<redacted>"
            representation += f"\n\t{key:30s} = {val}"
        return representation

    @property
    def secret_attrs(self) -> List[str]:
        """ These will be hidden from output """
        return ["s3_secret_access_key"]

    @property
    def args(self) -> List[str]:
        """ Accumulate argparsers here """
        parser = argparse.ArgumentParser()
        get_jzazbz_args(parser)
        get_google_application_args(parser)
        get_browser_args(parser)
        get_s3_args(parser)
        get_logger_args(parser)
        args, unknown = parser.parse_known_args()
        return vars(args)
