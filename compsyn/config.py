from __future__ import annotations

import argparse

import os

from .logger import get_logger
from .helperfunctions import get_browser_args
from .s3 import get_s3_args
from .jzazbz import get_jzazbz_args
from .trial import get_trial_args
from .utils import set_env_var
        

class CompsynConfig:
    """ 
    Convenient interface for setting compsyn config values in environment through code.
    The possible configuration values are gathered from argparse.
    """

    def __init__(self, show_secret_values: bool = False, **kwargs: Dict[str, str]) -> None:
        self.show_secret_values = show_secret_values
        self.config = dict()

        # fill argument values according to argparse config
        for key, val in self.args.items():
            set_env_var(key, val)
            self.config[key] = val

        # overwrite argparse values with those called in code
        for key, val in kwargs.items():
            set_env_var(key, val) # set vars in os environ
            self.config[key] = val # record on config object for convenience

    def __repr__(self) -> str:
        representation = f"{self.__class__.__name__}"
        for key, val in self.config.items():
            if key in self.secret_attrs and not self.show_secret_values and val is not None:
                val = "<redacted>"
            representation += f"\n\t{key:20s} = {val}"
        return representation


    @property
    def secret_attrs(self) -> List[str]:
        return [ "s3_secret_access_key" ]

    @property
    def args(self) -> List[str]:
        parser = argparse.ArgumentParser()
        get_trial_args(parser)
        get_jzazbz_args(parser)
        get_browser_args(parser)
        get_s3_args(parser)
        args, unknown = parser.parse_known_args()
        return vars(args)

