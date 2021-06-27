from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np

from .utils import env_default
from .logger import get_logger


def get_jzazbz_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:

    if parser is None:
        parser = argparse.ArgumentParser()

    jzazbz_parser = parser.add_argument_group("s3")

    jzazbz_parser.add_argument(
        "--jzazbz-array",
        type=str,
        action=env_default("COMPSYN_JZAZBZ_ARRAY"),
        default="jzazbz_array.npy",
        help="Path to jzazbz_array.npy file",
    )

    return parser


@lru_cache
def get_jzazbz_array() -> None:
    jzazbz_args, unknown = get_jzazbz_args().parse_known_args()
    get_logger("setup_jzazbz_array").debug(
        f"jzazbz transformation will use {jzazbz_args.jzazbz_array}"
    )
    return np.load(jzazbz_args.jzazbz_array)
