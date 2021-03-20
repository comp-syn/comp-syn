from __future__ import annotations

import argparse
import os
from pathlib import Path

from .utils import env_default

def get_browser_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:

    if parser is None:
        parser = argparse.ArgumentParser()

    browser_parser = parser.add_argument_group("browser")

    browser_parser.add_argument(
        "--driver-browser",
        type=str,
        action=env_default("COMPSYN_DRIVER_BROWSER"),
        required=True,
        help="Browser name, e.g. Firefox, Chrome",
    )

    browser_parser.add_argument(
        "--driver-path",
        type=str,
        action=env_default("COMPSYN_DRIVER_PATH"),
        required=True,
        help="Browser driver path"
    )


    return parser
