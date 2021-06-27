"""
- methods for helping with configuration from environment
- common tasks like compressing an image, creating human-readable bytes
"""

from __future__ import annotations

import argparse
import os
import tempfile
import logging
from pathlib import Path

from PIL import Image

from .logger import get_logger


def get_logger_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """ Fetches arguments for compsyn's logging """

    if parser is None:
        parser = argparse.ArgumentParser()

    logger_parser = parser.add_argument_group("logger")

    logger_parser.add_argument(
        "--log-level",
        type=str,
        action=env_default("COMPSYN_LOG_LEVEL"),
        default=str(logging.INFO),
        help="default log level for the console handler",
    )

    logger_parser.add_argument(
        "--log-file",
        type=str,
        action=env_default("COMPSYN_LOG_FILE"),
        required=False,
        help="file to use for debug level log file",
    )

    return parser


def set_env_var(key: str, val: Optional[str]) -> None:
    """ 
        Set the COMPSYN_ environment variable for key, warning when overwriting
        Will unset environment variable if val is None
    """
    log = get_logger("set_env_var")

    key = key.upper()
    if not key.startswith("COMPSYN_"):
        key = "COMPSYN_" + key

    existing_env_val = os.getenv(key)

    if existing_env_val is not None and existing_env_val != val:
        log.debug(
            f"existing environment {key}={existing_env_val} clobbered by {key}={val}"
        )

    if val is None:
        try:
            del os.environ[key]
        except KeyError:
            # already unset
            pass
    else:
        os.environ[key] = str(val)


class EnvDefault(argparse.Action):
    """
        An argparse action class that auto-sets missing default values from env vars. 
        Defaults to requiring the argument, meaning an error will be thrown if no value 
        was directly passed to argparse and the env_default returned None as well.
    """

    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar in os.environ:
            default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def env_default(envvar):
    """ functional sugar for EnvDefault """

    def wrapper(**kwargs):
        return EnvDefault(envvar, **kwargs)

    return wrapper


def compress_image(image_path: Path, quality: int = 30) -> Path:
    """ Common image compression strategy using PIL Image.save quality """
    import warnings

    warnings.filterwarnings(
        action="ignore", message="Implicitly cleaning", category=ResourceWarning
    )

    image = Image.open(image_path)

    temp_work_dir = Path(tempfile.TemporaryDirectory().name)
    temp_work_dir.mkdir(exist_ok=True, parents=True)
    compressed_image_path = temp_work_dir.joinpath(image_path.name)

    image.save(compressed_image_path, optimize=True, quality=quality)

    return compressed_image_path


def human_bytes(num, suffix="B") -> str:
    """ Create human readable representation of a number of bytes """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)
