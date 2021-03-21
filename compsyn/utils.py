"""Provides a utility to inject environment variables into argparse definitions.
Currently requires explicit naming of env vars to check for"""

import argparse
import os
import tempfile
from pathlib import Path

from PIL import Image


class EnvDefault(argparse.Action):
    """An argparse action class that auto-sets missing default values from env
    vars. Defaults to requiring the argument."""

    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar in os.environ:
            default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


# functional sugar for the above
def env_default(envvar):
    def wrapper(**kwargs):
        return EnvDefault(envvar, **kwargs)

    return wrapper


def compress_image(image_path: Path) -> Path:

    image = Image.open(image_path)

    temp_work_dir = Path(tempfile.TemporaryDirectory().name)
    temp_work_dir.mkdir(exist_ok=True, parents=True)
    compressed_image_path = temp_work_dir.joinpath(image_path.name)

    image.save(compressed_image_path, optimize=True, quality=30)

    return compressed_image_path


def human_bytes(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)
