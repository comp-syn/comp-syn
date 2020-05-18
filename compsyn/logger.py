from __future__ import annotations
import logging
import sys
from pathlib import Path

LOG_FILE_FORMAT = (
    "[%(asctime)s] %(pathname)s:%(lineno)d (%(name)s)  %(levelname)s: %(message)s"
)

CONSOLE_FORMAT = "[%(asctime)s] (%(name)s)  %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FILE_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
    filename=Path(__file__).parents[2].joinpath("compsyn.log"),
    filemode="w",
)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
        Wrapper for setting up and getting logger
    """

    logger = logging.getLogger(name)

    if len(logger.handlers) == 0 or not max(
        [isinstance(handler, logging.StreamHandler) for handler in logger.handlers]
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        logger.addHandler(console_handler)

    return logger
