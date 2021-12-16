from __future__ import annotations

import os
import logging
import sys
from pathlib import Path

FILE_FORMAT = (
    "[%(asctime)s] %(pathname)s:%(lineno)d (%(name)s)  %(levelname)s: %(message)s"
)
FILE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

CONSOLE_FORMAT = "[%(asctime)s] (%(name)s)  %(levelname)s: %(message)s"
CONSOLE_TIME_FORMAT = "%s"


def get_logger(
    name: str,
    console_level: int = int(os.getenv("COMPSYN_LOG_LEVEL", logging.INFO)),
    log_file: Union[str, Path, None] = os.getenv("COMPSYN_LOG_FILE", None),
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
        Wrapper for setting up and getting logger
    """

    if not name.startswith("compsyn."):
        name = "compsyn." + name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if len(logger.handlers) == 0 or not max(
        [isinstance(handler, logging.StreamHandler) for handler in logger.handlers]
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter(CONSOLE_FORMAT, CONSOLE_TIME_FORMAT)
        )
        logger.addHandler(console_handler)
        logger.debug(f"added console handler")
    if log_file is not None:
        log_file = Path(log_file)
        if len(logger.handlers) == 0 or not max(
            [isinstance(handler, logging.FileHandler) for handler in logger.handlers]
        ):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_TIME_FORMAT))
            logger.addHandler(file_handler)
            logger.debug(f"added file handler")

    return logger
