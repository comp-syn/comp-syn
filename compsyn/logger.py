from __future__ import annotations
import logging
import sys
from pathlib import Path

LOG_FILE_FORMAT = (
    "[%(asctime)s] %(pathname)s:%(lineno)d (%(name)s)  %(levelname)s: %(message)s"
)

CONSOLE_FORMAT = "[%(asctime)s] (%(name)s)  %(levelname)s: %(message)s"


def get_logger(
    name: str,
    console_level: int = logging.INFO,
    log_file: Optional[Path] = None,
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
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        logger.addHandler(console_handler)
    if log_file is not None:
        if len(logger.handlers) == 0 or not max(
            [isinstance(handler, logging.FileHandler) for handler in logger.handlers]
        ):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(LOG_FILE_FORMATTER))
            logger.addHandler(file_handler)

    return logger
