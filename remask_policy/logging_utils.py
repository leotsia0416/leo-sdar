from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "remask_policy", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def _parse_level(level: str) -> int:
    parsed = logging.getLevelName(level.upper())
    if isinstance(parsed, int):
        return parsed
    raise ValueError(f"Unknown log level: {level}")
