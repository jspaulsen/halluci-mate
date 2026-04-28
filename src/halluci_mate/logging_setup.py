"""Logging configuration shared across CLI scripts."""

from __future__ import annotations

import logging


def configure_script_logging(script_logger_name: str) -> None:
    """Configure logging for a script and halluci_mate modules only.

    Leaves third-party loggers (e.g., HuggingFace) at their defaults.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    for name in (script_logger_name, "halluci_mate"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        log.addHandler(handler)
