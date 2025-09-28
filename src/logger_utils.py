# src/logger_utils.py

"""
📡 Logger Utils

Lightweight logging helpers for standardized CLI output.
Use to enhance visibility and checkpoint traceability across
FreeCAD pipelines, validation modules, and utility layers.
"""

import datetime
import sys

def log_checkpoint(msg: str, *, emoji="🪵", newline=True) -> None:
    """
    Print a standardized checkpoint log.

    Args:
        msg (str): Message content to display.
        emoji (str, optional): Emoji prefix for log tag. Default: '🪵'.
        newline (bool, optional): Whether to append a newline after. Default: True.
    """
    ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
    print(f"{emoji} [{ts}] {msg}", end="\n" if newline else "")



