"""
helpers.py
----------
Reusable helper functions for logging and path handling.
"""

from pathlib import Path
from datetime import datetime


def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def timestamped_message(message: str) -> str:
    """
    Attach a timestamp to a log message.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{now}] {message}"
