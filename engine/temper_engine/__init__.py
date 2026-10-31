"""Temper engine: the local API the desktop app talks to."""

import os
from pathlib import Path

__version__ = "0.1.0"

# Temper sends no telemetry, and neither do the libraries it uses (Hugging Face's and others that honour
# DO_NOT_TRACK). Set before they're imported; workers inherit it.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


def home() -> Path:
    """Where runs, exports and settings live. The desktop shell sets TEMPER_HOME to its app-data dir."""
    path = Path(os.environ.get("TEMPER_HOME") or Path.home() / ".temper")
    path.mkdir(parents=True, exist_ok=True)
    return path
