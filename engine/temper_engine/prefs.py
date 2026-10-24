"""settings.json in Temper's home: the engine-side preferences (model source, models folder)."""

from __future__ import annotations

import json
from pathlib import Path

from . import home


def _file() -> Path:
    return home() / "settings.json"


def load() -> dict:
    try:
        return json.loads(_file().read_text())
    except (OSError, ValueError):
        return {}


def update(key: str, value) -> None:
    """Sets one key (None removes it) and keeps the rest."""
    data = load()
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    _file().write_text(json.dumps(data, indent=2))
