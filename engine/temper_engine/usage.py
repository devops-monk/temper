"""What Temper has done for you: tokens read and written by day and model, plus counts of chats, training
runs, exports and datasets, for Settings → Statistics.

Token counts come from every reply through the model API (the app's own chats included) and are kept
in usage.json in Temper's home, saved now and then rather than on every reply. Nothing leaves this
computer.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter

from . import home

router = APIRouter(prefix="/api/stats", tags=["runtime"])

SAVE_EVERY = 30  # seconds

_lock = threading.Lock()
_days: dict | None = None
_dirty_since: float | None = None


def _file() -> Path:
    return home() / "usage.json"


def _all() -> dict:
    global _days
    if _days is None:
        try:
            _days = dict(json.loads(_file().read_text()).get("days") or {})
        except (OSError, ValueError):
            _days = {}
    return _days


def _save() -> None:
    global _dirty_since
    path = _file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"days": _all()}))
    tmp.replace(path)
    _dirty_since = None


def flush() -> None:
    with _lock:
        if _dirty_since is not None:
            _save()


def add(model: str | None, input_tokens: int | None, output_tokens: int | None) -> None:
    """Counts one reply's tokens towards today and its model."""
    global _dirty_since
    if not (input_tokens or output_tokens):
        return
    with _lock:
        day = _all().setdefault(date.today().isoformat(), {"replies": 0, "input_tokens": 0, "output_tokens": 0, "models": {}})
        day["replies"] += 1
        day["input_tokens"] += input_tokens or 0
        day["output_tokens"] += output_tokens or 0
        name = model or "unknown"
        day["models"][name] = day["models"].get(name, 0) + (output_tokens or 0)
        now = time.time()
        if _dirty_since is None:
            _dirty_since = now
        elif now - _dirty_since >= SAVE_EVERY:
            _save()


def _count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _training() -> dict:
    """Runs, the steps trained and the time spent training, from each run's recorded events."""
    from .training import events, runs_dir

    runs = steps = seconds = done = 0
    root = runs_dir()
    for d in root.iterdir() if root.is_dir() else []:
        if not (d / "config.json").is_file():
            continue
        runs += 1
        evs = events(d)
        metrics = [e for e in evs if e.get("event") == "metrics"]
        if metrics:
            steps += metrics[-1].get("step") or 0
        times = [e["t"] for e in evs if isinstance(e.get("t"), (int, float))]
        if len(times) > 1:
            seconds += max(times) - min(times)
        if (d / "adapter").is_dir():
            done += 1
    return {"runs": runs, "finished": done, "steps": steps, "seconds": round(seconds)}


def _chats() -> dict:
    from .chats import db, db_path

    if not db_path().is_file():
        return {"chats": 0, "messages": 0}
    with db() as con:
        chats = con.execute("select count(*) from chats").fetchone()[0]
        messages = con.execute("select count(*) from messages where role = 'assistant' and error = 0").fetchone()[0]
    return {"chats": chats, "messages": messages}


@router.get("")
def stats(days: int = 30):
    with _lock:
        all_days = {k: dict(v, models=dict(v["models"])) for k, v in _all().items()}
    today = date.today()
    recent = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    models: dict[str, int] = {}
    for d in all_days.values():
        for name, n in d["models"].items():
            models[name] = models.get(name, 0) + n
    exports = home() / "exports"
    datasets = home() / "datasets"
    return {
        "tokens": {
            "output": sum(d["output_tokens"] for d in all_days.values()),
            "input": sum(d["input_tokens"] for d in all_days.values()),
            "replies": sum(d["replies"] for d in all_days.values()),
            "since": min(all_days) if all_days else None,
            "days": [{"date": k, "output_tokens": all_days.get(k, {}).get("output_tokens", 0), "replies": all_days.get(k, {}).get("replies", 0)} for k in recent],
            "models": sorted(({"model": m, "output_tokens": n} for m, n in models.items()), key=lambda x: -x["output_tokens"])[:5],
        },
        **_chats(),
        "training": _training(),
        "exports": sum(1 for d in exports.iterdir() if d.is_dir()) if exports.is_dir() else 0,
        "datasets": sum(1 for p in datasets.glob("*.jsonl") if not p.name.endswith(".eval.jsonl")) if datasets.is_dir() else 0,
        "dataset_rows": sum(_count_lines(p) for p in datasets.glob("*.jsonl") if not p.name.endswith(".eval.jsonl")) if datasets.is_dir() else 0,
    }
