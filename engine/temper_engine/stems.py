"""Separating music into stems with Demucs: vocals, drums, bass and the rest, plus the song without its vocals.

Demucs (Meta, MIT) needs PyTorch, so it installs on request into its own environment under engines/audio,
never touching the engine's. Its HTDemucs model is a Hugging Face download (adefossez/HTDemucs, 84 MB).
Each song is a job running stems_worker.py; the stems are saved as WAV files under stems/ in Temper's home.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import engines, home, storage
from .jobs import registry

router = APIRouter(prefix="/api/stems", tags=["speech"])

REPO = "adefossez/HTDemucs"
PACKAGES = ["demucs==4.1.0", "numpy", "av>=12"]
WORKER = Path(__file__).with_name("stems_worker.py")
ORDER = ("vocals", "instrumental", "drums", "bass", "other")


def venv() -> Path:
    return home() / "engines" / "audio"


def python() -> str | None:
    p = engines._venv_python(venv())
    return str(p) if p.is_file() and (venv() / ".installed").is_file() else None


def downloaded() -> bool:
    snapshots = storage.models_dir() / ("models--" + REPO.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any((s / "htdemucs.yaml").is_file() for s in snapshots.iterdir())


def folder() -> Path:
    path = home() / "stems"
    path.mkdir(exist_ok=True)
    return path


def status() -> dict:
    return {"installed": python() is not None, "downloaded": downloaded(), "ready": python() is not None and downloaded(), "repo": REPO, "size": 84_000_000}


def separations() -> list[dict]:
    """Songs split so far, newest first, with their stems in a fixed order."""
    out = []
    for d in folder().iterdir():
        wavs = {p.stem: str(p) for p in d.glob("*.wav")}
        if not d.is_dir() or not wavs:
            continue
        try:
            meta = json.loads((d / "source.json").read_text())
        except (OSError, ValueError):
            meta = {}
        out.append({"id": d.name, "name": meta.get("name") or d.name, "source": meta.get("source"), "created": meta.get("created") or d.stat().st_mtime,
                    "stems": {k: wavs[k] for k in sorted(wavs, key=lambda k: ORDER.index(k) if k in ORDER else len(ORDER))}})
    return sorted(out, key=lambda s: s["created"], reverse=True)


def _check() -> None:
    if python() is None:
        raise HTTPException(409, "Stem separation isn't installed. Install it on the Audio page, or run `temper stems --install`.")
    if not downloaded():
        raise HTTPException(409, f"Demucs's model isn't downloaded. Download it on the Audio page ({REPO}).")


class Separate(BaseModel):
    path: str


@router.get("")
def stems_status():
    return {**status(), "separations": separations()}


@router.post("/install")
def install():
    job = registry.create("engine-install", "Install stem separation (Demucs)")
    registry.run_thread(job, lambda j: engines.install_venv(j, venv(), PACKAGES, "stem separation"))
    return job.summary()


@router.delete("")
def remove():
    shutil.rmtree(venv(), ignore_errors=True)
    return status()


@router.post("/jobs")
def separate(req: Separate):
    _check()
    source = Path(req.path)
    if not source.is_file():
        raise HTTPException(404, f"No file at {req.path}.")
    name = re.sub(r"[^\w.-]+", "-", source.stem).strip("-") or "song"
    out = folder() / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    out.mkdir()
    (out / "source.json").write_text(json.dumps({"name": source.stem, "source": str(source), "created": time.time()}, ensure_ascii=False))
    job = registry.create("stems", f"Separate {source.name}")
    job.progress = 0.0
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    registry.run_command(job, [python(), "-u", str(WORKER), str(source), str(out)], env=env)
    return job.summary()


def _stem_dir(separation: str) -> Path:
    path = folder() / separation
    if not re.fullmatch(r"[\w.-]+", separation) or not path.is_dir():
        raise HTTPException(404, "No such separation.")
    return path


@router.get("/{separation}/{stem}.wav")
def stem_audio(separation: str, stem: str):
    path = _stem_dir(separation) / f"{stem}.wav"
    if not re.fullmatch(r"\w+", stem) or not path.is_file():
        raise HTTPException(404, "No such stem.")
    return FileResponse(path, media_type="audio/wav")


@router.delete("/{separation}")
def delete(separation: str):
    shutil.rmtree(_stem_dir(separation))
    return {"separations": separations()}
