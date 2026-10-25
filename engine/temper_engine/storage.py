"""Disk space: where models live, what Temper uses, and deleting or moving models.

Models stay in the standard Hugging Face cache layout wherever the folder is, so other tools that
use the same folder still see them. A custom folder is applied to huggingface_hub here and passed to
the training and export workers as HF_HUB_CACHE.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from huggingface_hub import constants
from pydantic import BaseModel

from . import home, prefs
from .jobs import Job, registry

router = APIRouter(tags=["storage"])

DEFAULT_DIR = constants.HF_HUB_CACHE  # before apply() changes it; honours HF_HOME / HF_HUB_CACHE
ENGINE_PARTS = ("engine-venv", "python", "llama.cpp", "bin")


def models_dir() -> Path:
    return Path(prefs.load().get("models_dir") or DEFAULT_DIR)


def apply() -> None:
    folder = str(models_dir())
    constants.HF_HUB_CACHE = folder
    os.environ["HF_HUB_CACHE"] = folder


def _size(path: Path) -> int:
    """Bytes on disk under `path`, counting each file once and not following links (the cache is full of them)."""
    total = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            pass
    return total


def _disk(path: Path) -> dict:
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "free": usage.free}


_cache: tuple[float, dict] | None = None
_cache_lock = threading.Lock()


@router.get("/api/storage")
def storage(fresh: bool = False):
    """What's on disk, by kind. Walking the engine's Python environment takes a moment, so results are kept for a minute."""
    global _cache
    with _cache_lock:
        if _cache and not fresh and time.time() - _cache[0] < 60:
            return _cache[1]
        root = home()
        datasets_cache = Path(constants.HF_HOME) / "datasets"
        parts = {
            "models": [models_dir()],
            "exports": [root / "exports", root / "converted"],
            "runs": [root / "runs", root / "recipe-runs"],
            "datasets": [root / "datasets", root / "prepare", root / "uploads", datasets_cache],
            "engine": [root / p for p in ENGINE_PARTS],
        }
        with ThreadPoolExecutor(8) as pool:
            sizes = {k: sum(pool.map(_size, paths)) for k, paths in parts.items()}
        result = {
            "models_dir": str(models_dir()),
            "default_dir": DEFAULT_DIR,
            "custom": str(models_dir()) != DEFAULT_DIR,
            "home": str(root),
            "disk": _disk(models_dir()),
            "home_disk": _disk(root),
            "usage": sizes,
            "datasets_cache": str(datasets_cache),
        }
        _cache = (time.time(), result)
        return result


def _forget_sizes() -> None:
    global _cache
    _cache = None


# --- Deleting -----------------------------------------------------------------------------------


def _repo_dir(repo_id: str) -> Path:
    folder = (models_dir() / ("models--" + repo_id.replace("/", "--"))).resolve()
    if folder.parent != models_dir().resolve() or not folder.is_dir():
        raise HTTPException(404, f"{repo_id} isn't in the models folder.")
    return folder


def _check_not_busy(folder: Path) -> None:
    from .runtime import manager

    # The loaded path goes through the snapshot link, so compare the path itself, not where it resolves.
    if manager.path and Path(os.path.abspath(manager.path)).is_relative_to(folder):
        raise HTTPException(409, "That model is loaded in chat. Unload it first.")
    name = folder.name.removeprefix("models--").replace("--", "/")
    for job in registry.jobs.values():
        if job.kind == "download" and job.status == "running" and (job.result.get("request") or {}).get("repo_id") == name:
            raise HTTPException(409, f"{name} is still downloading. Pause or cancel it first.")


@router.delete("/api/models/local")
def delete_repo(repo_id: str):
    folder = _repo_dir(repo_id)
    _check_not_busy(folder)
    freed = _size(folder)
    shutil.rmtree(folder)
    _forget_sizes()
    return {"freed": freed}


class DeleteFiles(BaseModel):
    repo_id: str
    files: list[str]


@router.post("/api/models/local/delete-files")
def delete_files(req: DeleteFiles):
    """Deletes some files of a downloaded model (one quant, or every part of a split GGUF) and frees
    their blobs once no snapshot links to them."""
    folder = _repo_dir(req.repo_id)
    _check_not_busy(folder)
    snapshots = folder / "snapshots"
    # The library lists files by name; split GGUFs often sit in a folder per quant.
    targets = [p for name in req.files for p in snapshots.rglob(Path(name).name) if p.resolve().is_relative_to(folder)]
    if not targets:
        raise HTTPException(404, "Those files aren't in this model.")
    blobs = {p.resolve() for p in targets}
    for p in targets:
        p.unlink()
    still_used = {p.resolve() for p in snapshots.rglob("*") if p.is_symlink() or p.is_file()}
    freed = 0
    for blob in blobs - still_used:
        if blob.is_relative_to(folder / "blobs") and blob.exists():
            freed += blob.stat().st_size
            blob.unlink()
    _forget_sizes()
    return {"freed": freed}


# --- Moving -------------------------------------------------------------------------------------


class ModelsDir(BaseModel):
    path: str
    move: bool = True


def _move(job: Job, source: Path, target: Path) -> None:
    repos = [p for p in source.iterdir() if p.is_dir() and p.name.startswith(("models--", "datasets--"))] if source.is_dir() else []
    total = sum(_size(p) for p in repos) or 1
    done = 0
    for repo in repos:
        if job.cancel_requested:
            return
        dest = target / repo.name
        if dest.exists():
            job.log(f"Skipped {repo.name}: it's already in the new folder")
            continue
        job.log(f"Moving {repo.name.removeprefix('models--').replace('--', '/')}")
        size = _size(repo)
        try:
            os.rename(repo, dest)  # same disk: instant
        except OSError:
            # Another disk: copy (links stay links, so the cache stays the same size), then delete.
            shutil.copytree(repo, dest, symlinks=True)
            shutil.rmtree(repo)
        done += size
        job.progress = done / total
        job.emit("progress", value=job.progress, received=done, total=total)


@router.put("/api/storage/models-dir")
def set_models_dir(req: ModelsDir):
    target = Path(req.path).expanduser()
    if not target.is_absolute():
        raise HTTPException(400, "Choose a folder by its full path.")
    current = models_dir()
    if target.resolve() == current.resolve():
        raise HTTPException(400, "Models are already stored there.")
    if target.resolve().is_relative_to(current.resolve()):
        raise HTTPException(400, "Choose a folder outside the current models folder.")
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".temper-write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise HTTPException(400, f"Temper can't write to {target} ({e.strerror or e}).") from e

    def switch() -> None:
        prefs.update("models_dir", None if str(target) == DEFAULT_DIR else str(target))
        apply()
        _forget_sizes()

    if not req.move:
        switch()
        return {"models_dir": str(target), "job": None}
    for folder in current.glob("models--*"):
        _check_not_busy(folder)
    need = sum(_size(p) for p in current.glob("models--*")) if current.is_dir() else 0
    same_disk = current.exists() and os.stat(current).st_dev == os.stat(target).st_dev
    if not same_disk and need > shutil.disk_usage(target).free:
        raise HTTPException(400, f"The new folder's disk has {shutil.disk_usage(target).free / 1e9:.1f} GB free; your models need {need / 1e9:.1f} GB.")

    job = registry.create("move", f"Move models to {target}")
    job.progress = 0.0

    def run(j: Job) -> None:
        _move(j, current, target)
        if not j.cancel_requested:
            switch()

    registry.run_thread(job, run)
    return {"models_dir": str(target), "job": job.summary()}
