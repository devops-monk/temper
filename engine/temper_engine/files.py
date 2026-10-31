"""The library of files Temper keeps: uploads, datasets, fine-tuned runs, exports, converted models, chat
attachments, generated pictures, saved speech and separated stems.

Each kind lives in its own folder under Temper's home, where the feature that made it put it. This
lists them in one place and keeps what the person adds on top in files.json: a display name,
favourites and folders. Renaming changes only the display name, because runs and exports refer to
their datasets and adapters by path. Later features (sandbox files) add a kind to KINDS.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home
from .jobs import registry

router = APIRouter(prefix="/api/files", tags=["files"])

DATASET_SUFFIXES = {".jsonl", ".json", ".csv"}


def _meta_file() -> Path:
    return home() / "files.json"


def _load_meta() -> dict:
    try:
        data = json.loads(_meta_file().read_text())
    except (OSError, ValueError):
        data = {}
    return {"folders": list(data.get("folders") or []), "items": dict(data.get("items") or {})}


def _save_meta(meta: dict) -> None:
    _meta_file().write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def _id(path: Path) -> str:
    return path.relative_to(home()).as_posix()


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# --- Kinds: each yields (path, extra fields) for the items it finds --------------------------------


def _uploads() -> Iterator[tuple[Path, dict]]:
    root = home() / "uploads"
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_file() and not p.name.startswith("."):
            yield p, {"dataset": p.suffix.lower() in DATASET_SUFFIXES}


def _datasets() -> Iterator[tuple[Path, dict]]:
    """Prepared and generated datasets; an `x.eval.jsonl` is shown as part of `x.jsonl`."""
    root = home() / "datasets"
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in DATASET_SUFFIXES) if root.is_dir() else []
    names = {p.name for p in files}
    for p in files:
        if p.name.endswith(".eval.jsonl") and p.name.removesuffix(".eval.jsonl") + ".jsonl" in names:
            continue
        evals = p.with_name(p.name.removesuffix(".jsonl") + ".eval.jsonl")
        parts = [p] + ([evals] if p.suffix == ".jsonl" and evals.is_file() and evals != p else [])
        yield p, {"dataset": True, "parts": [str(x) for x in parts], "size": sum(x.stat().st_size for x in parts)}


def _runs() -> Iterator[tuple[Path, dict]]:
    root = home() / "runs"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        cfg = _read_json(d / "config.json")
        if not cfg:
            continue
        yield d, {
            "title": cfg.get("name"),
            "base_model": cfg.get("source_id") or cfg.get("base_model"),
            "trained_on": cfg.get("dataset"),
            "adapter": (d / "adapter").is_dir(),
            "backend": cfg.get("backend"),
        }


def _exports() -> Iterator[tuple[Path, dict]]:
    root = home() / "exports"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir():
            # Named after the run it came from, which is easier to recognise than the folder's timestamps.
            run_id = _read_json(d / "export.json").get("run_id")
            run_name = _read_json(home() / "runs" / run_id / "config.json").get("name") if run_id else None
            yield d, {"title": run_name or run_id, "gguf": sorted(str(p) for p in d.rglob("*.gguf")), "formats": sorted(p.name for p in d.iterdir() if p.is_dir())}


def _converted() -> Iterator[tuple[Path, dict]]:
    root = home() / "converted"
    for p in sorted(root.glob("*/*.gguf")) if root.is_dir() else []:
        if not p.name.startswith("."):
            yield p, {"gguf": [str(p)]}


def _attachments() -> Iterator[tuple[Path, dict]]:
    """Files sent with chat messages; the file itself, named as it was attached."""
    root = home() / "attachments"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        info = _read_json(d / "meta.json")
        if info.get("file") and (d / info["file"]).is_file():
            yield d / info["file"], {"title": Path(info["name"]).stem, "attached": info.get("kind")}


def _pictures() -> Iterator[tuple[Path, dict]]:
    """Pictures made on the Images page, named after their prompt; the settings beside each go with it."""
    root = home() / "images"
    for p in sorted(root.glob("*.png")) if root.is_dir() else []:
        info = _read_json(p.with_suffix(".json"))
        yield p, {"title": (info.get("prompt") or p.stem)[:80], "parts": [str(p), str(p.with_suffix(".json"))], "size": p.stat().st_size, "made_with": info.get("model")}


def _speech() -> Iterator[tuple[Path, dict]]:
    """Clips saved from text to speech, named after what they say."""
    root = home() / "voice"
    for p in sorted(root.glob("*.wav")) if root.is_dir() else []:
        info = _read_json(p.with_suffix(".json"))
        yield p, {"title": (info.get("text") or p.stem)[:80], "parts": [str(p), str(p.with_suffix(".json"))], "size": p.stat().st_size, "voice": info.get("voice")}


def _stems() -> Iterator[tuple[Path, dict]]:
    """Songs split into stems: the folder of WAV files, named after the song."""
    root = home() / "stems"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        wavs = sorted(p.stem for p in d.glob("*.wav")) if d.is_dir() else []
        if wavs:
            yield d, {"title": _read_json(d / "source.json").get("name"), "stems": wavs}


KINDS: dict[str, Callable[[], Iterator[tuple[Path, dict]]]] = {
    "upload": _uploads,
    "dataset": _datasets,
    "fine-tune": _runs,
    "export": _exports,
    "model": _converted,
    "attachment": _attachments,
    "picture": _pictures,
    "speech": _speech,
    "stems": _stems,
}


def _items() -> list[dict]:
    meta = _load_meta()
    out = []
    for kind, find in KINDS.items():
        for path, extra in find():
            item_id = _id(path)
            mine = meta["items"].get(item_id, {})
            original = extra.pop("title", None) or (path.stem if path.is_file() else path.name)
            out.append({
                "id": item_id,
                "kind": kind,
                "name": mine.get("name") or original,
                "original": original,
                "path": str(path),
                "size": extra.pop("size", None) or _size(path),
                "modified": path.stat().st_mtime,
                "favourite": bool(mine.get("favourite")),
                "folder": mine.get("folder") if mine.get("folder") in meta["folders"] else None,
                **extra,
            })
    return out


def _find(item_id: str) -> dict:
    for item in _items():
        if item["id"] == item_id:
            return item
    raise HTTPException(404, "That file isn't in the library any more.")


@router.get("")
def list_files():
    return {"folders": _load_meta()["folders"], "items": sorted(_items(), key=lambda i: i["modified"], reverse=True)}


class ItemUpdate(BaseModel):
    id: str
    name: str | None = None
    favourite: bool | None = None
    folder: str | None = None  # "" takes it out of its folder


@router.patch("")
def update_item(req: ItemUpdate):
    item = _find(req.id)
    meta = _load_meta()
    mine = meta["items"].setdefault(req.id, {})
    if req.name is not None:
        name = req.name.strip()
        if name and name != item["original"]:
            mine["name"] = name[:200]
        else:
            mine.pop("name", None)  # empty or the original: back to the file's own name
    if req.favourite is not None:
        if req.favourite:
            mine["favourite"] = True
        else:
            mine.pop("favourite", None)
    if req.folder is not None:
        if req.folder and req.folder not in meta["folders"]:
            raise HTTPException(404, f"There's no folder called {req.folder}.")
        if req.folder:
            mine["folder"] = req.folder
        else:
            mine.pop("folder", None)
    if not mine:
        del meta["items"][req.id]
    _save_meta(meta)
    return _find(req.id)


# --- Folders --------------------------------------------------------------------------------------


class FolderRequest(BaseModel):
    name: str
    new_name: str | None = None


def _folder_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip()
    if not name:
        raise HTTPException(400, "Give the folder a name.")
    return name[:80]


@router.post("/folders")
def add_folder(req: FolderRequest):
    meta = _load_meta()
    name = _folder_name(req.name)
    if name in meta["folders"]:
        raise HTTPException(409, f"There's already a folder called {name}.")
    meta["folders"].append(name)
    _save_meta(meta)
    return {"folders": meta["folders"]}


@router.put("/folders")
def rename_folder(req: FolderRequest):
    meta = _load_meta()
    if req.name not in meta["folders"]:
        raise HTTPException(404, f"There's no folder called {req.name}.")
    new = _folder_name(req.new_name or "")
    if new != req.name and new in meta["folders"]:
        raise HTTPException(409, f"There's already a folder called {new}.")
    meta["folders"] = [new if f == req.name else f for f in meta["folders"]]
    for mine in meta["items"].values():
        if mine.get("folder") == req.name:
            mine["folder"] = new
    _save_meta(meta)
    return {"folders": meta["folders"]}


@router.delete("/folders")
def delete_folder(name: str):
    """Removes a folder; what was in it stays in the library, outside any folder."""
    meta = _load_meta()
    meta["folders"] = [f for f in meta["folders"] if f != name]
    for item_id, mine in list(meta["items"].items()):
        if mine.get("folder") == name:
            mine.pop("folder")
            if not mine:
                del meta["items"][item_id]
    _save_meta(meta)
    return {"folders": meta["folders"]}


# --- Adding and deleting --------------------------------------------------------------------------


class UploadRequest(BaseModel):
    paths: list[str]


def _free_name(folder: Path, name: str) -> Path:
    target = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while target.exists():
        target = folder / f"{stem} ({n}){suffix}"
        n += 1
    return target


@router.post("/upload")
def upload(req: UploadRequest):
    """Copies files into Temper's uploads folder, so they stay available if the originals move."""
    root = home() / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(p).expanduser() for p in req.paths]
    for p in sources:
        if not p.is_file():
            raise HTTPException(400, f"{p.name} isn't a file Temper can copy." if p.exists() else f"Temper can't find {p}.")
    added = []
    for p in sources:
        target = _free_name(root, p.name)
        shutil.copy2(p, target)
        added.append(_id(target))
    return {"added": added}


def _check_not_busy(item: dict) -> None:
    from .compare import loaded_paths

    path = Path(item["path"])
    if any(Path(os.path.abspath(p)).is_relative_to(path) for p in loaded_paths()):
        raise HTTPException(409, "That model is loaded in chat or compare mode. Unload it first.")
    for job in registry.jobs.values():
        if job.status != "running":
            continue
        used = {job.result.get("run_id") and str(home() / "runs" / job.result["run_id"]), job.result.get("output_dir")}
        if str(path) in used:
            raise HTTPException(409, f"{item['name']} is still being written by “{job.title}”. Wait for it to finish or cancel it.")


@router.delete("")
def delete_item(id: str):
    item = _find(id)
    _check_not_busy(item)
    path = Path(item["path"])
    freed = item["size"]
    if path.is_dir():
        shutil.rmtree(path)
    elif item["kind"] == "attachment":
        shutil.rmtree(path.parent)  # with its meta.json and extracted text; chats that sent it keep its name
    else:
        for part in item.get("parts") or [str(path)]:
            Path(part).unlink(missing_ok=True)
        if item["kind"] == "model":
            from . import library

            library._save([i for i in library._imports() if i["path"] != str(path)])
            if not any(path.parent.iterdir()):
                path.parent.rmdir()
    meta = _load_meta()
    if meta["items"].pop(id, None) is not None:
        _save_meta(meta)
    return {"freed": freed}
