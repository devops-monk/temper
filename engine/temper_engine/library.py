"""Models imported from your own files: a GGUF, a folder of GGUFs, a Hugging Face (safetensors) folder or an MLX folder.

Temper remembers where they are rather than copying them, since they're often many gigabytes; the list
lives in library.json. Each import is checked when added and again whenever the library is listed, so
a file that has moved shows as missing instead of failing to load.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home
from .hub import SPLIT, group_gguf
from .memory import read_gguf_metadata

router = APIRouter(prefix="/api/models/import", tags=["models"])

WEIGHTS = ("*.safetensors", "*.bin")
TOKENIZER = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")


def _file() -> Path:
    return home() / "library.json"


def _imports() -> list[dict]:
    try:
        return json.loads(_file().read_text()).get("imports", [])
    except (OSError, ValueError):
        return []


def _save(imports: list[dict]) -> None:
    _file().write_text(json.dumps({"imports": imports}, indent=2))


def _gguf_name(path: Path) -> str:
    """The model's own name from the GGUF header ("" if it has none). Raises ValueError for anything else."""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF")
        f.seek(0)
        try:
            meta = read_gguf_metadata(f.read)
        except struct.error as e:  # the header ends early
            raise OSError("the file is cut short") from e
    return meta.get("general.name") or meta.get("general.basename") or ""


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _ggufs(folder: Path, only: Path | None = None) -> list[dict]:
    """The GGUFs in a folder as library entries; with `only`, just that file (and its split parts)."""
    files = [{"name": p.name, "size": p.stat().st_size, "path": str(p)} for p in sorted(folder.glob("*.gguf")) if _is_gguf(p)]
    groups = group_gguf(files)
    if only:
        key = SPLIT.sub(".gguf", only.name)
        groups = [g for g in groups if SPLIT.sub(".gguf", g["name"]) == key]
    return groups


def _name(path: Path) -> str:
    """A folder's or file's name; for something inside the Hugging Face cache, the repo's name."""
    for part in reversed(path.parts):
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return path.stem if path.is_file() else path.name


def inspect(raw: str) -> dict:
    """What kind of model is at `raw`, or a 400 saying why it can't be used."""
    # absolute, not resolve(): files in the Hugging Face cache are symlinks to extensionless blobs.
    path = Path(os.path.abspath(Path(raw).expanduser()))
    if not path.exists():
        raise HTTPException(400, f"Temper can't find {path}.")
    if (path / "snapshots").is_dir() and (path / "refs" / "main").is_file():  # a cache repo folder: use its current snapshot
        path = path / "snapshots" / (path / "refs" / "main").read_text().strip()
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise HTTPException(400, "Import a .gguf file, or a folder that holds a model (config.json and .safetensors files).")
        try:
            name = _gguf_name(path)
        except ValueError as e:
            raise HTTPException(400, f"{path.name} isn't a GGUF file, despite its name.") from e
        except OSError as e:
            raise HTTPException(400, f"{path.name} looks damaged: Temper couldn't read its header ({e}).") from e
        if "mmproj" in path.name.lower():
            raise HTTPException(400, f"{path.name} is a vision projector, not a model. Import the model's GGUF; Temper finds the projector beside it.")
        gguf = _ggufs(path.parent, only=path)
        return {"kind": "gguf", "name": name or _name(path), "path": str(path), "gguf": gguf, "size": sum(g["size"] for g in gguf)}

    config_file = path / "config.json"
    if config_file.is_file():
        try:
            config = json.loads(config_file.read_text())
        except ValueError as e:
            raise HTTPException(400, f"{path.name}/config.json isn't valid JSON.") from e
        weights = [p for pattern in WEIGHTS for p in path.glob(pattern)]
        if not weights:
            raise HTTPException(400, f"{path.name} has a config.json but no .safetensors weights.")
        kind = "mlx" if "quantization" in config else "safetensors"
        warnings = [] if any((path / t).is_file() for t in TOKENIZER) else ["There are no tokenizer files in this folder, so training and export will fail."]
        return {"kind": kind, "name": _name(path), "path": str(path), "size": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()), "model_type": config.get("model_type"), "warnings": warnings}

    gguf = _ggufs(path)
    if gguf:
        return {"kind": "gguf", "name": _name(path), "path": str(path), "gguf": gguf, "size": sum(g["size"] for g in gguf)}
    if (path / "params.json").is_file():
        raise HTTPException(400, "This is a Mistral-format folder (params.json). Temper can't load that format yet; use the Hugging Face version of the model.")
    raise HTTPException(400, f"{path.name} doesn't hold a model: Temper looks for .gguf files, or config.json with .safetensors weights.")


def entries() -> list[dict]:
    """Imported models in the shape of /api/models/local, re-checked so moved files show as missing."""
    out = []
    for item in _imports():
        try:
            found = inspect(item["path"])
        except HTTPException as e:
            out.append({"id": item["name"], "path": item["path"], "kind": item["kind"], "imported": True, "missing": e.detail, "size": 0, "last_used": item["added"], "gguf": [], "files": 0, "safetensors": False})
            continue
        out.append({
            "id": item["name"],
            "path": found["path"],
            "kind": found["kind"],
            "imported": True,
            "size": found["size"],
            "last_used": item["added"],
            "gguf": found.get("gguf", []),
            "files": len(found.get("gguf", [])) or 1,
            "safetensors": found["kind"] in ("safetensors", "mlx"),
            "warnings": found.get("warnings", []),
        })
    return out


class ImportRequest(BaseModel):
    path: str


@router.post("")
def add(req: ImportRequest):
    found = inspect(req.path)
    imports = [i for i in _imports() if i["path"] != found["path"]]
    imports.insert(0, {"name": found["name"], "path": found["path"], "kind": found["kind"], "added": time.time()})
    _save(imports)
    return found


@router.delete("")
def remove(path: str):
    """Forgets an import. The files stay where they are."""
    _save([i for i in _imports() if i["path"] != path])
    return {"ok": True}
