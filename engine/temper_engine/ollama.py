"""Models already downloaded with Ollama, added to the library without copying them.

Ollama stores each model as an OCI-style manifest whose layers are content-addressed blobs: the
weights are a plain GGUF (llama.cpp loads it from the blob path), and small layers hold the
Modelfile's parameters (JSON) and system prompt. Temper imports the GGUF and keeps those settings for
chat. Ollama's Go chat template is skipped; the GGUF carries its own template, which llama.cpp uses.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import library

router = APIRouter(prefix="/api/ollama", tags=["models"])

MODEL = "application/vnd.ollama.image.model"
PARAMS = "application/vnd.ollama.image.params"
SYSTEM = "application/vnd.ollama.image.system"
PROJECTOR = "application/vnd.ollama.image.projector"
DEFAULT_HOST = "registry.ollama.ai"
# Ollama's parameter names → the catalog's (chat maps those onto its sliders).
SAMPLING = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "repeat_penalty": "repetition_penalty", "presence_penalty": "presence_penalty"}


def models_dirs() -> list[Path]:
    """Where Ollama keeps models: OLLAMA_MODELS, the user's folder, and the Linux service's folder."""
    dirs = [Path(os.environ["OLLAMA_MODELS"])] if os.environ.get("OLLAMA_MODELS") else []
    dirs.append(Path.home() / ".ollama" / "models")
    if platform.system() == "Linux":
        dirs.append(Path("/usr/share/ollama/.ollama/models"))
    return [d for d in dirs if (d / "manifests").is_dir()]


def _blob(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


def _name(manifest: Path, manifests: Path) -> str:
    """host/namespace/model/tag → "model:tag" for the official library, else "namespace/model:tag"."""
    host, namespace, *model, tag = manifest.relative_to(manifests).parts
    name = "/".join(model)
    if namespace != "library":
        name = f"{namespace}/{name}"
    if host != DEFAULT_HOST:
        name = f"{host}/{name}"
    return f"{name}:{tag}"


def _read(path: Path, limit: int = 1 << 20) -> str | None:
    try:
        return path.read_text()[:limit] if path.stat().st_size <= limit else None
    except OSError:
        return None


def scan() -> list[dict]:
    imported = {i["path"] for i in library._imports()}
    found = []
    for root in models_dirs():
        manifests = root / "manifests"
        for manifest in sorted(p for p in manifests.rglob("*") if p.is_file()):
            try:
                layers = json.loads(manifest.read_text()).get("layers", [])
                name = _name(manifest, manifests)
            except (OSError, ValueError, TypeError):
                continue
            by_type = {layer.get("mediaType"): layer for layer in layers}
            model = by_type.get(MODEL)
            if not model:
                continue
            path = _blob(root, model["digest"])
            if not library._is_gguf(path):
                continue
            params = {}
            if PARAMS in by_type:
                try:
                    params = json.loads(_read(_blob(root, by_type[PARAMS]["digest"])) or "{}")
                except ValueError:
                    pass
            system = _read(_blob(root, by_type[SYSTEM]["digest"])) if SYSTEM in by_type else None
            found.append({
                "name": name,
                "path": str(path),
                "size": model.get("size") or path.stat().st_size,
                "sampling": {ours: params[theirs] for theirs, ours in SAMPLING.items() if isinstance(params.get(theirs), (int, float))},
                "context": params.get("num_ctx") if isinstance(params.get("num_ctx"), int) else None,
                "system": (system or "").strip() or None,
                "vision": PROJECTOR in by_type,
                "imported": str(path) in imported,
            })
    return found


@router.get("")
def list_models():
    return {"dirs": [str(d) for d in models_dirs()], "models": scan()}


class ImportRequest(BaseModel):
    names: list[str]


@router.post("/import")
def import_models(req: ImportRequest):
    models = {m["name"]: m for m in scan()}
    missing = [n for n in req.names if n not in models]
    if missing:
        raise HTTPException(404, f"Ollama doesn't have {', '.join(missing)} any more.")
    for name in req.names:
        m = models[name]
        library.add_entry(m["path"], name, settings={"source": "Ollama", "sampling": m["sampling"], "system": m["system"], "context": m["context"]})
    return {"imported": req.names}
