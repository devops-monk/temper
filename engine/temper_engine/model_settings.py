"""How chat loads each model: context length, KV cache type, flash attention, GPU offload and speculative decoding.

Settings are saved per GGUF path in settings.json ("model_settings"), so they apply however the model
is loaded (the Models page, the library, compare mode). A request's own fields win over saved ones.

Speculative decoding drafts tokens cheaply and has the model check several at once:
- "draft": a small model with the same tokenizer (Qwen3-0.6B for a Qwen3-8B) proposes the tokens.
- "ngram": prompt lookup, with no extra model; it helps when the reply repeats the prompt (editing
  code, summarising, extracting).
- "mtp": the model's own multi-token-prediction layers, for models that ship them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import memory, prefs

router = APIRouter(prefix="/api/runtime/settings", tags=["runtime"])

CacheType = Literal["f16", "q8_0", "q4_0"]
Speculation = Literal["none", "draft", "ngram", "mtp"]
# Bytes per cached value, against f16's 2: llama.cpp's q8_0 and q4_0 blocks carry a scale per 32 values.
CACHE_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
DEFAULTS = {"context": 8192, "gpu_layers": 999, "flash_attention": True, "cache_type": "f16", "speculation": "none", "draft": None, "draft_max": 8}


class Settings(BaseModel):
    context: int | None = Field(default=None, ge=512, le=1_048_576)
    gpu_layers: int | None = Field(default=None, ge=0, le=999)
    flash_attention: bool | None = None
    cache_type: CacheType | None = None
    speculation: Speculation | None = None
    draft: str | None = None
    draft_max: int | None = Field(default=None, ge=1, le=64)


def saved(path: str) -> dict:
    """The model's settings with defaults filled in."""
    mine = (prefs.load().get("model_settings") or {}).get(path) or {}
    return {**DEFAULTS, **{k: v for k, v in mine.items() if k in DEFAULTS and v is not None}}


def resolve(path: str, given: dict) -> dict:
    """Saved settings with the fields a load request set explicitly on top."""
    return {**saved(path), **{k: v for k, v in given.items() if k in DEFAULTS and v is not None}}


def flags(s: dict) -> list[str]:
    """llama-server arguments for resolved settings (context, offload and flash attention are added by the runtime)."""
    out: list[str] = []
    if s["cache_type"] != "f16":
        out += ["-ctk", s["cache_type"], "-ctv", s["cache_type"]]
    if s["speculation"] == "draft" and s["draft"]:
        out += ["-md", s["draft"], "--spec-type", "draft-simple", "--spec-draft-n-max", str(s["draft_max"]), "-ngld", str(s["gpu_layers"])]
        if s["cache_type"] != "f16":
            out += ["-ctkd", s["cache_type"], "-ctvd", s["cache_type"]]
    elif s["speculation"] == "ngram":
        out += ["--spec-type", "ngram-mod"]
    elif s["speculation"] == "mtp":
        out += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(s["draft_max"])]
    return out


def check(s: dict) -> None:
    if s["cache_type"] != "f16" and not s["flash_attention"]:
        raise HTTPException(400, "A quantized KV cache needs flash attention. Turn flash attention on, or keep the cache at 16-bit.")
    if s["speculation"] == "draft" and not (s["draft"] and Path(s["draft"]).is_file()):
        raise HTTPException(400, "Choose a draft model that's on this computer, or turn speculative decoding off.")


# --- What a model offers --------------------------------------------------------------------------


def _meta(path: str) -> dict:
    try:
        with Path(path).open("rb") as f:
            return memory.read_gguf_metadata(f.read)
    except (OSError, ValueError):
        return {}


def tokenizer_id(meta: dict) -> tuple | None:
    """What a draft must share with the model: llama.cpp refuses vocabularies that differ by more than 128 tokens."""
    if "tokenizer.ggml.model" not in meta:
        return None
    return meta.get("tokenizer.ggml.model"), meta.get("tokenizer.ggml.pre"), meta.get("tokenizer.ggml.tokens.count") or 0


def compatible(model: dict, draft: dict) -> bool:
    a, b = tokenizer_id(model), tokenizer_id(draft)
    return bool(a and b) and a[:2] == b[:2] and abs(a[2] - b[2]) <= 128


def has_mtp(meta: dict) -> bool:
    arch = meta.get("general.architecture")
    return bool(meta.get(f"{arch}.nextn_predict_layers"))


def _local_ggufs() -> list[dict]:
    from . import files, hub

    out = [{"path": f["path"], "name": f["name"].split("/")[-1], "size": f["size"]} for m in hub.local_models() for f in m["gguf"] if f.get("complete") and f.get("path")]
    for _, info in files._exports():
        out += [{"path": g, "name": Path(g).name, "size": Path(g).stat().st_size} for g in info["gguf"]]
    return out


def drafts_for(path: str) -> list[dict]:
    """Local GGUFs that can draft for the model: the same tokenizer, and under half its size."""
    meta = _meta(path)
    size = Path(path).stat().st_size
    found = []
    for g in _local_ggufs():
        if g["path"] == path or g["size"] >= size / 2:
            continue
        other = _meta(g["path"])
        if compatible(meta, other):
            found.append({**g, "label": other.get("general.name") or g["name"]})
    return sorted(found, key=lambda g: g["size"])


@router.get("")
def get_settings(path: str):
    if not Path(path).is_file():
        raise HTTPException(404, "That model isn't on this computer any more.")
    meta = _meta(path)
    arch = meta.get("general.architecture")
    return {
        "settings": saved(path),
        "context_max": meta.get(f"{arch}.context_length"),
        "layers": meta.get(f"{arch}.block_count"),
        "mtp": has_mtp(meta),
        "drafts": drafts_for(path),
    }


class SaveSettings(BaseModel):
    path: str
    settings: Settings
    # Reload now if this model is the one chat has loaded.
    reload: bool = True


@router.put("")
def put_settings(req: SaveSettings):
    from .runtime import LoadRequest, manager

    merged = {**saved(req.path), **req.settings.model_dump(exclude_none=True)}
    if merged["speculation"] != "draft":
        merged["draft"] = None
    check(merged)
    stored = dict(prefs.load().get("model_settings") or {})
    stored[req.path] = {k: v for k, v in merged.items() if v != DEFAULTS[k]}
    prefs.update("model_settings", {p: s for p, s in stored.items() if s})
    if req.reload and manager.path == req.path and manager.state in ("ready", "loading", "failed"):
        manager.load(LoadRequest(path=req.path, name=manager.model))
    return {"settings": saved(req.path), "status": manager.status()}
