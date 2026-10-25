"""Generation presets: named chat settings (system prompt, sampling, thinking), for one model or any.

Kept in presets.json in Temper's home. A preset can be a model's default, which chat applies whenever
that model loads, ahead of the publisher's recommended settings.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home

router = APIRouter(prefix="/api/presets", tags=["chat"])


def _file() -> Path:
    return home() / "presets.json"


def _load() -> dict:
    try:
        data = json.loads(_file().read_text())
    except (OSError, ValueError):
        data = {}
    return {"presets": list(data.get("presets") or []), "defaults": dict(data.get("defaults") or {})}


def _save(data: dict) -> None:
    _file().write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _find(data: dict, preset_id: str) -> dict:
    for p in data["presets"]:
        if p["id"] == preset_id:
            return p
    raise HTTPException(404, "That preset doesn't exist any more.")


@router.get("")
def list_presets(model: str | None = None):
    """Presets for `model` and those for any model, plus the model's default preset."""
    data = _load()
    presets = [p for p in data["presets"] if model is None or p.get("model") in (None, model)]
    return {"presets": sorted(presets, key=lambda p: p["name"].lower()), "default": data["defaults"].get(model) if model else None}


class PresetIn(BaseModel):
    name: str
    model: str | None = None  # None: any model
    settings: dict
    thinking: dict | None = None


def _check_name(data: dict, name: str, model: str | None, skip: str | None = None) -> str:
    name = " ".join(name.split())[:80]
    if not name:
        raise HTTPException(400, "Give the preset a name.")
    if any(p["name"].lower() == name.lower() and p.get("model") == model and p["id"] != skip for p in data["presets"]):
        raise HTTPException(409, f"There's already a preset called {name}{' for this model' if model else ''}.")
    return name


@router.post("")
def create_preset(req: PresetIn):
    data = _load()
    preset = {"id": uuid.uuid4().hex[:12], "name": _check_name(data, req.name, req.model), "model": req.model, "settings": req.settings, "thinking": req.thinking, "created": time.time()}
    data["presets"].append(preset)
    _save(data)
    return preset


@router.put("/{preset_id}")
def update_preset(preset_id: str, req: PresetIn):
    data = _load()
    preset = _find(data, preset_id)
    preset.update(name=_check_name(data, req.name, req.model, skip=preset_id), model=req.model, settings=req.settings, thinking=req.thinking)
    if req.model:  # a default for other models no longer applies once the preset is for one model only
        data["defaults"] = {m: pid for m, pid in data["defaults"].items() if pid != preset_id or m == req.model}
    _save(data)
    return preset


@router.delete("/{preset_id}")
def delete_preset(preset_id: str):
    data = _load()
    data["presets"] = [p for p in data["presets"] if p["id"] != preset_id]
    data["defaults"] = {m: pid for m, pid in data["defaults"].items() if pid != preset_id}
    _save(data)
    return {"ok": True}


class DefaultRequest(BaseModel):
    model: str
    preset: str | None = None  # None: back to the recommended settings


@router.put("")
def set_default(req: DefaultRequest):
    data = _load()
    if req.preset:
        preset = _find(data, req.preset)
        if preset.get("model") not in (None, req.model):
            raise HTTPException(400, "That preset is for another model.")
        data["defaults"][req.model] = req.preset
    else:
        data["defaults"].pop(req.model, None)
    _save(data)
    return {"default": data["defaults"].get(req.model)}
