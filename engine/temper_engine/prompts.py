"""The prompt library: named prompts to reuse in chat, kept in prompts.json in Temper's home.

A prompt can hold {{variables}}; chat asks for their values when the prompt is used. Prompts are
listed most-used first, and a search matches their names and text.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home

router = APIRouter(prefix="/api/prompts", tags=["chat"])

VARIABLE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def _file() -> Path:
    return home() / "prompts.json"


def _load() -> list[dict]:
    try:
        data = json.loads(_file().read_text())
    except (OSError, ValueError):
        return []
    return [p for p in data.get("prompts") or [] if isinstance(p, dict) and p.get("id")]


def _save(prompts: list[dict]) -> None:
    tmp = _file().with_suffix(".tmp")
    tmp.write_text(json.dumps({"prompts": prompts}, indent=2, ensure_ascii=False))
    tmp.replace(_file())


def variables(text: str) -> list[str]:
    """The {{names}} in a prompt, each once, in order."""
    return list(dict.fromkeys(VARIABLE.findall(text)))


def _out(p: dict) -> dict:
    return {**p, "variables": variables(p["text"])}


def _find(prompts: list[dict], prompt_id: str) -> dict:
    for p in prompts:
        if p["id"] == prompt_id:
            return p
    raise HTTPException(404, "That prompt doesn't exist any more.")


class PromptIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=100_000)


@router.get("")
def list_prompts(q: str = ""):
    words = q.lower().split()
    found = [p for p in _load() if all(w in f"{p['name']} {p['text']}".lower() for w in words)]
    found.sort(key=lambda p: (-p.get("uses", 0), -(p.get("used") or p.get("created") or 0)))
    return {"prompts": [_out(p) for p in found]}


@router.post("")
def add_prompt(req: PromptIn):
    prompts = _load()
    name = req.name.strip()
    if any(p["name"].lower() == name.lower() for p in prompts):
        raise HTTPException(409, f"There's already a prompt called {name}.")
    prompt = {"id": uuid.uuid4().hex[:12], "name": name, "text": req.text, "created": time.time(), "used": None, "uses": 0}
    _save([*prompts, prompt])
    return _out(prompt)


@router.put("/{prompt_id}")
def update_prompt(prompt_id: str, req: PromptIn):
    prompts = _load()
    prompt = _find(prompts, prompt_id)
    name = req.name.strip()
    if any(p["name"].lower() == name.lower() and p["id"] != prompt_id for p in prompts):
        raise HTTPException(409, f"There's already a prompt called {name}.")
    prompt.update(name=name, text=req.text)
    _save(prompts)
    return _out(prompt)


@router.post("/{prompt_id}/used")
def mark_used(prompt_id: str):
    """Counts a use, so the prompts used most come first."""
    prompts = _load()
    prompt = _find(prompts, prompt_id)
    prompt.update(uses=prompt.get("uses", 0) + 1, used=time.time())
    _save(prompts)
    return _out(prompt)


@router.delete("/{prompt_id}")
def delete_prompt(prompt_id: str):
    prompts = _load()
    _find(prompts, prompt_id)
    _save([p for p in prompts if p["id"] != prompt_id])
    return {"ok": True}
