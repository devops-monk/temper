"""Standing instructions for chat: a global AGENTS.md plus a project folder's own, found the way Codex finds them.

The global file is AGENTS.md in Temper's home, edited in Settings. A chat can name a project folder;
Temper then reads AGENTS.md from the project's root (the nearest folder holding .git, else the folder
itself) down to the folder, one per directory, with AGENTS.override.md taking that directory's place
when present. Files closer to the folder come later, so their instructions win. Together they are
capped at MAX_BYTES, like Codex's project_doc_max_bytes, and go ahead of the chat's system prompt.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home

router = APIRouter(prefix="/api/instructions", tags=["chat"])

MAX_BYTES = 32 * 1024
NAMES = ("AGENTS.override.md", "AGENTS.md")


def global_file() -> Path:
    return home() / "AGENTS.md"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def project_root(folder: Path) -> Path:
    for d in (folder, *folder.parents):
        if (d / ".git").exists():
            return d
    return folder


def project_files(folder: str) -> list[Path]:
    """The AGENTS.md files that apply to `folder`, root first."""
    start = Path(folder).expanduser()
    if not start.is_dir():
        return []
    start = start.resolve()
    root = project_root(start)
    chain = [start, *start.parents]
    dirs = list(reversed(chain[: chain.index(root) + 1]))
    found = []
    for d in dirs:
        doc = next((d / n for n in NAMES if (d / n).is_file()), None)
        if doc:
            found.append(doc)
    return found


def gather(project: str | None) -> dict:
    """The instructions for a chat: [{source, path, text}], the combined text, and whether it was cut to MAX_BYTES."""
    parts = []
    if text := _read(global_file()):
        parts.append({"source": "global", "path": str(global_file()), "text": text})
    for doc in project_files(project) if project else []:
        if text := _read(doc):
            parts.append({"source": "project", "path": str(doc), "text": text})
    combined, used, truncated = [], 0, False
    for p in parts:
        block = p["text"].encode()
        room = MAX_BYTES - used
        if room <= 0:
            truncated = True
            break
        if len(block) > room:
            p = {**p, "text": block[:room].decode(errors="ignore")}
            truncated = True
        combined.append(p["text"])
        used += len(p["text"].encode())
    return {"parts": [{k: v for k, v in p.items() if k != "text"} | {"bytes": len(p["text"].encode())} for p in parts], "text": "\n\n".join(combined), "truncated": truncated}


def with_instructions(system: str, project: str | None) -> str:
    """The system prompt with the standing instructions ahead of it."""
    text = gather(project)["text"]
    return "\n\n".join(t for t in (text, system.strip()) if t)


@router.get("")
def get_instructions(project: str | None = None):
    if project and not Path(project).expanduser().is_dir():
        raise HTTPException(404, "That project folder isn't on this computer any more.")
    out = gather(project)
    return {"global": _read(global_file()), "parts": out["parts"], "truncated": out["truncated"], "max_bytes": MAX_BYTES}


class GlobalInstructions(BaseModel):
    text: str = Field(max_length=MAX_BYTES * 4)


@router.put("/global")
def put_global(req: GlobalInstructions):
    if req.text.strip():
        global_file().write_text(req.text, encoding="utf-8")
    else:
        global_file().unlink(missing_ok=True)
    return get_instructions()
