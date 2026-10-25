"""Importing conversations from Claude Code, Codex and Cursor into Temper's chat history.

Each tool keeps its history on this computer: Claude Code as one JSONL file per session under
~/.claude/projects, Codex as rollout JSONL files under ~/.codex/sessions, and Cursor in the SQLite
database behind its editor state. Only what people and the assistant said is kept: tool calls, tool
output, thinking and the context the tools inject (system reminders, environment blocks, AGENTS.md)
are left out, so an imported chat reads like a conversation and can be continued with a local model.
Importing again updates chats that grew and leaves the title, folder and pin people set.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .chats import db, title_from, write_messages
from .jobs import Job, registry

router = APIRouter(prefix="/api/chats/import", tags=["chats"])

# Text a tool adds to a turn on the person's behalf, not something they wrote.
INJECTED = ("<task-notification>", "<bash-input>", "<bash-stdout>", "<bash-stderr>", "<command-name>", "<command-message>", "<local-command-", "Caveat: The messages below",
            "<environment_context>", "<user_instructions>", "<permissions instructions>", "# AGENTS.md instructions", "<user_shell_command>", "<turn_aborted>", "[Request interrupted by user")
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _clean(text: str) -> str:
    return REMINDER.sub("", text).strip()


def _injected(text: str) -> bool:
    return text.lstrip().startswith(INJECTED)


def _timestamp(value) -> float | None:
    if isinstance(value, (int, float)):
        return value / 1000 if value > 1e11 else float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def merge(messages: list[dict]) -> list[dict]:
    """Joins consecutive turns from the same side (an agent's reply is often split around tool calls)."""
    out: list[dict] = []
    for m in messages:
        if not m["content"].strip():
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n\n" + m["content"]
        else:
            out.append(dict(m))
    return out


# --- Claude Code ----------------------------------------------------------------------------------


def claude_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"


def _claude_files() -> list[Path]:
    root = claude_dir()
    return sorted(root.glob("*/*.jsonl")) if root.is_dir() else []


def _blocks_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def parse_claude(path: Path) -> dict | None:
    messages, title, project, model = [], None, None, None
    stamps = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        kind = d.get("type")
        if kind == "ai-title":
            title = d.get("aiTitle") or title
        elif kind == "summary" and not title:
            title = d.get("summary")
        if kind not in ("user", "assistant") or d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
            continue
        msg = d.get("message") or {}
        text = _clean(_blocks_text(msg.get("content")))
        if not text or (kind == "user" and _injected(text)):
            continue
        project = project or (Path(d["cwd"]).name if d.get("cwd") else None)
        if kind == "assistant":
            model = msg.get("model") or model
        stamp = _timestamp(d.get("timestamp"))
        stamps += [stamp] if stamp else []
        messages.append({"role": kind, "content": text, "model": msg.get("model") if kind == "assistant" else None, "created": stamp})
    messages = merge(messages)
    if not any(m["role"] == "user" for m in messages):
        return None
    return {"ref": path.stem, "title": title, "project": project, "model": model, "messages": messages, "created": min(stamps, default=path.stat().st_mtime), "updated": max(stamps, default=path.stat().st_mtime)}


# --- Codex ----------------------------------------------------------------------------------------


def codex_dir() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _codex_files() -> list[Path]:
    root = codex_dir()
    return sorted(p for folder in ("sessions", "archived_sessions") if (root / folder).is_dir() for p in (root / folder).rglob("rollout-*.jsonl"))


def parse_codex(path: Path) -> dict | None:
    """Rollout files: `{type: session_meta | response_item | …, payload}` lines, or the older format
    where each line is the item itself."""
    messages, project, model, ref = [], None, None, None
    stamps = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        stamp = _timestamp(d.get("timestamp"))
        item = d.get("payload") if "payload" in d else d
        if not isinstance(item, dict):
            continue
        if d.get("type") == "session_meta" or ("id" in item and "instructions" in item and "role" not in item):
            ref = item.get("id") or ref
            project = Path(item["cwd"]).name if item.get("cwd") else project
            continue
        if d.get("type") == "turn_context":
            model = item.get("model") or model
            continue
        if item.get("type") != "message" or item.get("role") not in ("user", "assistant"):
            continue
        text = _clean("\n\n".join(c.get("text", "") for c in item.get("content") or [] if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text")))
        if not text or (item["role"] == "user" and _injected(text)):
            continue
        stamps += [stamp] if stamp else []
        messages.append({"role": item["role"], "content": text, "model": model if item["role"] == "assistant" else None, "created": stamp})
    messages = merge(messages)
    if not any(m["role"] == "user" for m in messages):
        return None
    mtime = path.stat().st_mtime
    return {"ref": ref or path.stem, "title": None, "project": project, "model": model, "messages": messages, "created": min(stamps, default=mtime), "updated": max(stamps, default=mtime)}


# --- Cursor ---------------------------------------------------------------------------------------


def cursor_db() -> Path:
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "Cursor" / "User" / "globalStorage" / "state.vscdb"


def _cursor_open() -> sqlite3.Connection | None:
    path = cursor_db()
    if not path.is_file():
        return None
    # Read-only and immutable: Cursor may have the database open, and Temper must never change it.
    return sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)


def _cursor_composers(con: sqlite3.Connection) -> Iterator[dict]:
    try:
        rows = con.execute("select value from cursorDiskKV where key like 'composerData:%'")
    except sqlite3.DatabaseError:
        return
    for (value,) in rows:
        try:
            data = json.loads(value) if value else None
        except ValueError:
            continue
        if isinstance(data, dict) and (data.get("fullConversationHeadersOnly") or data.get("conversation")):
            yield data


def parse_cursor(con: sqlite3.Connection, data: dict) -> dict | None:
    """A Composer chat: bubbles inline (older versions) or stored under bubbleId:<chat>:<bubble> keys.
    Bubble type 1 is the person, 2 the assistant."""
    cid = data.get("composerId")
    bubbles = data.get("conversation") or []
    if not bubbles:
        for header in data.get("fullConversationHeadersOnly") or []:
            row = con.execute("select value from cursorDiskKV where key = ?", (f"bubbleId:{cid}:{header.get('bubbleId')}",)).fetchone()
            try:
                bubbles.append(json.loads(row[0]) if row and row[0] else {})
            except ValueError:
                continue
    messages = []
    for b in bubbles:
        role = {1: "user", 2: "assistant"}.get(b.get("type"))
        text = _clean(b.get("text") or "")
        if role and text:
            model = (b.get("modelInfo") or {}).get("modelName") if role == "assistant" else None
            messages.append({"role": role, "content": text, "model": model, "created": _timestamp(b.get("createdAt"))})
    messages = merge(messages)
    if not cid or not any(m["role"] == "user" for m in messages):
        return None
    created = _timestamp(data.get("createdAt")) or time.time()
    return {"ref": cid, "title": data.get("name"), "project": None, "model": next((m["model"] for m in messages if m.get("model")), None), "messages": messages, "created": created, "updated": _timestamp(data.get("lastUpdatedAt")) or created}


# --- Sources --------------------------------------------------------------------------------------

LABELS = {"claude-code": "Claude Code", "codex": "Codex", "cursor": "Cursor"}


def _sessions(source: str) -> tuple[int, Callable[[], Iterator[dict | None]]]:
    """How many sessions a source has, and a generator that parses them one by one."""
    if source == "claude-code":
        files = _claude_files()
        return len(files), lambda: (parse_claude(p) for p in files)
    if source == "codex":
        files = _codex_files()
        return len(files), lambda: (parse_codex(p) for p in files)
    con = _cursor_open()
    if con is None:
        return 0, lambda: iter(())
    composers = list(_cursor_composers(con))

    def gen():
        try:
            for data in composers:
                yield parse_cursor(con, data)
        finally:
            con.close()

    return len(composers), gen


def _where(source: str) -> str:
    return str({"claude-code": claude_dir(), "codex": codex_dir() / "sessions", "cursor": cursor_db()}[source])


@router.get("/sources")
def sources():
    with db() as con:
        imported = {r["source"]: r["n"] for r in con.execute("select source, count(*) as n from chats where source is not null group by source")}
    out = []
    for source, label in LABELS.items():
        try:
            found, _ = _sessions(source)
        except (OSError, sqlite3.DatabaseError):
            found = 0
        out.append({"source": source, "label": label, "path": _where(source), "found": found, "imported": imported.get(label, 0)})
    return out


def save_session(con: sqlite3.Connection, label: str, s: dict) -> str:
    """Adds or refreshes one imported session; returns "added", "updated" or "unchanged"."""
    row = con.execute("select id, updated, title from chats where source = ? and source_ref = ?", (label, s["ref"])).fetchone()
    if row and row["updated"] >= s["updated"]:
        return "unchanged"
    if row:
        chat_id = row["id"]
        con.execute("update chats set updated = ?, model = coalesce(?, model), project = coalesce(?, project) where id = ?", (s["updated"], s["model"], s["project"], chat_id))
    else:
        chat_id = uuid.uuid4().hex[:16]
        con.execute(
            "insert into chats(id, title, model, created, updated, source, source_ref, project) values (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, (s["title"] or title_from(s["messages"])).strip()[:200], s["model"], s["created"], s["updated"], label, s["ref"], s["project"]),
        )
    write_messages(con, chat_id, s["messages"])
    return "updated" if row else "added"


def run_import(job: Job, source: str) -> None:
    label = LABELS[source]
    total, sessions = _sessions(source)
    counts = {"added": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for i, session in enumerate(sessions()):
        if job.cancel_requested:
            break
        try:
            if session is None:
                counts["skipped"] += 1  # nothing a person said, e.g. a session that only ran a command
            else:
                with db() as con:
                    counts[save_session(con, label, session)] += 1
        except (OSError, sqlite3.DatabaseError, KeyError, TypeError) as e:
            counts["skipped"] += 1
            job.log(f"Skipped one session: {e}")
        job.progress = (i + 1) / max(total, 1)
        job.emit("progress", value=job.progress, **counts)
    job.result.update(counts)
    job.log(f"{label}: {counts['added']} added, {counts['updated']} updated, {counts['unchanged']} already up to date, {counts['skipped']} without a conversation")


class ImportRequest(BaseModel):
    source: str


@router.post("")
def start_import(req: ImportRequest):
    if req.source not in LABELS:
        raise HTTPException(400, f"Import from one of {', '.join(LABELS.values())}.")
    for job in registry.jobs.values():
        if job.kind == "import" and job.status == "running" and job.result.get("source") == req.source:
            raise HTTPException(409, f"Chats from {LABELS[req.source]} are already being imported.")
    job = registry.create("import", f"Import chats from {LABELS[req.source]}")
    job.progress = 0.0
    job.result["source"] = req.source
    registry.run_thread(job, lambda j: run_import(j, req.source))
    return job.summary()
