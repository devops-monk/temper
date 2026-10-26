"""Chat history: threads in SQLite (chats.db in Temper's home), with full-text search, folders, pins and export.

The chat page keeps the conversation in memory while it streams and saves the whole thread after each
turn, so regenerating or editing never leaves stray messages behind. Message text is indexed with
FTS5 for search; titles are matched directly.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home

router = APIRouter(prefix="/api/chats", tags=["chats"])

SCHEMA = """
create table if not exists folders(name text primary key, created real);
create table if not exists chats(
  id text primary key,
  title text not null default '',
  model text,
  folder text,
  pinned integer not null default 0,
  settings text,
  created real not null,
  updated real not null,
  source text,
  source_ref text,
  project text,
  compaction text
);
create unique index if not exists chats_source on chats(source, source_ref);
create table if not exists messages(
  chat_id text not null references chats(id) on delete cascade,
  seq integer not null,
  role text not null,
  content text not null,
  model text,
  error integer not null default 0,
  created real,
  reasoning text,
  attachments text,
  primary key(chat_id, seq)
);
create virtual table if not exists messages_fts using fts5(content, chat_id unindexed, tokenize='unicode61 remove_diacritics 2');
"""
ROLES = {"user", "assistant", "system"}
TITLE_LENGTH = 60
_lock = threading.Lock()


def db_path() -> Path:
    return home() / "chats.db"


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """A connection with the schema in place; commits on success. Writes are serialised by a lock."""
    with _lock:
        con = sqlite3.connect(db_path(), timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("pragma foreign_keys = on")
            con.execute("pragma journal_mode = wal")
            con.executescript(SCHEMA)
            columns = {r["name"] for r in con.execute("pragma table_info(messages)")}
            for column in ("reasoning", "attachments"):  # added with the reasoning view and with attachments
                if column not in columns:
                    con.execute(f"alter table messages add column {column} text")
            if "compaction" not in {r["name"] for r in con.execute("pragma table_info(chats)")}:  # added with auto-compaction
                con.execute("alter table chats add column compaction text")
            yield con
            con.commit()
        finally:
            con.close()


def title_from(messages: list[dict]) -> str:
    """The first line of the first user message, cut at a word near TITLE_LENGTH characters."""
    first = next((m["content"] for m in messages if m["role"] == "user" and m["content"].strip()), "")
    if not first:  # a first message that's only files is named after them
        first = ", ".join(a["name"] for m in messages if m["role"] == "user" for a in (m.get("attachments") or []))
    line = re.sub(r"\s+", " ", first.strip().splitlines()[0] if first.strip() else "").strip()
    if len(line) <= TITLE_LENGTH:
        return line
    cut = line[:TITLE_LENGTH].rsplit(" ", 1)[0]
    return (cut if len(cut) > TITLE_LENGTH // 2 else line[:TITLE_LENGTH]).rstrip(" ,.;:") + "…"


def _summary(row: sqlite3.Row) -> dict:
    keys = row.keys()
    return {
        "id": row["id"],
        "title": row["title"] or "New chat",
        "model": row["model"],
        "folder": row["folder"],
        "pinned": bool(row["pinned"]),
        "created": row["created"],
        "updated": row["updated"],
        "source": row["source"],
        "project": row["project"],
        "messages": row["count"] if "count" in keys else None,
    }


def _get(con: sqlite3.Connection, chat_id: str) -> sqlite3.Row:
    row = con.execute("select * from chats where id = ?", (chat_id,)).fetchone()
    if not row:
        raise HTTPException(404, "That chat doesn't exist any more.")
    return row


def write_messages(con: sqlite3.Connection, chat_id: str, messages: list[dict]) -> None:
    """Replaces a chat's messages and their search index."""
    con.execute("delete from messages where chat_id = ?", (chat_id,))
    con.execute("delete from messages_fts where chat_id = ?", (chat_id,))
    now = time.time()
    con.executemany(
        "insert into messages(chat_id, seq, role, content, model, error, created, reasoning, attachments) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (chat_id, i, m["role"], m["content"], m.get("model"), int(bool(m.get("error"))), m.get("created") or now, m.get("reasoning") or None, json.dumps(m["attachments"]) if m.get("attachments") else None)
            for i, m in enumerate(messages)
        ],
    )
    con.executemany("insert into messages_fts(content, chat_id) values (?, ?)", [(m["content"], chat_id) for m in messages if not m.get("error")])


def _fts_query(text: str) -> str:
    """Each word as a quoted prefix term, so punctuation in what people type can't break FTS syntax."""
    return " ".join(f'"{w.replace(chr(34), "")}"*' for w in text.split() if w.replace('"', ""))


# --- Chats ----------------------------------------------------------------------------------------


@router.get("")
def list_chats(q: str = "", folder: str | None = None, source: str | None = None, limit: int = 500):
    with db() as con:
        where, args = [], []
        if folder is not None:
            where.append("c.folder = ?")
            args.append(folder)
        if source is not None:
            where.append("c.source is ?" if source == "" else "c.source = ?")
            args.append(None if source == "" else source)
        snippets: dict[str, str] = {}
        if q.strip():
            ids = {r["id"] for r in con.execute("select id from chats where title like ?", (f"%{q.strip()}%",))}
            try:
                for r in con.execute(
                    "select chat_id, snippet(messages_fts, 0, '[[', ']]', '…', 10) as s from messages_fts where messages_fts match ? limit 2000",
                    (_fts_query(q),),
                ):
                    ids.add(r["chat_id"])
                    snippets.setdefault(r["chat_id"], r["s"])
            except sqlite3.OperationalError:
                pass  # a query FTS can't parse still matches titles
            if not ids:
                return {"chats": [], "folders": _folders(con)}
            where.append(f"c.id in ({','.join('?' * len(ids))})")
            args.extend(ids)
        sql = (
            "select c.*, (select count(*) from messages m where m.chat_id = c.id) as count from chats c"
            + (f" where {' and '.join(where)}" if where else "")
            + " order by c.pinned desc, c.updated desc limit ?"
        )
        rows = con.execute(sql, (*args, limit)).fetchall()
        chats = [{**_summary(r), **({"snippet": snippets[r["id"]]} if r["id"] in snippets else {})} for r in rows]
        sources = {r["source"]: r["n"] for r in con.execute("select source, count(*) as n from chats where source is not null group by source")}
        return {"chats": chats, "folders": _folders(con), "sources": sources}


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    model: str | None = None
    error: bool = False
    # What a thinking model reasoned before answering.
    reasoning: str | None = None
    # Files sent with the message: [{id, name, kind}], kept in attachments/ in Temper's home.
    attachments: list[dict] | None = None


class NewChat(BaseModel):
    title: str | None = None
    model: str | None = None
    settings: dict | None = None
    folder: str | None = None
    messages: list[Message] = []


@router.post("")
def create_chat(req: NewChat):
    chat_id = uuid.uuid4().hex[:16]
    now = time.time()
    messages = [m.model_dump() for m in req.messages]
    with db() as con:
        if req.folder:
            _check_folder(con, req.folder)
        con.execute(
            "insert into chats(id, title, model, folder, settings, created, updated) values (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, (req.title or title_from(messages)).strip(), req.model, req.folder, json.dumps(req.settings) if req.settings else None, now, now),
        )
        write_messages(con, chat_id, messages)
        return _with_messages(con, chat_id)


def _with_messages(con: sqlite3.Connection, chat_id: str) -> dict:
    row = _get(con, chat_id)
    messages = [
        {"role": m["role"], "content": m["content"], "model": m["model"], "error": bool(m["error"]), "created": m["created"], "reasoning": m["reasoning"], "attachments": json.loads(m["attachments"]) if m["attachments"] else None}
        for m in con.execute("select * from messages where chat_id = ? order by seq", (chat_id,))
    ]
    return {**_summary(row), "messages": messages, "settings": json.loads(row["settings"]) if row["settings"] else None, "compaction": json.loads(row["compaction"]) if row["compaction"] else None}


@router.get("/{chat_id}")
def get_chat(chat_id: str):
    with db() as con:
        return _with_messages(con, chat_id)


class SaveMessages(BaseModel):
    messages: list[Message]
    model: str | None = None
    settings: dict | None = None
    # The summary standing in for messages[:upto] when the thread outgrew the context: {upto, text}.
    compaction: dict | None = None


@router.put("/{chat_id}/messages")
def save_messages(chat_id: str, req: SaveMessages):
    """Saves the whole thread; an untitled chat takes its title from the first message."""
    messages = [m.model_dump() for m in req.messages]
    with db() as con:
        row = _get(con, chat_id)
        write_messages(con, chat_id, messages)
        con.execute(
            "update chats set updated = ?, title = ?, model = coalesce(?, model), settings = coalesce(?, settings), compaction = ? where id = ?",
            (time.time(), row["title"] or title_from(messages), req.model, json.dumps(req.settings) if req.settings else None, json.dumps(req.compaction) if req.compaction else None, chat_id),
        )
        return _summary(_get(con, chat_id))


class ChatUpdate(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    folder: str | None = None  # "" takes it out of its folder


@router.patch("/{chat_id}")
def update_chat(chat_id: str, req: ChatUpdate):
    with db() as con:
        _get(con, chat_id)
        if req.title is not None:
            con.execute("update chats set title = ? where id = ?", (req.title.strip()[:200], chat_id))
        if req.pinned is not None:
            con.execute("update chats set pinned = ? where id = ?", (int(req.pinned), chat_id))
        if req.folder is not None:
            if req.folder:
                _check_folder(con, req.folder)
            con.execute("update chats set folder = ? where id = ?", (req.folder or None, chat_id))
        return _summary(_get(con, chat_id))


@router.delete("/{chat_id}")
def delete_chat(chat_id: str):
    with db() as con:
        con.execute("delete from messages_fts where chat_id = ?", (chat_id,))
        con.execute("delete from chats where id = ?", (chat_id,))
    return {"ok": True}


# --- Export ---------------------------------------------------------------------------------------


def to_markdown(chat: dict) -> str:
    who = {"user": "You", "assistant": "Assistant", "system": "System"}
    lines = [f"# {chat['title']}", ""]
    meta = [f"Model: {chat['model']}" if chat.get("model") else None, f"Imported from {chat['source']}" if chat.get("source") else None, time.strftime("%Y-%m-%d %H:%M", time.localtime(chat["created"]))]
    lines += [" · ".join(m for m in meta if m), ""]
    for m in chat["messages"]:
        name = who[m["role"]] + (f" ({m['model']})" if m["role"] == "assistant" and m.get("model") and m["model"] != chat.get("model") else "")
        lines += [f"## {name}", ""]
        if m.get("reasoning"):
            lines += ["<details><summary>Thinking</summary>", "", m["reasoning"].strip(), "", "</details>", ""]
        if m.get("attachments"):
            lines += ["Attached: " + ", ".join(a["name"] for a in m["attachments"]), ""]
        lines += [m["content"].strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def to_json(chat: dict) -> str:
    keep = ("id", "title", "model", "created", "updated", "source", "project", "settings")
    return json.dumps({**{k: chat.get(k) for k in keep}, "messages": [{k: m[k] for k in ("role", "content", "model", "created", "reasoning", "attachments") if m.get(k) is not None or k not in ("reasoning", "attachments")} for m in chat["messages"]]}, indent=2, ensure_ascii=False)


def _export_text(chat_id: str, format: str) -> str:
    if format not in ("md", "json"):
        raise HTTPException(400, "Export as md or json.")
    chat = get_chat(chat_id)
    return to_markdown(chat) if format == "md" else to_json(chat)


@router.get("/{chat_id}/export")
def export_chat(chat_id: str, format: str = "md"):
    """The export's text and a file name for it, for the browser to save."""
    title = get_chat(chat_id)["title"]
    return {"text": _export_text(chat_id, format), "filename": f"{re.sub(r'[^\w .-]+', '', title).strip()[:80] or 'chat'}.{format}"}


class ExportToFile(BaseModel):
    format: str = "md"
    path: str


@router.post("/{chat_id}/export")
def export_to_file(chat_id: str, req: ExportToFile):
    """Writes the export where the person chose in the save dialog (the webview can't write files)."""
    path = Path(req.path).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, "Choose where to save with its full path.")
    path.write_text(_export_text(chat_id, req.format), encoding="utf-8")
    return {"path": str(path)}


# --- Folders --------------------------------------------------------------------------------------


def _folders(con: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in con.execute("select name from folders order by created")]


def _check_folder(con: sqlite3.Connection, name: str) -> None:
    if not con.execute("select 1 from folders where name = ?", (name,)).fetchone():
        raise HTTPException(404, f"There's no folder called {name}.")


class FolderRequest(BaseModel):
    name: str
    new_name: str | None = None


def _clean(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()[:80]
    if not name:
        raise HTTPException(400, "Give the folder a name.")
    return name


@router.post("/folders/add")
def add_folder(req: FolderRequest):
    name = _clean(req.name)
    with db() as con:
        if con.execute("select 1 from folders where name = ?", (name,)).fetchone():
            raise HTTPException(409, f"There's already a folder called {name}.")
        con.execute("insert into folders(name, created) values (?, ?)", (name, time.time()))
        return {"folders": _folders(con)}


@router.post("/folders/rename")
def rename_folder(req: FolderRequest):
    new = _clean(req.new_name or "")
    with db() as con:
        _check_folder(con, req.name)
        if new != req.name and con.execute("select 1 from folders where name = ?", (new,)).fetchone():
            raise HTTPException(409, f"There's already a folder called {new}.")
        con.execute("update folders set name = ? where name = ?", (new, req.name))
        con.execute("update chats set folder = ? where folder = ?", (new, req.name))
        return {"folders": _folders(con)}


@router.post("/folders/remove")
def remove_folder(req: FolderRequest):
    """Removes a folder; its chats stay, outside any folder."""
    with db() as con:
        con.execute("delete from folders where name = ?", (req.name,))
        con.execute("update chats set folder = null where folder = ?", (req.name,))
        return {"folders": _folders(con)}
