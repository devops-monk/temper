"""API keys for the model API (/v1): named, revocable keys to hand to other apps instead of Temper's own
token, with usage per key and a log of recent requests (the API monitor).

Keys are kept hashed in api_keys.json in Temper's home, so the full key is shown only when it's made.
A key opens /v1 and nothing else: managing Temper (/api) and its MCP server (/mcp) still take the app's
own token. Usage is counted in memory and saved now and then, so a busy API doesn't write on every request.

The log can also hold each request's prompt and reply, but only while someone is reading it with
content=1 (`temper log --content`, or the monitor's switch): it is never written to disk, and it is
dropped from every entry once nobody has asked for it for CONTENT_WINDOW seconds.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import home

router = APIRouter(prefix="/api/keys", tags=["runtime"])

PREFIX = "tk-"
LOG_SIZE = 200
SAVE_EVERY = 30  # seconds between saving usage counts

_lock = threading.Lock()
_keys: list[dict] | None = None
_dirty_since: float | None = None
log: deque[dict] = deque(maxlen=LOG_SIZE)
CONTENT_WINDOW = 30  # seconds a content reader keeps prompts and replies being recorded
CONTENT_CHARS = 8000  # of each prompt and reply
_seq = 0  # bumped on every change to an entry, so a stream sends only what changed
_content_until = 0.0


def _file() -> Path:
    return home() / "api_keys.json"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _all() -> list[dict]:
    global _keys
    if _keys is None:
        try:
            _keys = list(json.loads(_file().read_text()).get("keys") or [])
        except (OSError, ValueError):
            _keys = []
    return _keys


def _save() -> None:
    global _dirty_since
    path = _file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"keys": _all()}, indent=2))
    tmp.chmod(0o600)
    tmp.replace(path)
    _dirty_since = None


def flush() -> None:
    with _lock:
        if _dirty_since is not None:
            _save()


def _public(k: dict) -> dict:
    return {f: k.get(f) for f in ("id", "name", "hint", "created", "last_used", "requests", "input_tokens", "output_tokens")}


def find(key: str) -> dict | None:
    """The key record for a presented key, if it's one of ours."""
    if not key.startswith(PREFIX):
        return None
    digest = _hash(key)
    with _lock:
        return next((k for k in _all() if hmac.compare_digest(k["hash"], digest)), None)


def _changed(entry: dict) -> None:
    global _seq
    _seq += 1
    entry["seq"] = _seq


def capturing() -> bool:
    return time.time() < _content_until


def watch_content() -> None:
    """A reader wants prompts and replies: record them for the next CONTENT_WINDOW seconds."""
    global _content_until
    _content_until = time.time() + CONTENT_WINDOW


def start(path: str, method: str, key: dict | None) -> dict:
    """Logs a request as it arrives; the returned entry is filled in as it's answered."""
    entry = {"id": uuid.uuid4().hex[:12], "time": time.time(), "method": method, "path": path, "key": key["name"] if key else "Temper", "key_id": key["id"] if key else None,
             "status": None, "ms": None, "model": None, "input_tokens": None, "output_tokens": None, "first_ms": None, "tokens_per_second": None, "done": False}
    with _lock:
        if not capturing():  # nobody's watching: what was recorded for them goes
            for e in log:
                e.pop("prompt", None)
                e.pop("reply", None)
        _changed(entry)
        log.append(entry)
        if key:
            key["requests"] = key.get("requests", 0) + 1
            key["last_used"] = entry["time"]
            _touch()
    return entry


def finish(entry: dict, status: int) -> None:
    """The response has started. A streamed reply isn't done until its stream ends (done())."""
    entry["status"] = status
    if not entry.get("streaming"):
        done(entry)
    else:
        _changed(entry)


def done(entry: dict | None) -> None:
    if entry is None:
        return
    entry["ms"] = round((time.time() - entry["time"]) * 1000)
    if not entry.get("streaming"):
        entry["first_ms"] = None  # the whole reply came at once: there was no first token to time
    out, first = entry.get("output_tokens"), entry.get("first_ms")
    if out and first is not None and entry["ms"] > first:
        entry["tokens_per_second"] = round(out / ((entry["ms"] - first) / 1000), 1)
    entry["done"] = True
    _changed(entry)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # parts: keep the text, name the rest
        return "\n".join(p.get("text") or f"[{p.get('type', 'part')}]" for p in content if isinstance(p, dict))
    return ""


def record_prompt(entry: dict | None, messages: list | None) -> None:
    """The request's last message, while someone's reading the content."""
    if entry is None or not capturing() or not messages:
        return
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    entry["prompt"] = _text(last.get("content"))[:CONTENT_CHARS]
    _changed(entry)


def record_reply(entry: dict | None, text: str) -> None:
    if entry is None or not text:
        return
    if entry.get("first_ms") is None:
        entry["first_ms"] = round((time.time() - entry["time"]) * 1000)
    if capturing():
        entry["reply"] = (entry.get("reply", "") + text)[:CONTENT_CHARS]
    _changed(entry)


def observe(entry: dict | None, line: str) -> None:
    """Notes the first token and, while watched, the reply's text, from an OpenAI stream line passing through."""
    if entry is None or not line.startswith("data:") or '"choices"' not in line:
        return
    try:
        choices = json.loads(line[5:]).get("choices") or []
    except ValueError:
        return
    delta = (choices[0].get("delta") or {}) if choices else {}
    if delta.get("reasoning_content") or delta.get("reasoning"):
        if entry.get("first_ms") is None:
            entry["first_ms"] = round((time.time() - entry["time"]) * 1000)
    record_reply(entry, delta.get("content") or "")


def tokens(entry: dict | None, model: str | None, input_tokens: int | None, output_tokens: int | None) -> None:
    """Records a reply's model and token counts, on the request's entry and its key's totals."""
    if entry is None:
        return
    entry["model"] = model
    entry["input_tokens"], entry["output_tokens"] = input_tokens, output_tokens
    _changed(entry)
    from . import usage

    usage.add(model, input_tokens, output_tokens)
    if entry.get("key_id") and (input_tokens or output_tokens):
        with _lock:
            for k in _all():
                if k["id"] == entry["key_id"]:
                    k["input_tokens"] = k.get("input_tokens", 0) + (input_tokens or 0)
                    k["output_tokens"] = k.get("output_tokens", 0) + (output_tokens or 0)
                    _touch()


def _touch() -> None:
    """Marks usage as changed (called holding the lock), saving at most every SAVE_EVERY seconds."""
    global _dirty_since
    now = time.time()
    if _dirty_since is None:
        _dirty_since = now
    elif now - _dirty_since >= SAVE_EVERY:
        _save()


def usage_from_stream(entry: dict | None, chunk: bytes, model: str | None) -> None:
    """Picks the token counts out of an OpenAI stream's usage chunk as it passes through."""
    if entry is None or b'"usage"' not in chunk:
        return
    for line in chunk.split(b"\n"):
        if line.startswith(b"data:") and b'"usage"' in line:
            try:
                usage = json.loads(line[5:]).get("usage") or {}
            except ValueError:
                continue
            if usage:
                tokens(entry, model, usage.get("prompt_tokens"), usage.get("completion_tokens"))


class NewKey(BaseModel):
    name: str = Field(min_length=1, max_length=60)


@router.get("")
def list_keys():
    with _lock:
        return [_public(k) for k in _all()]


@router.post("")
def create_key(req: NewKey):
    key = PREFIX + secrets.token_urlsafe(32)
    record = {"id": uuid.uuid4().hex[:12], "name": req.name.strip(), "hash": _hash(key), "hint": f"{key[:6]}…{key[-4:]}", "created": time.time(),
              "last_used": None, "requests": 0, "input_tokens": 0, "output_tokens": 0}
    with _lock:
        _all().append(record)
        _save()
    return {**_public(record), "key": key}  # the only time the whole key is shown


@router.delete("/{key_id}")
def revoke_key(key_id: str):
    with _lock:
        keys = _all()
        kept = [k for k in keys if k["id"] != key_id]
        if len(kept) == len(keys):
            raise HTTPException(404, "No such key")
        keys[:] = kept
        _save()
    return {"ok": True}


def _shown(entry: dict, content: bool) -> dict:
    return entry if content else {k: v for k, v in entry.items() if k not in ("prompt", "reply")}


@router.get("/log")
def request_log(limit: int = 100, content: bool = False):
    if content:
        watch_content()
    with _lock:
        return [_shown(e, content) for e in list(log)[-limit:][::-1]]


@router.get("/log/stream")
async def request_log_stream(content: bool = False, since: int = 0):
    """Server-sent events: each new or changed entry (oldest first), as it happens. `since` is the last seq seen."""

    async def events():
        last = since
        try:
            while True:
                if content:
                    watch_content()
                with _lock:
                    fresh = [_shown(e, content) for e in log if e.get("seq", 0) > last]
                for e in fresh:
                    last = max(last, e["seq"])
                    yield f"data: {json.dumps(e)}\n\n".encode()
                if not fresh:
                    yield b": idle\n\n"  # keeps proxies from closing it, and notices a reader that left
                await asyncio.sleep(0.3 if fresh else 1.0)
        except asyncio.CancelledError:
            return

    return StreamingResponse(events(), media_type="text/event-stream")
