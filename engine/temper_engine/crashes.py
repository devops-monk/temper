"""Crash reports, kept on this computer and sent only by hand.

Temper sends no telemetry. When the engine hits an error it didn't expect, or the interface does, a
report is saved in crashes/ in Temper's home with home folders, keys and tokens taken out. Settings →
Privacy lists them; each can be opened as a GitHub issue in the browser, filled in for the user to read
and send themselves. "Offer to report crashes" (off unless turned on) also brings a new one up as it
happens.
"""

from __future__ import annotations

import json
import platform
import re
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import __version__, home, prefs

router = APIRouter(prefix="/api/crashes", tags=["runtime"])

ISSUES = "https://github.com/devops-monk/temper/issues/new"
KEEP = 50  # the newest reports kept
URL_LIMIT = 7000  # browsers and GitHub cut longer addresses

SECRETS = [
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_…"),
    (re.compile(r"\btk-[A-Za-z0-9_-]{20,}"), "tk-…"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "sk-…"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1…"),
    (re.compile(r"(?i)((?:api[_-]?key|token|password|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"), r"\1…"),
    (re.compile(r"\b[0-9a-f]{32}\b"), "…"),  # Temper's own key
]


def _folder() -> Path:
    path = home() / "crashes"
    path.mkdir(exist_ok=True)
    return path


def scrub(text: str) -> str:
    """The text without the user's home folder, keys and tokens."""
    for user_home in {str(Path.home()), str(Path.home().resolve())}:
        text = text.replace(user_home, "~")
    for pattern, replacement in SECRETS:
        text = pattern.sub(replacement, text)
    return text


def save(where: str, error: str, trace: str, context: str = "") -> dict:
    crash = {
        "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "time": time.time(),
        "where": where,
        "error": scrub(error)[:500],
        "trace": scrub(trace)[-6000:],
        "context": scrub(context)[:300],
        "version": __version__,
        "system": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    folder = _folder()
    (folder / f"{crash['id']}.json").write_text(json.dumps(crash, indent=2))
    for old in sorted(folder.glob("*.json"))[:-KEEP]:
        old.unlink(missing_ok=True)
    return crash


def from_exception(exc: BaseException, context: str) -> dict:
    return save("engine", f"{type(exc).__name__}: {exc}", "".join(traceback.format_exception(exc)), context)


def report_url(crash: dict) -> str:
    """A new GitHub issue, filled in from the report, for the user to read before sending."""
    title = f"Crash: {crash['error'][:80]}"
    head = f"**What happened**\n\n<!-- What were you doing when this happened? -->\n\n**Temper** {crash['version']} on {crash['system']}, in the {crash['where']}"
    if crash.get("context"):
        head += f" ({crash['context']})"
    trace = crash["trace"]
    while True:
        body = f"{head}\n\n```\n{crash['error']}\n\n{trace}\n```\n"
        url = f"{ISSUES}?{urlencode({'title': title, 'body': body, 'labels': 'crash'})}"
        if len(url) <= URL_LIMIT or len(trace) < 200:
            return url
        trace = "…" + trace[len(trace) // 4 :]  # keep the end, where the error is


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@router.get("")
def list_crashes():
    found = [c for p in sorted(_folder().glob("*.json"), reverse=True) if (c := _load(p))]
    return {"offer": bool(prefs.load().get("offer_crash_reports")), "crashes": [{**c, "report_url": report_url(c)} for c in found]}


class UiCrash(BaseModel):
    error: str = Field(max_length=2000)
    trace: str = Field("", max_length=20000)
    context: str = Field("", max_length=500)


@router.post("")
def report_ui_crash(req: UiCrash):
    crash = save("interface", req.error, req.trace, req.context)
    return {**crash, "report_url": report_url(crash)}


class CrashSettings(BaseModel):
    offer: bool


@router.post("/settings")
def crash_settings(req: CrashSettings):
    prefs.update("offer_crash_reports", req.offer or None)
    return list_crashes()


@router.delete("")
def clear_crashes():
    for p in _folder().glob("*.json"):
        p.unlink(missing_ok=True)
    return {"ok": True}


@router.delete("/{crash_id}")
def delete_crash(crash_id: str):
    path = _folder() / f"{crash_id}.json"
    if not re.fullmatch(r"[\w-]+", crash_id) or not path.is_file():
        raise HTTPException(404, "No such report")
    path.unlink()
    return {"ok": True}
