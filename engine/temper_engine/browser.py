"""A web browser the model can drive: the browse tool, for pages that need clicking and typing.

read_web_page (web.py) reads a page's HTML as it arrives; many pages only fill in once their scripts run,
and forms and searches need typing and clicking. The browse tool opens pages in a headless Chromium
(browser_worker.py), lists the links, buttons and fields on each with numbers, and acts on them by number.
Screenshots are saved in the chat's working folder (where the code tools can open them) and shown in the reply.

Playwright and Chromium's headless shell, about 350 MB on disk, install on demand into their own environment
(Settings → Browser). One browser runs for every chat, each chat in its own context (its own cookies and
history), and stops after ten minutes unused. Like read_web_page, it refuses addresses on this computer or
the local network, for the page's own requests too, and marks what pages say as theirs, not the person's.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from . import engines, home, sandbox
from .jobs import Job, registry
from .speech import Worker

router = APIRouter(prefix="/api/browser", tags=["chat"])

VERSION = "1.64.0"
WORKER = Path(__file__).with_name("browser_worker.py")
ACTIONS = ("open", "click", "type", "select", "scroll", "back", "screenshot")


def folder() -> Path:
    return home() / "engines" / "browser"


def browsers() -> Path:
    return folder() / "browsers"


def python() -> str | None:
    p = engines._venv_python(folder())
    return str(p) if p.is_file() and (folder() / ".installed").is_file() and (folder() / ".chromium").is_file() else None


def installed() -> bool:
    return python() is not None


def _settings() -> str:
    # Tests point the browser at servers on this computer, which it otherwise refuses.
    allow = [h for h in os.environ.get("TEMPER_BROWSER_ALLOW", "").split(",") if h]
    return json.dumps({"allow": allow})


worker = Worker("browser", lambda: [python(), "-I", "-u", str(WORKER), str(browsers()), _settings()])


def _install(job: Job) -> None:
    if not engines.install_venv(job, folder(), [f"playwright=={VERSION}"], "the browser"):
        return
    (folder() / ".chromium").unlink(missing_ok=True)
    cmd = [str(engines._venv_python(folder())), "-m", "playwright", "install", "--only-shell", "chromium"]
    job.log("$ playwright install --only-shell chromium")
    env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(browsers())}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env)
    job.process = proc
    assert proc.stdout
    for line in proc.stdout:
        if m := re.search(r"(\d+)% of", line):  # Playwright's download bar
            job.progress = int(m.group(1)) / 100
            job.emit("progress", value=job.progress)
        elif line.strip():
            job.log(line.rstrip())
    if proc.wait() != 0:
        if job.cancel_requested:
            job.finish("cancelled")
            return
        raise RuntimeError("Downloading Chromium failed; the log says why.")
    (folder() / ".chromium").write_text(VERSION)
    job.finish("done")


def screenshots(chat: str | None) -> Path:
    out = sandbox.workdir(chat) / "screenshots"
    out.mkdir(exist_ok=True)
    return out


def browse(args: dict, chat: str | None) -> str:
    """Runs one browse tool call in the chat's browser context; returns the page as text."""
    if not installed():
        raise RuntimeError("The browser isn't installed. The person can add it in Settings → Browser.")
    action = args.get("action")
    if action not in ACTIONS:
        raise ValueError(f"Give an action: one of {', '.join(ACTIONS)}.")
    if action == "open" and not str(args.get("url") or "").strip():
        raise ValueError("Give the page's `url` to open.")
    request = {k: args[k] for k in ("url", "ref", "text", "submit", "direction") if k in args}
    request.update(chat=chat or "scratch", action=action)
    shot = None
    if action == "screenshot":
        shot = screenshots(chat) / f"{secrets.token_hex(8)}.png"
        request["out"] = str(shot)
    done = worker.request(request)
    if shot:
        return f"Took a screenshot of the page; the person can see it. It's saved as screenshots/{shot.name} in this chat's working folder.\n\n{done['page']}"
    return done["page"]


def forget(chat: str) -> None:
    """Closes a deleted chat's browser context, if the browser is running."""
    if worker.running():
        try:
            worker.request({"chat": chat, "action": "close"})
        except (RuntimeError, ValueError):
            pass


@router.get("")
def status():
    return {"installed": installed(), "version": VERSION, "running": worker.running()}


@router.post("/install")
def install():
    job = registry.create("browser-install", "Install the browser (Playwright and Chromium)")
    registry.run_thread(job, _install)
    return job.summary()


@router.delete("")
def remove():
    worker.stop()
    shutil.rmtree(folder(), ignore_errors=True)
    return status()


@router.get("/screenshots/{name}")
def screenshot(name: str):
    """A screenshot the browse tool took, by its file name (random, so it names one chat's file)."""
    if not re.fullmatch(r"[0-9a-f]{16}\.png", name):
        raise HTTPException(404, "There's no such screenshot.")
    found = next(iter((sandbox.root() / "chats").glob(f"*/screenshots/{name}")), None)
    if not found:
        raise HTTPException(404, "There's no such screenshot.")
    return FileResponse(found, media_type="image/png")
