"""Temper's engine as a background service: running detached from any terminal or window, and starting at login.

`temper daemon up` starts `temper serve` in its own session, with its output in daemon.log and its pid
in daemon.pid; `down` stops it. The login item runs the same command when you log in: a launchd agent on
macOS, a systemd user unit on Linux, a Task Scheduler task on Windows. The app finds a running background
engine through engine.json and uses it rather than starting a second one on the same files.

Secrets in the system keychain (the Hugging Face token) aren't passed to a background engine; set
HF_TOKEN in its environment, or sign in again in the app, for gated models.
"""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home

router = APIRouter(prefix="/api/daemon", tags=["runtime"])

LABEL = "app.temper.engine"
UNIT = "temper-engine.service"
TASK = "Temper engine"


def command(host: str = "127.0.0.1", port: int = 8765) -> list[str]:
    """How to run the engine headless with this copy of Temper's Python."""
    return [sys.executable, "-m", "temper_engine", "serve", "--host", host, "--port", str(port)]


def log_file() -> Path:
    return home() / "daemon.log"


def _pid_file() -> Path:
    return home() / "daemon.pid"


def pid() -> int | None:
    """The background engine's process, if one started by `up` is still running."""
    try:
        found = int(_pid_file().read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        proc = psutil.Process(found)
        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and "temper_engine" in " ".join(proc.cmdline()):
            return found
    except psutil.Error:
        pass
    return None


def start(host: str = "127.0.0.1", port: int = 8765) -> int:
    """Starts the engine detached, and returns its pid; the caller waits for it to answer."""
    env = {**os.environ, "TEMPER_HOME": str(home())}
    out = open(log_file(), "ab")
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT, "env": env, "cwd": str(home())}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True  # not ended by the terminal closing or Ctrl-C there
    proc = subprocess.Popen(command(host, port), **kwargs)
    out.close()
    _pid_file().write_text(str(proc.pid))
    return proc.pid


def stop(timeout: float = 15) -> bool:
    """Stops the background engine started by `up`; False when there wasn't one."""
    found = pid()
    if found is None:
        _pid_file().unlink(missing_ok=True)
        return False
    proc = psutil.Process(found)
    proc.terminate()
    try:
        proc.wait(timeout)
    except psutil.TimeoutExpired:
        proc.kill()
    _pid_file().unlink(missing_ok=True)
    running = home() / "engine.json"  # what it would have cleared on a clean exit
    try:
        if json.loads(running.read_text()).get("pid") == found:
            running.unlink()
    except (OSError, ValueError):
        pass
    return True


# --- Starting at login -----------------------------------------------------------------------------


def _plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _unit() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user" / UNIT


def _domain() -> str:
    """launchd's domain for this user's agents (getuid is POSIX-only; the tests stand in for macOS anywhere)."""
    return f"gui/{getattr(os, 'getuid', lambda: 0)()}"


def _run(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def login_item() -> bool:
    """Whether the engine is set to start at login."""
    if sys.platform == "darwin":
        return _plist().is_file()
    if sys.platform == "win32":
        return _run("schtasks", "/Query", "/TN", TASK).returncode == 0
    return _unit().is_file()


def install() -> str:
    """Sets the engine to start at login (and starts it now, as the service managers do). Returns where."""
    cmd = command()
    if sys.platform == "darwin":
        path = _plist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plistlib.dumps({
            "Label": LABEL,
            "ProgramArguments": cmd,
            "EnvironmentVariables": {"TEMPER_HOME": str(home())},
            "RunAtLoad": True,
            "StandardOutPath": str(log_file()),
            "StandardErrorPath": str(log_file()),
        }))
        _run("launchctl", "bootout", f"{_domain()}/{LABEL}")  # an older copy, if loaded
        r = _run("launchctl", "bootstrap", _domain(), str(path))
        if r.returncode != 0:
            raise RuntimeError(f"launchctl couldn't load it: {r.stderr.strip() or r.stdout.strip()}")
        return str(path)
    if sys.platform == "win32":
        line = subprocess.list2cmdline(cmd)
        r = _run("schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", TASK, "/TR", line)
        if r.returncode != 0:
            raise RuntimeError(f"Task Scheduler refused it: {r.stderr.strip() or r.stdout.strip()}")
        return f"Task Scheduler: {TASK}"
    path = _unit()
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    path.write_text(
        "[Unit]\nDescription=Temper engine\nAfter=network-online.target\n\n"
        f"[Service]\nExecStart={quoted}\nEnvironment=\"TEMPER_HOME={home()}\"\nRestart=on-abnormal\n\n"
        "[Install]\nWantedBy=default.target\n"
    )
    _run("systemctl", "--user", "daemon-reload")
    r = _run("systemctl", "--user", "enable", "--now", UNIT)
    if r.returncode != 0:
        raise RuntimeError(f"systemctl couldn't enable it: {r.stderr.strip() or r.stdout.strip()}")
    return str(path)


def uninstall() -> bool:
    """Stops the engine starting at login (and stops the one it started). False when it wasn't set up."""
    if not login_item():
        return False
    if sys.platform == "darwin":
        _run("launchctl", "bootout", f"{_domain()}/{LABEL}")
        _plist().unlink(missing_ok=True)
    elif sys.platform == "win32":
        _run("schtasks", "/Delete", "/F", "/TN", TASK)
    else:
        _run("systemctl", "--user", "disable", "--now", UNIT)
        _unit().unlink(missing_ok=True)
        _run("systemctl", "--user", "daemon-reload")
    return True


def wait_until_answering(url_of: callable, timeout: float = 90) -> str | None:
    """Polls `url_of()` (the running engine's address, or None) until it answers; None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if url := url_of():
            return url
        time.sleep(0.5)
    return None


class Login(BaseModel):
    enabled: bool


@router.get("")
def daemon_status():
    return {"login": login_item(), "pid": pid()}


@router.put("/login")
def set_login(req: Login):
    """Settings' "Start the engine at login". Turning it on also runs it now, under the service manager."""
    try:
        if req.enabled:
            install()
        else:
            uninstall()
    except (RuntimeError, OSError) as e:
        raise HTTPException(500, str(e)) from e
    return daemon_status()
