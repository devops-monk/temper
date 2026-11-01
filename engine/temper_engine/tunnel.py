"""Remote access: an HTTPS link to this Temper from anywhere, through a Cloudflare quick tunnel.

Connect → Remote access starts `cloudflared tunnel --url` (downloaded on first use, checked against its SHA-256),
which needs no Cloudflare account and gives a random https://….trycloudflare.com address. The tunnel reaches a
listener of its own on this computer, so the engine knows which requests came through it. It only starts once there's
an admin account (accounts.py): the link opens the browser interface's sign-in, and members reach only chatting.
Requests through the link can't use tools, the code sandbox or Temper's MCP server unless that's turned on.

The link changes each time the tunnel starts, and the tunnel stops when Temper does.
"""

from __future__ import annotations

import contextvars
import io
import os
import platform
import re
import socket
import subprocess
import tarfile
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home, prefs

router = APIRouter(prefix="/api/tunnel", tags=["runtime"])

VERSION = "2026.10.0"
BUILDS = {  # (system, machine): (asset, its SHA-256), from github.com/cloudflare/cloudflared/releases
    ("Darwin", "arm64"): ("cloudflared-darwin-arm64.tgz", "a2f79ff7b9420aa537d74af239f376da170bbabeb529aec416002adac6a72e70"),
    ("Darwin", "x86_64"): ("cloudflared-darwin-amd64.tgz", "903845b81828c8cb3c5d13d816a2de71c06a3da5785469df8eb0e1b736d92f9f"),
    ("Linux", "x86_64"): ("cloudflared-linux-amd64", "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"),
    ("Linux", "aarch64"): ("cloudflared-linux-arm64", "e6422b9d4f72d3194bc5a38676f13667c06666523217b842a877d72a80b5ac08"),
    ("Windows", "AMD64"): ("cloudflared-windows-amd64.exe", "86aee4017b26625cee8484c113558f48effa4cd47f7aa05fcf425604e5d2b23c"),
}
LINK = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
# What a request through the link can't reach while tools are off: they run on this computer.
TOOL_PATHS = ("/mcp", "/api/sandbox", "/api/browser", "/api/tools", "/api/mcp")

# Whether the request being answered came through the tunnel.
remote: contextvars.ContextVar[bool] = contextvars.ContextVar("remote", default=False)


def _settings() -> dict:
    return prefs.load().get("tunnel") or {}


def tools_allowed() -> bool:
    """Whether this request may use tools: always on this computer, through the link only if that's turned on."""
    return not remote.get() or bool(_settings().get("tools"))


def through_tunnel(app):
    """`app` as the tunnel's listener serves it: requests marked as remote, tools kept out unless allowed."""

    async def asgi(scope, receive, send):
        if scope["type"] == "lifespan":  # the main listener runs the app's startup and shutdown
            return
        if scope["type"] in ("http", "websocket"):
            path = scope["path"]
            if path == "/" and scope["type"] == "http":
                await send({"type": "http.response.start", "status": 307, "headers": [(b"location", b"/ui/")]})
                await send({"type": "http.response.body", "body": b""})
                return
            if path.startswith(TOOL_PATHS) and not _settings().get("tools"):
                await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"detail": "Tools are turned off for the remote link."}'})
                return
        token = remote.set(True)
        try:
            await app(scope, receive, send)
        finally:
            remote.reset(token)

    return asgi


# --- cloudflared ------------------------------------------------------------------------------------


def _build() -> tuple[str, str] | None:
    system, machine = platform.system(), platform.machine()
    machine = {"amd64": "x86_64", "arm64": "aarch64" if system == "Linux" else "arm64"}.get(machine.lower(), machine)
    return BUILDS.get((system, "AMD64" if system == "Windows" and machine == "x86_64" else machine))


def _installed() -> Path:
    return home() / "cloudflared" / VERSION / ("cloudflared.exe" if os.name == "nt" else "cloudflared")


def binary() -> str | None:
    """Temper's own cloudflared, else one on PATH."""
    import shutil

    mine = _installed()
    return str(mine) if mine.is_file() else shutil.which("cloudflared")


def install(report=lambda message, fraction: None) -> str:
    from .llamacpp import _download

    build = _build()
    if not build:
        raise RuntimeError(f"There's no cloudflared build for {platform.system()} on {platform.machine()}. Install cloudflared yourself and start again.")
    asset, sha256 = build
    data = _download(f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/{asset}", sha256, report, "cloudflared")
    if asset.endswith(".tgz"):
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            member = next(m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "cloudflared")
            data = tar.extractfile(member).read()  # type: ignore[union-attr]
    path = _installed()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.chmod(0o755)
    tmp.replace(path)
    return str(path)


# --- The tunnel -------------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Tunnel:
    def __init__(self) -> None:
        from .lan import Background

        self.app = None  # set by app.py
        self.listener = Background("tunnel-listener")
        self.proc: subprocess.Popen | None = None
        self.state = "off"  # off | installing | starting | on | failed
        self.url: str | None = None
        self.error: str | None = None
        self.progress: float | None = None
        self.log: list[str] = []
        self.lock = threading.Lock()

    def status(self) -> dict:
        from . import accounts

        return {"state": self.state, "url": f"{self.url}/ui/" if self.url else None, "error": self.error, "progress": self.progress,
                "tools": bool(_settings().get("tools")), "installed": binary() is not None, "accounts": accounts.enabled()}

    def start(self) -> None:
        from . import accounts
        from .app import ui_folder

        if self.state in ("installing", "starting", "on"):
            return
        if not accounts.enabled():
            raise HTTPException(409, "Add an admin account first (Settings → Accounts): anyone with the link reaches the sign-in page.")
        if self.app is None or ui_folder() is None:
            raise HTTPException(409, "This Temper has no browser interface to share.")
        self.state, self.url, self.error, self.progress, self.log = "starting", None, None, None, []
        threading.Thread(target=self._run, name="tunnel", daemon=True).start()

    def _run(self) -> None:
        try:
            path = binary()
            if not path:
                self.state = "installing"
                path = install(lambda message, fraction: setattr(self, "progress", fraction))
                self.state, self.progress = "starting", None
            port = _free_port()
            self.listener.start(through_tunnel(self.app), "127.0.0.1", port)
            if not self.listener.running:
                raise RuntimeError(self.listener.error or "The tunnel's listener didn't start.")
            with self.lock:
                if self.state != "starting":  # stopped meanwhile
                    self.listener.stop()
                    return
                self.proc = subprocess.Popen([path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            proc = self.proc
            assert proc.stdout
            for line in proc.stdout:
                self.log = (self.log + [line.rstrip()])[-50:]
                if self.state == "starting" and (found := LINK.search(line)):
                    self.url, self.state = found.group(0), "on"
            if proc is self.proc and self.state != "off":
                self.state, self.url = "failed", None
                self.error = "The tunnel closed:\n" + "\n".join(self.log[-6:])
                self.listener.stop()
        except Exception as e:  # a download or start that failed shows on Connect
            self.state, self.error = "failed", str(e)
            self.listener.stop()

    def stop(self) -> None:
        with self.lock:
            proc, self.proc = self.proc, None
            self.state, self.url, self.progress = "off", None, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.listener.stop()


tunnel = Tunnel()


class TunnelRequest(BaseModel):
    enabled: bool


class ToolsRequest(BaseModel):
    allowed: bool


@router.get("")
def get_tunnel():
    return tunnel.status()


@router.post("")
def set_tunnel(req: TunnelRequest):
    if remote.get():
        raise HTTPException(403, "The remote link can't be turned on or off through itself.")
    tunnel.start() if req.enabled else tunnel.stop()
    return tunnel.status()


@router.put("/tools")
def set_tools(req: ToolsRequest):
    if remote.get():
        raise HTTPException(403, "Tools for the remote link can only be changed on this computer.")
    prefs.update("tunnel", {**_settings(), "tools": req.allowed})
    return tunnel.status()
