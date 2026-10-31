"""LAN access: the model API served to other devices on the home network.

Off by default. When it's on, a second listener on every network interface serves the same app, but
only its model API (/v1) and health check: managing Temper stays on this computer. Requests need a key,
as on this computer, so Connect → API keys is where a phone or another laptop gets one.
"""

from __future__ import annotations

import socket
import threading

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import prefs

router = APIRouter(prefix="/api/lan", tags=["runtime"])

DEFAULT_PORT = 8770
ALLOWED = ("/v1/", "/api/health")


def lan_only(app):
    """`app` with everything but the model API answered 404, for the LAN listener."""

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and not scope["path"].startswith(ALLOWED):
            await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"detail": "Only the model API (/v1) is served on the network."}'})
            return
        if scope["type"] == "lifespan":  # the main listener runs the app's startup and shutdown
            return
        await app(scope, receive, send)

    return asgi


def addresses() -> list[str]:
    """This computer's IPv4 addresses on its networks, the one that routes outward first."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent; it only picks the outgoing interface
            found.append(s.getsockname()[0])
    except OSError:
        pass
    for name, addrs in psutil.net_if_addrs().items():
        stats = psutil.net_if_stats().get(name)
        if stats and not stats.isup:
            continue
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")) and a.address not in found:
                found.append(a.address)
    return found


def answering(port: int) -> bool:
    """Whether something already listens on this computer's port. Binding alone doesn't tell: another app
    on the IPv6 wildcard leaves 127.0.0.1 free to bind, yet "localhost" would still reach that app."""
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((address, port)) == 0:
                    return True
        except OSError:
            pass
    return False


class Background:
    """A uvicorn server for an extra address, in a thread of its own, that can be started and stopped."""

    def __init__(self, name: str):
        self.name = name
        self.server = None
        self.thread: threading.Thread | None = None
        self.port: int | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return bool(self.server and self.server.started and self.thread and self.thread.is_alive())

    def start(self, asgi, host: str, port: int) -> None:
        import uvicorn

        self.stop()
        self.error = None
        if answering(port):
            self.error = f"Port {port} is in use by another app. Choose another port."
            return
        self.server = server = uvicorn.Server(uvicorn.Config(asgi, host=host, port=port, log_level="warning", lifespan="off"))
        self.port = port

        def run():
            try:
                server.run()
            except SystemExit:  # uvicorn exits this way when the port is taken
                pass
            if not server.started and not server.should_exit:
                self.error = f"Port {port} is in use by another app. Choose another port."

        self.thread = threading.Thread(target=run, name=self.name, daemon=True)
        self.thread.start()
        for _ in range(50):  # wait up to 5 s for it to start or fail
            if server.started or not self.thread.is_alive():
                break
            threading.Event().wait(0.1)

    def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)
        self.server = self.thread = None


class Listener(Background):
    def __init__(self):
        super().__init__("lan-listener")
        self.app = None

    def apply(self) -> None:
        """Starts or stops the listener to match the saved setting."""
        cfg = prefs.load().get("lan") or {}
        if cfg.get("enabled") and self.app is not None:
            self.start(lan_only(self.app), "0.0.0.0", int(cfg.get("port") or DEFAULT_PORT))
        else:
            self.stop()


listener = Listener()


def status() -> dict:
    cfg = prefs.load().get("lan") or {}
    port = int(cfg.get("port") or DEFAULT_PORT)
    ips = addresses()
    return {
        "enabled": bool(cfg.get("enabled")),
        "port": port,
        "running": listener.running,
        "error": listener.error,
        "addresses": ips,
        "url": f"http://{ips[0]}:{port}" if ips else None,
    }


class LanSettings(BaseModel):
    enabled: bool
    port: int = Field(DEFAULT_PORT, ge=1024, le=65535)


@router.get("")
def get_lan():
    return status()


@router.post("")
def set_lan(req: LanSettings):
    import os

    if req.enabled and str(req.port) == os.environ.get("TEMPER_PORT"):
        raise HTTPException(400, "That's the port Temper already uses on this computer. Choose another.")
    prefs.update("lan", {"enabled": req.enabled, "port": req.port})
    listener.apply()
    out = status()
    if req.enabled and not out["running"]:
        prefs.update("lan", {"enabled": False, "port": req.port})
        raise HTTPException(409, out["error"] or "The network listener didn't start.")
    return out
