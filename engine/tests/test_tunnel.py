import io
import sys
import tarfile
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import accounts, app as app_module, tunnel
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
# Stands in for cloudflared: prints a quick tunnel's address, as cloudflared does, then waits to be stopped.
FAKE = """import sys, time
print("INF Requesting new quick Tunnel on trycloudflare.com...", flush=True)
print("INF |  https://calm-river-test.trycloudflare.com  |", flush=True)
time.sleep(60)
"""


@pytest.fixture
def app(tmp_path, monkeypatch):
    script = tmp_path / "cloudflared.py"
    script.write_text(FAKE)
    launcher = tmp_path / ("cloudflared.cmd" if sys.platform == "win32" else "cloudflared")
    launcher.write_text(f'@"{sys.executable}" "{script}"\n' if sys.platform == "win32" else f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n')
    launcher.chmod(0o755)
    monkeypatch.setattr(tunnel, "binary", lambda: str(launcher))
    monkeypatch.setattr(app_module, "ui_folder", lambda: tmp_path)
    with TestClient(create_app(token="t")) as c:
        yield c
    tunnel.tunnel.stop()


def wait(c, state):
    for _ in range(100):
        s = c.get("/api/tunnel", headers=H).json()
        if s["state"] == state:
            return s
        time.sleep(0.05)
    raise AssertionError(s)


def test_it_needs_an_admin_account_first(app):
    r = app.post("/api/tunnel", headers=H, json={"enabled": True})
    assert r.status_code == 409 and "admin account" in r.json()["detail"]


def test_the_link_reaches_the_engine_with_tools_off(app):
    accounts.add("ann", "password1", admin=True)
    app.post("/api/tunnel", headers=H, json={"enabled": True})
    s = wait(app, "on")
    assert s["url"] == "https://calm-river-test.trycloudflare.com/ui/" and s["tools"] is False
    # What cloudflared forwards to: the tunnel's own listener.
    base = f"http://127.0.0.1:{tunnel.tunnel.listener.port}"
    assert httpx.get(f"{base}/api/health").status_code == 200
    assert httpx.get(f"{base}/", follow_redirects=False).headers["location"] == "/ui/"
    assert httpx.get(f"{base}/api/jobs").status_code == 401  # still needs a sign-in
    assert httpx.post(f"{base}/mcp", headers=H, json={}).status_code == 403
    r = httpx.post(f"{base}/api/chat/generations", headers=H, json={"body": {"messages": []}, "tools": ["python"]})
    assert r.status_code == 403 and "remote link" in r.json()["detail"]
    assert httpx.post(f"{base}/api/tunnel", headers=H, json={"enabled": False}).status_code == 403  # not from outside
    # Turned on, here only.
    app.put("/api/tunnel/tools", headers=H, json={"allowed": True})
    assert httpx.get(f"{base}/api/sandbox", headers=H).status_code == 200
    app.post("/api/tunnel", headers=H, json={"enabled": False})
    assert app.get("/api/tunnel", headers=H).json()["state"] == "off"
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{base}/api/health", timeout=1)


def test_a_tunnel_that_closes_says_why(app, tmp_path, monkeypatch):
    accounts.add("ann", "password1", admin=True)
    (tmp_path / "cloudflared.py").write_text('print("ERR failed to request quick Tunnel: 429 Too Many Requests", flush=True)\n')
    app.post("/api/tunnel", headers=H, json={"enabled": True})
    s = wait(app, "failed")
    assert "429 Too Many Requests" in s["error"]


def test_install_checks_the_download(monkeypatch, tmp_path):
    binary = b"#!/bin/sh\necho cloudflared\n"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        info = tarfile.TarInfo("cloudflared")
        info.size = len(binary)
        tar.addfile(info, io.BytesIO(binary))
    seen = {}

    def download(url, sha256, report, label):
        seen.update(url=url, sha256=sha256)
        return archive.getvalue()

    monkeypatch.setattr(tunnel, "_build", lambda: ("cloudflared-darwin-arm64.tgz", "abc"))
    monkeypatch.setattr("temper_engine.llamacpp._download", download)
    path = tunnel.install()
    assert open(path, "rb").read() == binary and seen["sha256"] == "abc"
    assert seen["url"].endswith(f"/{tunnel.VERSION}/cloudflared-darwin-arm64.tgz")
