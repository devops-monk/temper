import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from temper_engine import browser, browser_worker, tools
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
# Stands in for browser_worker.py: answers each request with the page text naming what it was asked,
# and writes a screenshot where it's told to.
FAKE = """
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["action"] == "screenshot":
        open(req["out"], "wb").write(b"\\x89PNG fake")
    if req["action"] == "click" and req.get("ref") == 99:
        print(json.dumps({"error": "There's no [99] on the page now."}), flush=True)
        continue
    print(json.dumps({"done": {"page": json.dumps(req, sort_keys=True)}}), flush=True)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    script = tmp_path / "fake_browser.py"
    script.write_text(FAKE)
    monkeypatch.setattr(browser, "python", lambda: sys.executable)
    monkeypatch.setattr(browser.worker, "command", lambda: [sys.executable, "-u", str(script)])
    with TestClient(create_app(token="t")) as c:
        yield c
    browser.worker.stop()


def test_the_tool_is_offered_only_once_the_browser_is_installed(monkeypatch):
    assert "browse" not in [t.name for t in tools.available()]
    with pytest.raises(RuntimeError, match="Settings → Browser"):
        browser.browse({"action": "open", "url": "https://example.com"}, "c1")
    monkeypatch.setattr(browser, "python", lambda: sys.executable)
    assert "browse" in [t.name for t in tools.available()]
    assert tools.TOOLS["browse"].mode == "ask"


def test_calls_go_to_the_worker_with_the_chat(client):
    out = browser.browse({"action": "type", "ref": 2, "text": "hi", "submit": True, "junk": 1}, "c1")
    assert out == '{"action": "type", "chat": "c1", "ref": 2, "submit": true, "text": "hi"}'
    with pytest.raises(ValueError, match="no \\[99\\]"):
        browser.browse({"action": "click", "ref": 99}, "c1")
    with pytest.raises(ValueError, match="Give an action"):
        browser.browse({"action": "hack"}, "c1")
    with pytest.raises(ValueError, match="url"):
        browser.browse({"action": "open"}, "c1")


def test_a_screenshot_is_saved_in_the_chat_folder_and_served(client):
    out = browser.browse({"action": "screenshot"}, "c1")
    name = out.split("screenshots/")[1].split(" ")[0]
    assert (browser.screenshots("c1") / name).read_bytes() == b"\x89PNG fake"
    r = client.get(f"/api/browser/screenshots/{name}", headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/browser/screenshots/..%2F..%2Fsecret.png", headers=H).status_code == 404
    assert client.get("/api/browser/screenshots/0123456789abcdef.png", headers=H).status_code == 404


def test_deleting_a_chat_closes_its_browser_context(client, monkeypatch):
    sent = []
    browser.browse({"action": "open", "url": "https://example.com"}, "c1")
    real = browser.worker.request
    monkeypatch.setattr(browser.worker, "request", lambda req, *a: sent.append(req) or real(req, *a))
    chat = client.post("/api/chats", headers=H, json={"title": "x"}).json()
    assert client.delete(f"/api/chats/{chat['id']}", headers=H).status_code == 200
    assert {"chat": chat["id"], "action": "close"} in sent


def test_status_and_remove(client, tmp_path, monkeypatch):
    assert client.get("/api/browser", headers=H).json()["installed"] is True
    monkeypatch.setattr(browser, "python", lambda: None)
    assert client.delete("/api/browser", headers=H).json()["installed"] is False


# --- The worker's proxy, which every connection the browser makes goes through ----------------------


class _Page(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = f"hello from {self.path}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def proxy(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(browser_worker, "ALLOW", {"127.0.0.1"})  # this computer's own server, standing in for a public one
    p = browser_worker.Proxy()
    yield p.server_address[1], server.server_address[1]
    p.shutdown()
    server.shutdown()


def _through(proxy_port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=10) as s:
        s.sendall(request)
        out = b""
        while chunk := s.recv(65536):
            out += chunk
            if request.startswith(b"CONNECT"):
                break
        return out


def test_the_proxy_passes_public_addresses(proxy):
    proxy_port, port = proxy
    out = _through(proxy_port, f"GET http://127.0.0.1:{port}/page?q=1 HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nProxy-Connection: keep-alive\r\n\r\n".encode())
    assert out.startswith(b"HTTP/1.0 200") and out.endswith(b"hello from /page?q=1")


def test_the_proxy_refuses_this_computer_and_the_local_network(proxy):
    proxy_port, port = proxy
    for target in (f"http://localhost:{port}/secret", "http://10.0.0.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]:80/"):
        out = _through(proxy_port, f"GET {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        assert out.startswith(b"HTTP/1.1 403"), target
    out = _through(proxy_port, f"CONNECT localhost:{port} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n".encode())
    assert out.startswith(b"HTTP/1.1 403")


def test_allowed_checks_scheme_and_address():
    assert not browser_worker.allowed("file:///etc/passwd")
    assert not browser_worker.allowed("http://127.0.0.1:8000/")
    assert not browser_worker.allowed("http://192.168.1.10/")
    assert browser_worker.allowed("https://203.0.114.1/") is True
