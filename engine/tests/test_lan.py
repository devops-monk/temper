import socket

import httpx
from fastapi.testclient import TestClient

from temper_engine import lan, prefs
from temper_engine.app import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_network_listener_serves_only_the_model_api():
    port = _free_port()
    h = {"Authorization": "Bearer t"}
    with TestClient(create_app(token="t")) as c:
        assert c.get("/api/lan", headers=h).json()["enabled"] is False
        on = c.post("/api/lan", headers=h, json={"enabled": True, "port": port}).json()
        try:
            assert on["running"] and on["port"] == port and (on["url"] is None or on["url"].endswith(f":{port}"))
            base = f"http://127.0.0.1:{port}"
            assert httpx.get(f"{base}/v1/models", headers=h).status_code == 200
            assert httpx.get(f"{base}/v1/models").status_code == 401  # still needs a key
            assert httpx.get(f"{base}/api/lan", headers=h).status_code == 404  # managing Temper stays local
            assert httpx.get(f"{base}/api/health").json()["ok"] is True
        finally:
            off = c.post("/api/lan", headers=h, json={"enabled": False, "port": port}).json()
        assert not off["running"] and not lan.listener.running


def test_a_taken_port_is_reported_and_leaves_it_off():
    with socket.socket() as busy:
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with TestClient(create_app(token="t")) as c:
            r = c.post("/api/lan", headers={"Authorization": "Bearer t"}, json={"enabled": True, "port": port})
    assert r.status_code == 409 and "in use" in r.json()["detail"]
    assert prefs.load()["lan"]["enabled"] is False
