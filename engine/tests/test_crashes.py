import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from temper_engine import crashes, home
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}


def test_reports_leave_out_home_folders_keys_and_tokens():
    text = f"open {Path.home()}/models/x.gguf failed; Authorization: Bearer abcdefghijklmnop; hf_{'a' * 30}; token=supersecret1; key {'0f' * 16}"
    out = crashes.scrub(text)
    assert str(Path.home()) not in out and "~/models/x.gguf" in out
    assert "abcdefghijklmnop" not in out and "a" * 30 not in out and "supersecret1" not in out and "0f" * 16 not in out


def test_an_unexpected_engine_error_is_saved_and_explained(monkeypatch):
    app = create_app(token="t")

    @app.get("/api/boom")
    def boom():
        raise RuntimeError(f"disk at {Path.home()}/x is odd")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/api/boom", headers=H)
        assert r.status_code == 500 and "unexpected error: RuntimeError: disk at ~/x is odd" in r.json()["detail"]
        listed = c.get("/api/crashes", headers=H).json()
    assert listed["offer"] is False  # off unless turned on
    crash = listed["crashes"][0]
    assert crash["id"] == r.json()["crash_id"] and crash["where"] == "engine" and crash["context"] == "GET /api/boom" and "boom" in crash["trace"]
    query = parse_qs(urlparse(crash["report_url"]).query)
    assert query["title"][0].startswith("Crash: RuntimeError") and "GET /api/boom" in query["body"][0] and query["labels"] == ["crash"]


def test_interface_crashes_settings_and_clearing():
    with TestClient(create_app(token="t")) as c:
        made = c.post("/api/crashes", headers=H, json={"error": "TypeError: x is undefined", "trace": "at Chat (Chat.tsx:10)\n" * 800, "context": "chat"}).json()
        assert made["where"] == "interface" and len(made["report_url"]) <= crashes.URL_LIMIT  # a long trace is cut to fit
        assert c.post("/api/crashes/settings", headers=H, json={"offer": True}).json()["offer"] is True
        assert c.delete(f"/api/crashes/{made['id']}", headers=H).json() == {"ok": True}
        assert c.delete("/api/crashes/../../x", headers=H).status_code in (404, 405)
        for _ in range(3):
            c.post("/api/crashes", headers=H, json={"error": "e"})
        assert c.delete("/api/crashes", headers=H).json() == {"ok": True}
        assert c.get("/api/crashes", headers=H).json()["crashes"] == []
    assert not list((home() / "crashes").glob("*.json"))


def test_the_libraries_are_told_not_to_send_telemetry():
    import os

    import temper_engine  # noqa: F401

    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1" and os.environ["DO_NOT_TRACK"] == "1"
