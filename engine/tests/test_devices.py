import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from temper_engine import devices, runtime
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
KEY = "tk-desk-key-123456"


@pytest.fixture
def desk(monkeypatch):
    """A desktop at http://desk:8765 running Temper with qwen loaded, and this engine's app."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "desk":
            raise httpx.ConnectError("no route", request=request)
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"detail": "bad key"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}, {"id": "llama", "loaded": False}, {"id": "phi@laptop", "device": "laptop"}]})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "from the desk"}}], "usage": {}})

    transport = httpx.MockTransport(handler)
    real_async = httpx.AsyncClient
    monkeypatch.setattr(devices.httpx, "get", lambda url, **kw: httpx.Client(transport=transport).get(url, **{k: v for k, v in kw.items() if k != "timeout"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw))
    devices._seen.clear()
    with TestClient(create_app(token="t")) as c:
        c.sent = sent
        yield c


def test_linking_checks_the_address_and_the_key(desk):
    assert desk.post("/api/devices", headers=H, json={"url": "elsewhere:8765", "key": KEY}).json()["detail"].endswith("Nothing answered at http://elsewhere:8765.")
    assert "refused the key" in desk.post("/api/devices", headers=H, json={"url": "desk", "key": "wrong-key-000"}).json()["detail"]
    d = desk.post("/api/devices", headers=H, json={"url": "http://desk:8765/v1/", "key": KEY, "name": "Desk"}).json()
    assert d["url"] == "http://desk:8765" and d["online"] and [m["id"] for m in d["models"]] == ["qwen", "llama"]  # not the ones it reaches itself
    assert KEY not in json.dumps(desk.get("/api/devices", headers=H).json())  # the key is never shown again
    assert desk.post("/api/devices", headers=H, json={"url": "desk", "key": KEY, "name": "desk"}).status_code == 409
    assert desk.post("/api/devices", headers=H, json={"url": "desk", "key": KEY, "name": "a@b"}).status_code == 400


def test_a_devices_models_are_listed_and_answered_there(desk):
    desk.post("/api/devices", headers=H, json={"url": "desk", "key": KEY, "name": "Desk"})
    listed = {m["id"]: m for m in desk.get("/v1/models", headers=H).json()["data"]}
    assert listed["qwen@Desk"]["device"] == "Desk" and listed["llama@Desk"]["loaded"] is False and "phi@laptop@Desk" not in listed
    r = desk.post("/v1/chat/completions", headers=H, json={"model": "qwen@desk", "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["choices"][0]["message"]["content"] == "from the desk"
    assert desk.sent[-1]["model"] == "qwen" and desk.sent[-1][devices.HOP] is True  # so the desk doesn't pass it back
    log = desk.get("/api/keys/log", headers=H).json()[0]
    assert log["model"] == "qwen on Desk"
    # By its bare name too, when this computer doesn't have it.
    desk.post("/v1/chat/completions", headers=H, json={"model": "llama", "messages": [{"role": "user", "content": "hi"}]})
    assert desk.sent[-1]["model"] == "llama"


def test_this_computer_answers_for_its_own_models_unless_a_device_is_preferred(desk):
    d = desk.post("/api/devices", headers=H, json={"url": "desk", "key": KEY, "name": "Desk"}).json()
    assert devices.target("qwen", lambda: True) is None
    assert devices.target("qwen", lambda: False).device["id"] == d["id"]
    assert devices.target("gpt-4o", lambda: False) is None  # nobody has it: chat's model, as before
    assert desk.put("/api/devices/preferred", headers=H, json={"device": d["id"]}).json()["preferred"] == d["id"]
    assert devices.target("qwen", lambda: True).model == "qwen on Desk"
    desk.delete(f"/api/devices/{d['id']}", headers=H)
    assert devices.preferred() is None and devices.target("qwen@Desk", lambda: False) is None


def test_a_passed_on_request_is_answered_here(desk):
    desk.post("/api/devices", headers=H, json={"url": "desk", "key": KEY, "name": "Desk"})
    with pytest.raises(HTTPException, match="No model is loaded"):
        runtime.route({"model": "qwen@Desk", devices.HOP: True, "messages": []})


def test_an_offline_device_says_so(desk, monkeypatch):
    d = desk.post("/api/devices", headers=H, json={"url": "desk", "key": KEY, "name": "Desk"}).json()
    devices._save([{**devices.linked()[0], "url": "http://gone:8765"}])
    listed = desk.get("/api/devices?fresh=true", headers=H).json()["devices"][0]
    assert listed["id"] == d["id"] and not listed["online"] and "Nothing answered" in listed["error"]
    assert devices.target("qwen", lambda: False) is None


def test_the_devices_command(desk, monkeypatch, capsys):
    import contextlib
    import os

    from temper_engine import cli

    desk.headers.update(H)
    monkeypatch.setattr(cli, "running", lambda: {"url": "http://testserver", "token": "t", "pid": os.getpid()})
    monkeypatch.setattr(cli, "_client", lambda engine: contextlib.nullcontext(desk))
    assert cli.main(["devices"]) == 0 and "No devices are linked" in capsys.readouterr().out
    assert cli.main(["devices", "add", "desk", "--key", KEY, "--name", "Desk"]) == 0
    assert "with 2 models. Ask for one as <model>@Desk" in capsys.readouterr().out
    assert cli.main(["devices", "prefer", "desk"]) == 0 and devices.preferred()
    assert cli.main(["devices"]) == 0
    out = capsys.readouterr().out
    assert "online, preferred" in out and "llama (would load)" in out
    assert cli.main(["devices", "prefer", "this"]) == 0 and devices.preferred() is None
    assert cli.main(["devices", "remove", "nope"]) == 1
    assert cli.main(["devices", "remove", "Desk"]) == 0 and devices.linked() == []
