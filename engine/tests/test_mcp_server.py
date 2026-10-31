import json

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import mcp, mcp_server, runtime
from temper_engine.app import create_app
from temper_engine.jobs import registry

AUTH = {"Authorization": "Bearer secret"}
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the .timeout. argument")


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPER_HOME", str(tmp_path / "home"))
    with TestClient(create_app(token="secret")) as c:
        yield c
    mcp.manager.close_all()
    mcp.manager.client = None


def rpc(c, method, params=None, rid=1):
    return c.post("/mcp", headers=AUTH, json={"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params else {})})


def test_the_endpoint_needs_the_token_and_speaks_json_rpc(app_client):
    assert app_client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    r = rpc(app_client, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert r.json()["result"]["protocolVersion"] == "2025-03-26" and r.headers["mcp-session-id"]
    assert app_client.post("/mcp", headers=AUTH, json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
    assert rpc(app_client, "nope").json()["error"]["code"] == -32601
    assert app_client.get("/mcp", headers=AUTH).status_code == 405
    batch = app_client.post("/mcp", headers=AUTH, json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "id": 2, "method": "ping"}]).json()
    assert [a["id"] for a in batch] == [1, 2]


def test_temper_s_own_client_can_use_temper_s_server(app_client, monkeypatch):
    """Round trip: the MCP client (mcp.py) connected to the engine's own /mcp."""
    mcp.manager.client = app_client
    mcp.save({"self": {"url": "http://testserver/mcp", "headers": AUTH}})
    conn = mcp.manager.connection("self")
    assert {t["name"] for t in conn.tools} == set(mcp_server.TOOLS)
    status = json.loads(conn.call("temper_status", {})[0])
    assert status["chat"]["state"] == "idle" and status["alongside"] == []
    job = registry.create("download", "Fetch a model")
    job.log("halfway")
    listed, failed = conn.call("list_jobs", {})
    assert not failed and json.loads(listed)[0]["id"] == job.id
    assert json.loads(conn.call("get_job", {"id": job.id})[0])["log"] == ["halfway"]
    # Asking with nothing loaded says so, as an error the agent can read.
    text, failed = conn.call("ask_model", {"prompt": "hi"})
    assert failed and "No model is loaded" in text
    text, failed = conn.call("load_model", {})
    assert failed and "path" in text
    mcp.manager.disconnect("self")


def test_ask_model_returns_the_answer_without_its_thinking(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout, headers=None):
        seen.update(json)
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>\nParis."}}]})

    monkeypatch.setattr(runtime.httpx, "post", fake_post)
    monkeypatch.setattr(runtime, "route", lambda body: (None, "http://model/v1/chat/completions", body))
    out = mcp_server._call("ask_model", {"prompt": "Capital of France?", "system": "Be brief.", "max_tokens": 50})
    assert out["content"][0]["text"] == "Paris." and seen["messages"][0] == {"role": "system", "content": "Be brief."} and seen["max_tokens"] == 50
