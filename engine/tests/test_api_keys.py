import json

from fastapi.testclient import TestClient

from temper_engine import api_keys, home
from temper_engine.app import create_app


def test_a_key_opens_the_model_api_and_nothing_else_until_revoked():
    with TestClient(create_app(token="t")) as c:
        h = {"Authorization": "Bearer t"}
        made = c.post("/api/keys", headers=h, json={"name": "Laptop"}).json()
        key = made["key"]
        assert key.startswith("tk-") and made["hint"].startswith(key[:6]) and made["name"] == "Laptop"
        assert key not in (home() / "api_keys.json").read_text()  # kept hashed
        assert "key" not in c.get("/api/keys", headers=h).json()[0]

        assert c.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 200
        assert c.get("/v1/models", headers={"x-api-key": key}).status_code == 200
        assert c.get("/api/keys", headers={"Authorization": f"Bearer {key}"}).status_code == 401  # not the app's API
        assert c.post("/mcp", headers={"Authorization": f"Bearer {key}"}, json={}).status_code == 401

        listed = c.get("/api/keys", headers=h).json()[0]
        assert listed["requests"] == 2 and listed["last_used"]
        log = c.get("/api/keys/log", headers=h).json()
        assert [e["key"] for e in log[:2]] == ["Laptop", "Laptop"] and log[0]["status"] == 200 and log[0]["path"] == "/v1/models"

        assert c.delete(f"/api/keys/{made['id']}", headers=h).json() == {"ok": True}
        assert c.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 401
        assert c.delete(f"/api/keys/{made['id']}", headers=h).status_code == 404


def test_token_counts_are_added_to_the_request_and_its_key():
    record = api_keys.create_key(api_keys.NewKey(name="ci"))
    key = api_keys.find(record["key"])
    entry = api_keys.start("/v1/chat/completions", "POST", key)
    chunk = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4}}\n\n'
    api_keys.usage_from_stream(entry, chunk, "qwen")
    api_keys.finish(entry, 200)
    assert entry["model"] == "qwen" and (entry["input_tokens"], entry["output_tokens"]) == (11, 4) and entry["status"] == 200
    api_keys.flush()
    saved = json.loads((home() / "api_keys.json").read_text())["keys"][0]
    assert (saved["requests"], saved["input_tokens"], saved["output_tokens"]) == (1, 11, 4)
