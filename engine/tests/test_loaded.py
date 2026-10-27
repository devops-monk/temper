import time

import pytest
from fastapi.testclient import TestClient

from temper_engine import compare, memory
from temper_engine.app import create_app
from temper_engine.runtime import RuntimeManager, manager

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}
GB = memory.GB


def call(method, path, status=200, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == status, r.text
    return r.json()


@pytest.fixture
def models(tmp_path, monkeypatch):
    """GGUFs needing 2, 3 and 9 GB with 8 GB free; servers pretend to load and are ready at once."""
    sizes = {"a": 2, "b": 3, "huge": 9}
    paths = {}
    for name in sizes:
        paths[name] = tmp_path / f"{name}.gguf"
        paths[name].write_bytes(b"GGUF")
    monkeypatch.setattr(memory, "local_chat_estimate", lambda path, context: {"total": sizes[path.stem] * GB})
    monkeypatch.setattr(memory, "memory_budget", lambda: {"kind": "ram", "total": 16 * GB, "free": 8 * GB})

    def fake_load(self, req):
        self.model, self.path, self.state, self.port, self.used = req.name or req.path.rsplit("/", 1)[-1], req.path, "ready", 1, time.time()

    def fake_unload(self):
        self.model, self.path, self.state = None, None, "idle"

    monkeypatch.setattr(RuntimeManager, "load", fake_load)
    monkeypatch.setattr(RuntimeManager, "unload", fake_unload)
    monkeypatch.setattr(compare.Pool, "_start_reaper", lambda self: None)
    yield {k: str(v) for k, v in paths.items()}
    for d in (compare.pool.servers, compare.pool.used, compare.pool.kept):
        d.clear()
    fake_unload(manager)


def test_models_load_alongside_and_answer_by_name(models):
    call("POST", "/api/runtime/load", json={"path": models["a"]})
    status = call("POST", "/api/runtime/load", json={"path": models["b"], "alongside": True})
    assert status["path"] == models["a"]
    assert [m["path"] for m in call("GET", "/api/runtime")["alongside"]] == [models["b"]]
    assert {m["id"] for m in call("GET", "/v1/models")["data"]} == {"a.gguf", "b.gguf"}
    # Compare mode reuses an alongside model and doesn't unload it.
    call("POST", "/api/compare/unload")
    assert models["b"] in compare.pool.kept
    out = call("POST", "/api/runtime/load", 409, json={"path": models["huge"], "alongside": True})
    assert "free" in out["detail"]


def test_switching_swaps_chat_and_an_alongside_model(models):
    call("POST", "/api/runtime/load", json={"path": models["a"]})
    call("POST", "/api/runtime/load", json={"path": models["b"], "alongside": True})
    listing = call("POST", "/api/runtime/loaded/switch", json={"path": models["b"]})
    assert manager.path == models["b"] and [m["path"] for m in listing["models"]] == [models["b"], models["a"]]
    assert listing["models"][0]["chat"] and not listing["models"][1]["chat"]
    # Loading a model that's already alongside switches to it rather than reloading.
    call("POST", "/api/runtime/load", json={"path": models["a"]})
    assert manager.path == models["a"] and compare.pool.servers[models["b"]].path == models["b"]
    # Unloading chat's model hands chat to the one still loaded.
    call("POST", "/api/runtime/loaded/unload", json={"path": models["a"]})
    assert manager.path == models["b"] and not compare.pool.servers


def test_idle_models_unload(models):
    call("POST", "/api/runtime/load", json={"path": models["a"]})
    call("POST", "/api/runtime/load", json={"path": models["b"], "alongside": True})
    assert call("PUT", "/api/runtime/loaded/idle", json={"minutes": 5})["idle_minutes"] == 5
    compare.pool.reap(now=time.time() + 60)
    assert manager.state == "ready" and models["b"] in compare.pool.servers
    compare.pool.reap(now=time.time() + 6 * 60)
    assert manager.state == "idle" and not compare.pool.servers
    call("PUT", "/api/runtime/loaded/idle", json={"minutes": 0})
    call("POST", "/api/runtime/load", json={"path": models["a"]})
    compare.pool.reap(now=time.time() + 24 * 3600)
    assert manager.state == "ready"
