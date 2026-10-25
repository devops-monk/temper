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
    """Three GGUFs needing 2, 3 and 9 GB, 10 GB free, and servers that only pretend to start."""
    paths = {}
    for name, size in (("a", 2), ("b", 3), ("huge", 9)):
        paths[name] = tmp_path / f"{name}.gguf"
        paths[name].write_bytes(b"GGUF")
        paths[name + "_size"] = size
    monkeypatch.setattr(memory, "local_chat_estimate", lambda path, context: {"total": paths[path.stem + "_size"] * GB})
    monkeypatch.setattr(memory, "memory_budget", lambda: {"kind": "ram", "total": 16 * GB, "free": 10 * GB})

    def fake_load(self, req):
        self.model, self.path, self.state, self.port = req.name or req.path, req.path, "loading", 1

    monkeypatch.setattr(RuntimeManager, "load", fake_load)
    monkeypatch.setattr(RuntimeManager, "unload", lambda self: None)
    yield {k: str(v) for k, v in paths.items() if not k.endswith("_size")}
    compare.pool.servers.clear()
    compare.pool.used.clear()


def test_models_that_fit_together_load_together(models):
    out = call("POST", "/api/compare/load", json={"models": [{"path": models["a"], "name": "A"}, {"path": models["b"]}]})
    assert out["together"] is True
    assert sorted(s["path"] for s in out["servers"]) == sorted([models["a"], models["b"]])
    assert sorted(compare.loaded_paths()) == sorted([models["a"], models["b"]])
    # Asking again changes nothing; dropping one unloads it.
    call("POST", "/api/compare/load", json={"models": [{"path": models["a"]}, {"path": models["b"]}]})
    out = call("POST", "/api/compare/load", json={"models": [{"path": models["a"]}]})
    assert [s["path"] for s in out["servers"]] == [models["a"]]


def test_models_too_big_together_go_one_at_a_time(models):
    out = call("POST", "/api/compare/load", json={"models": [{"path": models["a"]}, {"path": models["huge"]}]})
    assert out == {"together": False, "servers": []}
    out = call("POST", "/api/compare/load", json={"models": [{"path": models["huge"]}]})
    assert out["together"] is True and [s["path"] for s in out["servers"]] == [models["huge"]]


def test_the_chat_model_is_not_loaded_twice(models, monkeypatch):
    monkeypatch.setattr(manager, "path", models["huge"])
    out = call("POST", "/api/compare/load", json={"models": [{"path": models["huge"]}, {"path": models["a"]}]})
    assert [s["path"] for s in out["servers"]] == [models["a"]]
    monkeypatch.setattr(manager, "path", None)


def test_a_model_too_big_for_the_machine_is_refused(models, monkeypatch):
    monkeypatch.setattr(memory, "memory_budget", lambda: {"kind": "ram", "total": 8 * GB, "free": 8 * GB})
    detail = call("POST", "/api/compare/load", json={"models": [{"path": models["huge"], "name": "Huge"}]}, status=409)["detail"]
    assert "Huge needs about 9.0 GB" in detail


def test_requests_route_by_model(models):
    call("POST", "/api/compare/load", json={"models": [{"path": models["a"], "name": "A"}]})
    r = client.post("/v1/chat/completions", headers=AUTH, json={"model": "A", "messages": []})
    assert r.status_code == 409 and "A is still loading" in r.json()["detail"]
    # Any other model name goes to chat's model, and nothing is loaded there.
    r = client.post("/v1/chat/completions", headers=AUTH, json={"model": "other", "messages": []})
    assert r.status_code == 409 and "No model is loaded" in r.json()["detail"]
    call("POST", "/api/compare/unload")
    assert compare.loaded_paths() == []


def test_a_model_whose_header_cant_be_read_is_left_to_llama_cpp(models, monkeypatch):
    def unreadable(path, context):
        raise TypeError("not a chat model")

    monkeypatch.setattr(memory, "local_chat_estimate", unreadable)
    out = call("POST", "/api/compare/load", json={"models": [{"path": models["a"]}, {"path": models["b"]}]})
    assert out["together"] is True and len(out["servers"]) == 2


def test_an_unsupported_architecture_is_named():
    import subprocess

    server = RuntimeManager()
    server.model, server.state, server.port = "parakeet.gguf", "loading", 1
    line = "llama_model_load: error loading model: unknown model architecture: 'parakeet'"
    server.proc = subprocess.Popen(["sh", "-c", f'echo "{line}"; exit 1'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    server._watch(server.proc)
    assert server.state == "failed"
    assert server.error.startswith("llama.cpp can't run parakeet.gguf: it doesn't support the parakeet architecture.\n")
