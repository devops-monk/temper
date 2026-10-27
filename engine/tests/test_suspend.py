import pytest

from temper_engine import runtime
from temper_engine.runtime import LoadRequest, RuntimeManager


class Reply:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def json(self):
        return self.data


def test_suspend_saves_used_slots_and_resume_restores_them(monkeypatch):
    server = RuntimeManager()
    server.state, server.port, server.model, server.path = "ready", 1, "m", "/m.gguf"
    server.request = LoadRequest(path="/m.gguf", name="m", context=16384)
    calls = []

    def post(url, params=None, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], params["action"], json["filename"]))
        if params["action"] == "save":
            (runtime.kv_dir() / json["filename"]).write_bytes(b"kv")
            return Reply({"n_saved": 900 if url.endswith("/0") else 0})
        return Reply({"n_restored": 900})

    monkeypatch.setattr(runtime.httpx, "get", lambda url, timeout=None: Reply([{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]))
    monkeypatch.setattr(runtime.httpx, "post", post)
    out = server.suspend()
    assert out["state"] == "idle" and out["suspended"] == {"model": "m", "slots": 1}
    (slot, name), = server.suspended["slots"]
    assert slot == 0 and (runtime.kv_dir() / name).is_file() and len(list(runtime.kv_dir().iterdir())) == 1

    loads = []
    monkeypatch.setattr(RuntimeManager, "load", lambda self, req, restore=None: loads.append((req, restore)))
    server.resume()
    req, restore = loads[-1]
    assert req.context == 16384 and restore == [(0, name)] and server.suspended is None
    server.port, server.restore = 1, restore
    server._restore_slots()
    assert server.restored == 900 and not (runtime.kv_dir() / name).exists()


def test_a_busy_model_isnt_suspended(monkeypatch):
    server = RuntimeManager()
    server.state, server.port, server.request = "ready", 1, LoadRequest(path="/m.gguf")
    monkeypatch.setattr(runtime.httpx, "get", lambda url, timeout=None: Reply([{"id": 0, "is_processing": True}]))
    with pytest.raises(Exception, match="writing a reply"):
        server.suspend()
    with pytest.raises(Exception, match="No model is suspended"):
        RuntimeManager().resume()
