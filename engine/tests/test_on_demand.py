import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from temper_engine import compare, memory, on_demand, prefs
from temper_engine.app import create_app
from temper_engine.model_names import pick
from temper_engine.runtime import RuntimeManager, manager, route

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}
GB = memory.GB


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Three downloaded models needing 2, 3 and 5 GB with 8 GB free; servers are ready as soon as they start."""
    sizes = {"small": 2, "mid": 3, "big": 5, "broken": 1}
    rows = []
    for name in sizes:
        path = tmp_path / f"{name}-Q4_K_M.gguf"
        path.write_bytes(b"GGUF")
        rows.append({"id": f"acme/{name}-GGUF", "gguf": [{"name": path.name, "path": str(path), "size": 1}]})
    monkeypatch.setattr(on_demand, "_local", lambda: rows)
    monkeypatch.setattr(memory, "local_chat_estimate", lambda path, context: {"total": sizes[path.stem.split("-")[0]] * GB})

    def free():
        used = sum(sizes[Path(p).stem.split("-")[0]] for p in compare.pool.servers)
        return {"kind": "ram", "total": 16 * GB, "free": (8 - used) * GB}

    monkeypatch.setattr(memory, "memory_budget", free)
    loads = []

    def fake_load(self, req):
        loads.append(req.path)
        if "broken" in req.path:
            self.state, self.error = "failed", "The model file is damaged."
            return
        self.model, self.path, self.state, self.port, self.used = req.name, req.path, "ready", 1, time.time()

    def fake_unload(self):
        self.model, self.path, self.state = None, None, "idle"

    monkeypatch.setattr(RuntimeManager, "load", fake_load)
    monkeypatch.setattr(RuntimeManager, "unload", fake_unload)
    monkeypatch.setattr(compare.Pool, "_start_reaper", lambda self: None)
    yield loads
    for d in (compare.pool.servers, compare.pool.used, compare.pool.kept, compare.pool.on_demand, compare.pool.ttl, compare.pool.aliases):
        d.clear()
    fake_unload(manager)


def test_a_request_naming_a_downloaded_model_loads_it(local):
    server, _, body = route({"model": "acme/small-GGUF", "messages": [], "ttl": 120})
    assert server.model == "small-Q4_K_M.gguf" and server.state == "ready"
    assert "ttl" not in body  # not passed on to the model's server
    pool = compare.pool
    assert pool.on_demand == {server.path} and pool.ttl[server.path] == 120
    # Asked again, by the same name or the file's, it's the same server: nothing loads twice.
    assert route({"model": "acme/small-GGUF", "messages": []})[0] is server
    assert route({"model": "small-Q4_K_M", "messages": []})[0] is server
    assert local == [server.path]


def test_other_names_are_not_loaded(local):
    with pytest.raises(HTTPException) as e:
        route({"model": "gpt-4o", "messages": []})  # a client's default: goes to chat's model, of which there's none
    assert e.value.status_code == 409 and "No model is loaded" in e.value.detail
    with pytest.raises(HTTPException):
        route({"model": "smal", "messages": []})  # part of a name loads nothing
    assert local == []


def test_it_can_be_turned_off(local):
    prefs.update("load_on_demand", False)
    with pytest.raises(HTTPException):
        route({"model": "acme/small-GGUF", "messages": []})
    assert local == [] and on_demand.listing() == []


def test_room_is_made_from_models_loaded_on_demand_least_recently_used_first(local):
    small = route({"model": "acme/small-GGUF", "messages": []})[0]
    mid = route({"model": "acme/mid-GGUF", "messages": []})[0]
    small.used, mid.used = time.time(), time.time() - 60  # mid went unused longer
    big = route({"model": "acme/big-GGUF", "messages": []})[0]  # 5 GB with 3 free: unloading mid (unused longest) is enough
    assert big.state == "ready"
    assert set(compare.pool.servers) == {small.path, big.path}


def test_models_loaded_by_hand_are_never_unloaded_for_room(local):
    from temper_engine.loaded import load_alongside
    from temper_engine.runtime import LoadRequest

    hand = on_demand.resolve("acme/mid-GGUF").path
    load_alongside(LoadRequest(path=hand))
    with pytest.raises(HTTPException) as e:
        route({"model": "acme/big-GGUF", "messages": []})
    assert e.value.status_code == 409 and "Unload a model" in e.value.detail
    assert hand in compare.pool.servers


def test_a_failed_load_says_why_and_leaves_nothing_behind(local):
    with pytest.raises(HTTPException) as e:
        route({"model": "acme/broken-GGUF", "messages": []})
    assert e.value.status_code == 503 and "damaged" in e.value.detail
    assert compare.pool.servers == {}


def test_ttl_unloads_after_its_own_idle_time(local):
    server = route({"model": "acme/small-GGUF", "messages": [], "ttl": 30})[0]
    compare.pool.reap(now=server.used + 20)
    assert server.path in compare.pool.servers
    compare.pool.reap(now=server.used + 31)
    assert compare.pool.servers == {} and compare.pool.aliases == {}


def test_v1_models_lists_what_can_load(local):
    route({"model": "acme/small-GGUF", "messages": []})
    data = client.get("/v1/models", headers=AUTH).json()["data"]
    assert [(m["id"], m.get("loaded", True)) for m in data] == [
        ("small-Q4_K_M.gguf", True),
        ("acme/mid-GGUF", False),
        ("acme/big-GGUF", False),
        ("acme/broken-GGUF", False),
    ]


def test_pick_matches_whole_names_for_requests_and_parts_for_people():
    rows = [{"id": "acme/Qwen3-4B-GGUF", "gguf": [{"name": "Qwen3-4B-Q8_0.gguf", "path": "/m/q8"}, {"name": "Qwen3-4B-Q4_K_M.gguf", "path": "/m/q4"}]}]
    assert pick(rows, "acme/qwen3-4b-gguf", loose=False) == ("/m/q4", "Qwen3-4B-Q4_K_M.gguf")
    assert pick(rows, "Qwen3-4B-GGUF", loose=False)[0] == "/m/q4"
    assert pick(rows, "qwen3-4b-q8_0", loose=False)[0] == "/m/q8"
    assert pick(rows, "qwen3", quant="Q8_0")[0] == "/m/q8"
    with pytest.raises(LookupError):
        pick(rows, "qwen3", loose=False)
    with pytest.raises(LookupError, match="no Q2_K"):
        pick(rows, "qwen3", quant="Q2_K")


def test_speech_and_embedding_ggufs_are_not_offered(local, monkeypatch):
    monkeypatch.setattr(memory, "read_gguf_metadata", lambda read: {"general.architecture": "parakeet"})
    assert on_demand.listing() == []
    with pytest.raises(HTTPException):
        route({"model": "acme/small-GGUF", "messages": []})
    assert local == []
