import struct

import pytest
from fastapi.testclient import TestClient

from temper_engine import model_settings
from temper_engine.app import create_app
from temper_engine.runtime import RuntimeManager, manager

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def call(method, path, status=200, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == status, r.text
    return r.json()


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(path, arch="qwen3", tokenizer=("gpt2", "qwen2"), vocab=151936, layers=28, size=0, nextn=0):
    """A GGUF header with the keys Temper reads, padded to `size` bytes."""
    kv = [("general.architecture", 8, _string(arch)), (f"{arch}.block_count", 4, struct.pack("<I", layers)), (f"{arch}.context_length", 4, struct.pack("<I", 40960)),
          (f"{arch}.embedding_length", 4, struct.pack("<I", 1024)), (f"{arch}.attention.head_count", 4, struct.pack("<I", 16)), (f"{arch}.attention.head_count_kv", 4, struct.pack("<I", 8))]
    if nextn:
        kv.append((f"{arch}.nextn_predict_layers", 4, struct.pack("<I", nextn)))
    kv += [("tokenizer.ggml.model", 8, _string(tokenizer[0])), ("tokenizer.ggml.pre", 8, _string(tokenizer[1]))]
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv) + 1)
    for key, kind, value in kv:
        body += _string(key) + struct.pack("<I", kind) + value
    body += _string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", vocab)
    path.write_bytes(body + b"\0" * max(0, size - len(body)))
    return str(path)


def test_settings_become_llama_server_flags():
    s = {**model_settings.DEFAULTS, "cache_type": "q8_0", "speculation": "draft", "draft": "/m/small.gguf", "draft_max": 12}
    assert model_settings.flags(s) == ["-ctk", "q8_0", "-ctv", "q8_0", "-md", "/m/small.gguf", "--spec-type", "draft-simple", "--spec-draft-n-max", "12", "-ngld", "999", "-ctkd", "q8_0", "-ctvd", "q8_0"]
    assert model_settings.flags({**model_settings.DEFAULTS, "speculation": "ngram"}) == ["--spec-type", "ngram-mod"]
    assert model_settings.flags(model_settings.DEFAULTS) == []
    with pytest.raises(Exception, match="needs flash attention"):
        model_settings.check({**model_settings.DEFAULTS, "cache_type": "q4_0", "flash_attention": False})


def test_drafts_need_the_same_tokenizer_and_a_smaller_size(tmp_path, monkeypatch):
    big = gguf(tmp_path / "Qwen3-8B.gguf", size=40_000)
    small = gguf(tmp_path / "Qwen3-0.6B.gguf", layers=28, size=4_000)
    other = gguf(tmp_path / "Llama-1B.gguf", arch="llama", tokenizer=("gpt2", "llama-bpe"), vocab=128256, size=4_000)
    near = gguf(tmp_path / "Qwen-tiny.gguf", vocab=151936 + 64, size=4_000)
    monkeypatch.setattr(model_settings, "_local_ggufs", lambda: [{"path": p, "name": p.rsplit("/", 1)[-1], "size": (tmp_path / p.rsplit("/", 1)[-1]).stat().st_size} for p in (big, small, other, near)])
    out = call("GET", "/api/runtime/settings", params={"path": big})
    assert {d["path"] for d in out["drafts"]} == {small, near}
    assert out["layers"] == 28 and out["context_max"] == 40960 and out["mtp"] is False
    assert call("GET", "/api/runtime/settings", params={"path": gguf(tmp_path / "GLM.gguf", arch="glm4moe", nextn=1)})["mtp"] is True


def test_saved_settings_apply_to_loads_and_reload_the_loaded_model(tmp_path, monkeypatch):
    model = gguf(tmp_path / "Qwen3-4B.gguf", size=8_000)
    draft = gguf(tmp_path / "Qwen3-0.6B.gguf", size=2_000)
    loads = []
    monkeypatch.setattr(RuntimeManager, "load", lambda self, req: loads.append(model_settings.resolve(req.path, req.model_dump())))
    monkeypatch.setattr(manager, "path", model)
    monkeypatch.setattr(manager, "state", "ready")
    out = call("PUT", "/api/runtime/settings", json={"path": model, "settings": {"context": 16384, "cache_type": "q8_0", "speculation": "draft", "draft": draft}})
    assert out["settings"]["context"] == 16384 and out["settings"]["draft"] == draft
    assert loads and loads[-1]["cache_type"] == "q8_0" and loads[-1]["speculation"] == "draft"
    # A load request's own fields win; turning drafting off forgets the draft.
    assert model_settings.resolve(model, {"context": 4096})["context"] == 4096
    call("PUT", "/api/runtime/settings", json={"path": model, "settings": {"speculation": "none"}, "reload": False})
    assert model_settings.saved(model)["draft"] is None and model_settings.saved(model)["context"] == 16384
    call("PUT", "/api/runtime/settings", 400, json={"path": model, "settings": {"speculation": "draft", "draft": str(tmp_path / "gone.gguf")}})


def test_the_memory_estimate_counts_the_cache_type_and_the_draft(tmp_path):
    model = gguf(tmp_path / "Qwen3-4B.gguf", size=8_000)
    draft = gguf(tmp_path / "Qwen3-0.6B.gguf", size=2_000)
    plain = call("POST", "/api/memory/chat", json={"path": model})
    q8 = call("POST", "/api/memory/chat", json={"path": model, "cache_type": "q8_0"})
    assert plain["context"] == 8192 and q8["kv_cache"] == pytest.approx(plain["kv_cache"] * 17 / 32, rel=0.01)
    drafted = call("POST", "/api/memory/chat", json={"path": model, "speculation": "draft", "draft": draft})
    assert drafted["draft"] > 2_000 and drafted["total"] == plain["total"] + drafted["draft"]
