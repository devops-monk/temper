import struct

from fastapi.testclient import TestClient

from temper_engine import catalog
from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}
SETTINGS = {"system": "Be terse.", "temperature": 0.3, "top_p": 0.9, "top_k": 40, "min_p": 0, "repeat_penalty": 1, "presence_penalty": 0, "max_tokens": 1024}


def call(method, path, status=200, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == status, r.text
    return r.json()


def test_presets_for_a_model_or_any_and_defaults():
    terse = call("POST", "/api/presets", json={"name": "Terse", "settings": SETTINGS})
    coder = call("POST", "/api/presets", json={"name": "Coder", "model": "qwen.gguf", "settings": {**SETTINGS, "temperature": 0.1}, "thinking": {"on": False}})
    call("POST", "/api/presets", status=409, json={"name": "coder", "model": "qwen.gguf", "settings": SETTINGS})
    call("POST", "/api/presets", status=400, json={"name": "  ", "settings": SETTINGS})
    assert [p["name"] for p in call("GET", "/api/presets", params={"model": "qwen.gguf"})["presets"]] == ["Coder", "Terse"]
    assert [p["name"] for p in call("GET", "/api/presets", params={"model": "llama.gguf"})["presets"]] == ["Terse"]

    call("PUT", "/api/presets", json={"model": "qwen.gguf", "preset": coder["id"]})
    call("PUT", "/api/presets", json={"model": "llama.gguf", "preset": terse["id"]})
    call("PUT", "/api/presets", status=400, json={"model": "llama.gguf", "preset": coder["id"]})
    assert call("GET", "/api/presets", params={"model": "qwen.gguf"})["default"] == coder["id"]

    # Narrowing a preset to one model drops it as other models' default; deleting drops it everywhere.
    call("PUT", f"/api/presets/{terse['id']}", json={"name": "Terse", "model": "qwen.gguf", "settings": SETTINGS})
    assert call("GET", "/api/presets", params={"model": "llama.gguf"})["default"] is None
    call("DELETE", f"/api/presets/{coder['id']}")
    assert call("GET", "/api/presets", params={"model": "qwen.gguf"})["default"] is None
    call("PUT", "/api/presets", json={"model": "qwen.gguf", "preset": None})


def gguf_with(path, pairs):
    """A GGUF header with string and float32/uint32 keys."""
    s = lambda t: struct.pack("<Q", len(t)) + t.encode()  # noqa: E731
    body = b""
    for key, value in pairs.items():
        if isinstance(value, str):
            body += s(key) + struct.pack("<I", 8) + s(value)
        elif isinstance(value, float):
            body += s(key) + struct.pack("<If", 6, value)
        else:
            body += s(key) + struct.pack("<II", 4, value)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + body + b"\0" * 32)
    return path


def test_recommended_settings_from_the_gguf(tmp_path):
    f = gguf_with(tmp_path / "my-model.gguf", {"general.name": "My Model", "general.sampling.temp": 0.8, "general.sampling.top_k": 20, "general.sampling.penalty_repeat": 1.05})
    match = call("GET", "/api/catalog/match", params={"model": str(f)})["match"]
    assert match["sampling"] == {"temperature": 0.8, "top_k": 20, "repetition_penalty": 1.05}
    assert match["source"] == "the settings stored in this GGUF"


def test_recommended_settings_from_the_base_model(tmp_path, monkeypatch):
    f = gguf_with(tmp_path / "ft.gguf", {"general.name": "Fine-tune", "general.base_model.0.repo_url": "https://huggingface.co/acme/base-7b"})
    seen = []
    monkeypatch.setattr(catalog, "_hub_sampling", lambda repo: seen.append(repo) or (repo, {"temperature": 0.6, "top_p": 0.95}))
    match = call("GET", "/api/catalog/match", params={"model": str(f)})["match"]
    assert seen == ["acme/base-7b"] and match["source"] == "generation_config.json from acme/base-7b"
    assert catalog._repos("/c/models--org--name-GGUF/snapshots/x/a.gguf", {}) == ["org/name-GGUF"]
