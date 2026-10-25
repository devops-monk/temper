import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from temper_engine.app import create_app

from .test_library import gguf

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def blob(root, data: bytes) -> str:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    (root / "blobs" / digest.replace(":", "-")).write_bytes(data)
    return digest


@pytest.fixture
def ollama(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    (root / "blobs").mkdir(parents=True)
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "nohome")
    weights = blob(root, gguf(tmp_path / "w.gguf").read_bytes())
    params = blob(root, json.dumps({"temperature": 0.6, "repeat_penalty": 1.1, "num_ctx": 8192, "stop": ["<|im_end|>"]}).encode())
    system = blob(root, b"You are a pirate.")

    def manifest(path, layers):
        p = root / "manifests" / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"layers": [{"mediaType": f"application/vnd.ollama.image.{t}", "digest": d, "size": 1} for t, d in layers]}))

    manifest("registry.ollama.ai/library/qwen2.5/7b", [("model", weights), ("params", params), ("system", system)])
    manifest("registry.ollama.ai/someone/tiny/latest", [("model", weights)])
    manifest("hf.co/org/Repo-GGUF/Q4_K_M", [("model", blob(root, b"not a gguf"))])  # skipped
    return root


def test_scan_reads_names_settings_and_skips_non_gguf(ollama):
    models = {m["name"]: m for m in client.get("/api/ollama", headers=AUTH).json()["models"]}
    assert set(models) == {"qwen2.5:7b", "someone/tiny:latest"}
    q = models["qwen2.5:7b"]
    assert q["sampling"] == {"temperature": 0.6, "repetition_penalty": 1.1} and q["context"] == 8192 and q["system"] == "You are a pirate."


def test_import_puts_it_in_the_library_and_chat_gets_its_settings(ollama):
    assert client.post("/api/ollama/import", json={"names": ["qwen2.5:7b"]}, headers=AUTH).status_code == 200
    entry = next(m for m in client.get("/api/models/local", headers=AUTH).json() if m.get("imported"))
    assert entry["id"] == "qwen2.5:7b" and entry["source"] == "Ollama" and entry["gguf"][0]["complete"]
    assert entry["gguf"][0]["name"] == "qwen2.5:7b"  # not the blob's hash
    match = client.get("/api/catalog/match", params={"model": entry["gguf"][0]["path"]}, headers=AUTH).json()["match"]
    assert match["publisher"] == "Ollama" and match["sampling"]["temperature"] == 0.6 and match["system"] == "You are a pirate."
    assert client.get("/api/ollama", headers=AUTH).json()["models"][0]["imported"] is True
    assert client.post("/api/ollama/import", json={"names": ["gone:1b"]}, headers=AUTH).status_code == 404
