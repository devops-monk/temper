import hashlib
import json
import sys

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


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in ollama command is a script with a shebang")
def test_an_export_is_added_to_ollama_with_a_modelfile(tmp_path, monkeypatch):
    import time

    from temper_engine import ollama

    model = tmp_path / "run-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    # A stand-in for the ollama command that records what it was asked.
    fake = tmp_path / "ollama"
    fake.write_text(f"#!{sys.executable}\nimport sys, pathlib\npathlib.Path(sys.argv[0]).with_name('called').write_text(' '.join(sys.argv[1:]))\nprint('success')\n")
    fake.chmod(0o755)
    monkeypatch.setattr(ollama, "binary", lambda: str(fake))
    monkeypatch.setattr(ollama, "running", lambda: "0.40.0")
    with TestClient(create_app(token="t")) as c:
        h = {"Authorization": "Bearer t"}
        assert c.post("/api/ollama/create", headers=h, json={"gguf": str(model), "name": "Not OK"}).status_code == 422
        assert c.post("/api/ollama/create", headers=h, json={"gguf": str(tmp_path / "missing.gguf"), "name": "x"}).status_code == 404
        job = c.post("/api/ollama/create", headers=h, json={"gguf": str(model), "name": "tutor:q4_k_m", "system": "Be brief."}).json()
        for _ in range(100):
            if c.get(f"/api/jobs/{job['id']}", headers=h).json()["status"] != "running":
                break
            time.sleep(0.05)
        assert c.get(f"/api/jobs/{job['id']}", headers=h).json()["status"] == "done"
    assert (tmp_path / "called").read_text() == f"create tutor:q4_k_m -f {tmp_path / 'run-Q4_K_M.Modelfile'}"
    assert (tmp_path / "run-Q4_K_M.Modelfile").read_text() == f'FROM {model}\nSYSTEM """Be brief."""\n'
    monkeypatch.setattr(ollama, "running", lambda: None)
    with TestClient(create_app(token="t")) as c:
        assert c.post("/api/ollama/create", headers={"Authorization": "Bearer t"}, json={"gguf": str(model), "name": "x"}).status_code == 409
