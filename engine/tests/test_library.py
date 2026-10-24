import json
import os
import struct

from fastapi.testclient import TestClient

from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def gguf(path, name="Tiny Model"):
    """A minimal GGUF v3 header: no tensors, one string key."""
    s = lambda t: struct.pack("<Q", len(t)) + t.encode()  # noqa: E731
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("general.name") + struct.pack("<I", 8) + s(name) + b"\0" * 64)
    return path


def post(path):
    return client.post("/api/models/import", json={"path": str(path)}, headers=AUTH)


def local():
    return [m for m in client.get("/api/models/local", headers=AUTH).json() if m.get("imported")]


def test_import_a_gguf_and_its_split_parts(tmp_path):
    gguf(tmp_path / "m-Q4_K_M-00001-of-00002.gguf")
    gguf(tmp_path / "m-Q4_K_M-00002-of-00002.gguf")
    gguf(tmp_path / "other-Q8_0.gguf")
    r = post(tmp_path / "m-Q4_K_M-00001-of-00002.gguf").json()
    assert r["kind"] == "gguf" and r["name"] == "Tiny Model"
    assert len(r["gguf"]) == 1 and r["gguf"][0]["complete"] and len(r["gguf"][0]["parts"]) == 2
    entry = local()[0]
    assert entry["id"] == "Tiny Model" and entry["gguf"][0]["path"].endswith("00001-of-00002.gguf")


def test_import_folders(tmp_path):
    hf = tmp_path / "my-finetune"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (hf / "model.safetensors").write_bytes(b"x")
    (hf / "tokenizer.json").write_text("{}")
    assert post(hf).json()["kind"] == "safetensors"
    mlx = tmp_path / "mlx-4bit"
    mlx.mkdir()
    (mlx / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}}))
    (mlx / "model.safetensors").write_bytes(b"x")
    r = post(mlx).json()
    assert r["kind"] == "mlx" and "tokenizer" in r["warnings"][0]
    folder = tmp_path / "ggufs"
    folder.mkdir()
    gguf(folder / "a-Q4_0.gguf")
    gguf(folder / "a-Q8_0.gguf")
    assert len(post(folder).json()["gguf"]) == 2
    assert [m["kind"] for m in local()] == ["gguf", "mlx", "safetensors"]


def test_bad_imports_say_why(tmp_path):
    (tmp_path / "fake.gguf").write_bytes(b"nope")
    assert "isn't a GGUF" in post(tmp_path / "fake.gguf").json()["detail"]
    (tmp_path / "notes.txt").write_text("hi")
    assert post(tmp_path / "notes.txt").status_code == 400
    (tmp_path / "params.json").write_text("{}")
    assert "Mistral-format" in post(tmp_path).json()["detail"]
    assert post(tmp_path / "missing").status_code == 400


def test_moved_files_show_as_missing_and_remove_keeps_files(tmp_path):
    f = gguf(tmp_path / "m.gguf")
    post(f)
    f.rename(tmp_path / "moved.gguf")
    assert "can't find" in local()[0]["missing"]
    client.delete("/api/models/import", params={"path": str(f)}, headers=AUTH)
    assert local() == [] and (tmp_path / "moved.gguf").exists()


def test_hugging_face_cache_paths(tmp_path):
    repo = tmp_path / "models--org--Tiny-GGUF"
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc")
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    gguf(repo / "blobs" / "f00d", name="")
    # A native relative target, as huggingface_hub writes it (Windows doesn't resolve "/" in symlinks).
    (snap / "tiny-Q4_0.gguf").symlink_to(os.path.join("..", "..", "blobs", "f00d"))
    r = post(snap / "tiny-Q4_0.gguf").json()
    assert r["kind"] == "gguf" and r["name"] == "org/Tiny-GGUF" and r["path"].endswith("tiny-Q4_0.gguf")
    assert post(repo).json()["path"] == str(snap)
