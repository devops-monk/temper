import os
import time

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import constants

from temper_engine import prefs, storage
from temper_engine.app import create_app
from temper_engine.runtime import manager

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    folder = tmp_path / "hub"
    folder.mkdir()
    monkeypatch.setattr(storage, "DEFAULT_DIR", str(folder))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(folder))
    yield folder
    prefs.update("models_dir", None)


def repo(cache, name="org/M-GGUF", files=None):
    """A cache entry like huggingface_hub writes: blobs, a snapshot of links, refs/main."""
    root = cache / ("models--" + name.replace("/", "--"))
    (root / "blobs").mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("c1")
    for snap, entries in (files or {"c1": {"a-Q4_0.gguf": "aaaa", "a-Q8_0.gguf": "bbbbbbbb"}}).items():
        (root / "snapshots" / snap).mkdir(parents=True)
        for fname, content in entries.items():
            blob = root / "blobs" / f"sha-{content}"
            blob.write_text(content)
            (root / "snapshots" / snap / fname).symlink_to(os.path.join("..", "..", "blobs", blob.name))
    return root


def test_delete_one_quant_frees_its_blob(cache):
    root = repo(cache)
    r = client.post("/api/models/local/delete-files", json={"repo_id": "org/M-GGUF", "files": ["a-Q8_0.gguf"]}, headers=AUTH).json()
    assert r["freed"] == 8 and not (root / "blobs" / "sha-bbbbbbbb").exists() and (root / "snapshots/c1/a-Q4_0.gguf").exists()


def test_delete_split_parts_in_a_quant_folder(cache):
    root = repo(cache, files={"c1": {}})
    (root / "snapshots/c1/Q4").mkdir()
    for i, c in ((1, "p1"), (2, "p2")):
        (root / "blobs" / c).write_text(c)
        (root / f"snapshots/c1/Q4/m-0000{i}-of-00002.gguf").symlink_to(os.path.join("..", "..", "..", "blobs", c))
    r = client.post("/api/models/local/delete-files", json={"repo_id": "org/M-GGUF", "files": ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]}, headers=AUTH).json()
    assert r["freed"] == 4 and not any((root / "blobs").iterdir())


def test_a_blob_another_snapshot_uses_is_kept(cache):
    root = repo(cache, files={"c1": {"a.gguf": "same"}, "c2": {"a.gguf": "same"}})
    r = client.post("/api/models/local/delete-files", json={"repo_id": "org/M-GGUF", "files": ["a.gguf"]}, headers=AUTH).json()
    assert r["freed"] == 4 and not (root / "blobs/sha-same").exists()  # both links went, so the blob did
    root2 = repo(cache, name="org/N", files={"c1": {"a.gguf": "x"}, "c2": {"b.gguf": "x"}})
    client.post("/api/models/local/delete-files", json={"repo_id": "org/N", "files": ["a.gguf"]}, headers=AUTH)
    assert (root2 / "blobs/sha-x").exists()  # still linked as b.gguf


def test_delete_a_model_and_refusals(cache, monkeypatch):
    root = repo(cache)
    monkeypatch.setattr(manager, "path", str(root / "snapshots/c1/a-Q4_0.gguf"))
    assert client.delete("/api/models/local", params={"repo_id": "org/M-GGUF"}, headers=AUTH).status_code == 409
    monkeypatch.setattr(manager, "path", None)
    assert client.delete("/api/models/local", params={"repo_id": "../../etc"}, headers=AUTH).status_code == 404
    assert client.delete("/api/models/local", params={"repo_id": "org/M-GGUF"}, headers=AUTH).json()["freed"] == 14  # blobs plus refs/main
    assert not root.exists()


def test_move_models_to_a_new_folder(cache, tmp_path):
    repo(cache)
    target = tmp_path / "elsewhere" / "models"
    r = client.put("/api/storage/models-dir", json={"path": str(target), "move": True}, headers=AUTH).json()
    job_id = r["job"]["id"]
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    moved = target / "models--org--M-GGUF" / "snapshots/c1/a-Q4_0.gguf"
    assert moved.read_text() == "aaaa" and moved.is_symlink()
    assert constants.HF_HUB_CACHE == str(target) == os.environ["HF_HUB_CACHE"]
    assert client.get("/api/storage", params={"fresh": True}, headers=AUTH).json()["custom"] is True
    assert client.put("/api/storage/models-dir", json={"path": str(target / "inside"), "move": False}, headers=AUTH).status_code == 400
    assert client.put("/api/storage/models-dir", json={"path": "relative/dir"}, headers=AUTH).status_code == 400


def test_usage_report(cache):
    repo(cache)
    body = client.get("/api/storage", params={"fresh": True}, headers=AUTH).json()
    assert body["usage"]["models"] == 14 and body["disk"]["free"] > 0
    assert set(body["usage"]) == {"models", "exports", "runs", "datasets", "engine"}
