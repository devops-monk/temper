import os

from fastapi.testclient import TestClient
from huggingface_hub import constants, hf_hub_url, scan_cache_dir

from temper_engine import downloads, sources
from temper_engine.app import create_app
from temper_engine.jobs import Job

from .test_downloads import DATA, SHA, hub  # noqa: F401  (the local file server fixture)

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def put(**body):
    return client.put("/api/hub/source", json=body, headers=AUTH)


def test_mirror_changes_the_endpoint_everywhere():
    try:
        assert put(source="mirror", mirror="hf-mirror.com/").json()["endpoint"] == "https://hf-mirror.com"
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert hf_hub_url("org/m", "a.gguf").startswith("https://hf-mirror.com/org/m/resolve/")
        assert sources.settings().source == "mirror"  # saved
        put(source="huggingface")
        assert "HF_ENDPOINT" not in os.environ and constants.ENDPOINT == "https://huggingface.co"
    finally:
        sources.apply(sources.HubSettings())


def test_bad_mirror_is_refused():
    assert put(source="mirror", mirror="ftp://x").status_code == 400


def test_modelscope_search_maps_to_hub_shape(monkeypatch):
    seen = {}

    def fake(path, **params):
        seen.update(params)
        return {"data": {"models": [
            {"id": "Qwen/Qwen3-0.6B-GGUF", "downloads": 5, "likes": 1, "last_modified": "2025-05-12T11:16:51Z", "tags": ["library:gguf", "license:apache-2.0"], "tasks": ["text-generation"], "gated": False},
            {"id": "Qwen/Qwen3-0.6B", "downloads": 9, "tags": ["library:transformers"], "tasks": []},
        ]}}

    monkeypatch.setattr(sources, "_ms_get", fake)
    put(source="modelscope")
    try:
        results = client.get("/api/models/search", params={"q": "qwen3", "gguf": True, "sort": "lastModified"}, headers=AUTH).json()
    finally:
        put(source="huggingface")
    assert seen["search"] == "qwen3 gguf" and seen["sort"] == "last_modified"
    assert [r["id"] for r in results] == ["Qwen/Qwen3-0.6B-GGUF"]
    assert results[0]["tags"] == ["gguf"] and results[0]["pipeline"] == "text-generation"


def test_modelscope_download_lands_in_the_hf_cache(hub, monkeypatch):  # noqa: F811
    url = downloads.hf_hub_url("x", "y")  # the test server, from the hub fixture
    monkeypatch.setattr(sources, "modelscope", lambda: True)
    monkeypatch.setattr(sources, "ms_url", lambda repo, name: url)
    monkeypatch.setattr(sources, "ms_files", lambda repo: [{"name": "sub/model.gguf", "size": len(DATA), "sha256": SHA}])
    monkeypatch.setattr(sources, "ms_commit", lambda repo: "a" * 40)
    job = Job("download", "t")
    downloads.download(job, "Qwen/M-GGUF", [("sub/model.gguf", len(DATA))])
    snapshot = downloads.repo_cache("Qwen/M-GGUF") / "snapshots" / ("a" * 40) / "sub" / "model.gguf"
    assert snapshot.read_bytes() == DATA and job.result["path"] == str(snapshot)
    repo = next(r for r in scan_cache_dir(hub).repos if r.repo_id == "Qwen/M-GGUF")
    assert {f.file_name for r in repo.revisions for f in r.files} == {"model.gguf"}
