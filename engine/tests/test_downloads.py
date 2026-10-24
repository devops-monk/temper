import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from temper_engine import downloads
from temper_engine.jobs import Job

DATA = bytes(range(256)) * 40_000  # about 10 MB
SHA = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    honour_range = True
    requests: list[str] = []

    def do_GET(self):
        rng = self.headers.get("Range")
        Handler.requests.append(rng or "")
        start = int(rng.split("=")[1].split("-")[0]) if rng and Handler.honour_range else 0
        body = DATA[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # the client paused mid-transfer
            pass

    def log_message(self, *_args):
        pass


@pytest.fixture
def hub(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/model.gguf"
    Handler.honour_range, Handler.requests = True, []
    monkeypatch.setattr(downloads.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(downloads, "hf_hub_url", lambda repo, name: url)
    monkeypatch.setattr(downloads, "get_token", lambda: None)
    monkeypatch.setattr(downloads, "get_hf_file_metadata", lambda *_a, **_k: SimpleNamespace(etag=f'"{SHA}"', size=len(DATA)))
    monkeypatch.setattr(downloads, "hf_hub_download", lambda repo, name: str(downloads.repo_cache(repo) / "blobs" / SHA))
    yield tmp_path
    server.shutdown()


def partial(repo="org/model"):
    path = downloads.repo_cache(repo) / "blobs" / f"{SHA}.temper-partial"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_resumes_a_partial_download(hub):
    partial().write_bytes(DATA[:3_000_000])
    job = Job("download", "t")
    downloads.download(job, "org/model", [("model.gguf", len(DATA))])
    assert Handler.requests == ["bytes=3000000-"]
    blob = downloads.repo_cache("org/model") / "blobs" / SHA
    assert blob.read_bytes() == DATA and not partial().exists()
    assert job.result["received"] == len(DATA) and job.progress == 1.0
    assert any("Resuming" in e.get("line", "") for e in job.events)


def test_server_ignoring_range_starts_over(hub):
    Handler.honour_range = False
    partial().write_bytes(DATA[:3_000_000])
    downloads.download(Job("download", "t"), "org/model", [("model.gguf", len(DATA))])
    assert (downloads.repo_cache("org/model") / "blobs" / SHA).read_bytes() == DATA


def test_corrupt_partial_is_deleted(hub):
    partial().write_bytes(b"x" * 3_000_000)
    with pytest.raises(RuntimeError, match="checksum"):
        downloads.download(Job("download", "t"), "org/model", [("model.gguf", len(DATA))])
    assert not partial().exists()


def test_pause_keeps_the_partial(hub):
    job = Job("download", "t")
    job.cancel_requested = True
    downloads.download(job, "org/model", [("model.gguf", len(DATA))])
    assert partial().exists()
    assert any("Paused" in e.get("line", "") for e in job.events)


def test_split_ggufs_are_grouped():
    from temper_engine.hub import group_gguf

    files = [
        {"name": "Q4_K_M/M-Q4_K_M-00002-of-00002.gguf", "size": 3},
        {"name": "Q4_K_M/M-Q4_K_M-00001-of-00002.gguf", "size": 5},
        {"name": "M-Q8_0-00001-of-00003.gguf", "size": 4},
        {"name": "M-Q2_K.gguf", "size": 2},
        {"name": "mmproj-F16.gguf", "size": 1},
    ]
    grouped = {g["name"]: g for g in group_gguf(files)}
    assert list(grouped) == ["M-Q2_K.gguf", "M-Q8_0-00001-of-00003.gguf", "Q4_K_M/M-Q4_K_M-00001-of-00002.gguf"]
    q4 = grouped["Q4_K_M/M-Q4_K_M-00001-of-00002.gguf"]
    assert q4["size"] == 8 and len(q4["parts"]) == 2 and q4["complete"]
    assert not grouped["M-Q8_0-00001-of-00003.gguf"]["complete"]
