import sys
import time

import pytest
from fastapi.testclient import TestClient

from temper_engine import stems
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
FAKE = """
import json, sys
from pathlib import Path
source, out = sys.argv[1], Path(sys.argv[2])
def emit(event, **data): print("@@" + json.dumps({"event": event, **data}), flush=True)
if open(source, "rb").read() == b"bad":
    emit("error", message="That file has no audio in it.")
    sys.exit(1)
emit("progress", value=0.5)
written = {}
for name in ("drums", "bass", "other", "vocals", "instrumental"):
    (out / f"{name}.wav").write_bytes(name.encode())
    written[name] = str(out / f"{name}.wav")
emit("result", stems=written, folder=str(out), duration=8.0)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The engine with Demucs installed, played by a stand-in worker that writes a file per stem."""
    script = tmp_path / "fake_stems.py"
    script.write_text(FAKE)
    monkeypatch.setattr(stems, "WORKER", script)
    monkeypatch.setattr(stems, "python", lambda: sys.executable)
    monkeypatch.setattr(stems, "downloaded", lambda: True)
    with TestClient(create_app(token="t")) as c:
        yield c


def wait(client, job):
    for _ in range(200):
        done = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if done["status"] != "running":
            return done
        time.sleep(0.05)
    raise AssertionError("still running")


def test_a_song_is_split_and_listed_with_vocals_first(client, tmp_path):
    song = tmp_path / "My Song.mp3"
    song.write_bytes(b"music")
    done = wait(client, client.post("/api/stems/jobs", headers=H, json={"path": str(song)}).json())
    assert done["status"] == "done" and done["result"]["duration"] == 8.0
    listed = client.get("/api/stems", headers=H).json()["separations"]
    assert [s["name"] for s in listed] == ["My Song"] and list(listed[0]["stems"]) == ["vocals", "instrumental", "drums", "bass", "other"]
    sid = listed[0]["id"]
    assert client.get(f"/api/stems/{sid}/vocals.wav", headers=H).content == b"vocals"
    assert client.get(f"/api/stems/{sid}/..%2F..%2Fsecret.wav", headers=H).status_code == 404
    assert client.delete(f"/api/stems/{sid}", headers=H).json()["separations"] == []


def test_a_failed_split_reports_why(client, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"bad")
    done = wait(client, client.post("/api/stems/jobs", headers=H, json={"path": str(bad)}).json())
    assert done["status"] == "failed" and done["error"] == "That file has no audio in it."
    assert client.post("/api/stems/jobs", headers=H, json={"path": str(tmp_path / "missing.mp3")}).status_code == 404


def test_it_says_what_is_missing(client, monkeypatch, tmp_path):
    song = tmp_path / "s.wav"
    song.write_bytes(b"x")
    monkeypatch.setattr(stems, "downloaded", lambda: False)
    assert "isn't downloaded" in client.post("/api/stems/jobs", headers=H, json={"path": str(song)}).json()["detail"]
    monkeypatch.setattr(stems, "python", lambda: None)
    assert "isn't installed" in client.post("/api/stems/jobs", headers=H, json={"path": str(song)}).json()["detail"]
