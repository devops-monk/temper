import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from temper_engine import speech
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
FAKE = """
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if open(req["audio"], "rb").read() == b"bad":
        print(json.dumps({"error": "That file has no audio in it."}), flush=True)
        continue
    words = "hello from whisper" if req["task"] == "transcribe" else "in english"
    print(json.dumps({"segment": {"start": 0.0, "end": 1.5, "text": words}, "progress": 0.5}), flush=True)
    print(json.dumps({"segment": {"start": 61.25, "end": 62.0, "text": "again"}, "progress": 1.0}), flush=True)
    print(json.dumps({"done": {"text": words + " again", "language": req["language"] or "en", "duration": 62.0, "model_folder": req["model"]}}), flush=True)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The engine with speech installed and a stand-in worker that answers like Whisper."""
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE)
    monkeypatch.setattr(speech, "WORKER", script)
    monkeypatch.setattr(speech, "python", lambda: sys.executable)
    monkeypatch.setattr(speech, "_folder", lambda model: f"/models/{model}" if model == speech.MODELS[0]["id"] else None)
    monkeypatch.setattr(speech, "default_model", lambda: speech.MODELS[0]["id"])
    with TestClient(create_app(token="t")) as c:
        yield c
    speech.worker.stop()


def test_dictation_returns_the_text_and_keeps_the_worker_loaded(client):
    r = client.post("/api/speech/transcribe?suffix=.webm", headers=H, content=b"RIFF....").json()
    assert r["text"] == "hello from whisper again" and r["language"] == "en" and len(r["segments"]) == 2
    first = speech.worker.proc.pid
    assert client.post("/api/speech/transcribe?language=de", headers=H, content=b"x").json()["language"] == "de"
    assert speech.worker.proc.pid == first  # the same process: the model stays loaded
    assert client.post("/api/speech/transcribe", headers=H, content=b"bad").json()["detail"] == "That file has no audio in it."
    assert client.post("/api/speech/transcribe", headers=H, content=b"").status_code == 400


def test_the_openai_audio_api(client):
    files = {"file": ("talk.mp3", b"mp3 bytes", "audio/mpeg")}
    assert client.post("/v1/audio/transcriptions", headers=H, files=files, data={"model": "whisper-1"}).json() == {"text": "hello from whisper again"}
    srt = client.post("/v1/audio/transcriptions", headers=H, files=files, data={"response_format": "srt"}).text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nhello from whisper\n\n2\n00:01:01,250 --> 00:01:02,000\nagain")
    vtt = client.post("/v1/audio/transcriptions", headers=H, files=files, data={"response_format": "vtt"}).text
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n")
    verbose = client.post("/v1/audio/transcriptions", headers=H, files=files, data={"response_format": "verbose_json"}).json()
    assert verbose["duration"] == 62.0 and verbose["segments"][1]["id"] == 1
    assert client.post("/v1/audio/translations", headers=H, files=files).json()["text"] == "in english again"


def test_a_file_job_saves_text_and_subtitles(client, tmp_path):
    audio = tmp_path / "Meeting notes.m4a"
    audio.write_bytes(b"audio")
    job = client.post("/api/speech/jobs", headers=H, json={"path": str(audio)}).json()
    for _ in range(100):
        done = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if done["status"] != "running":
            break
        time.sleep(0.05)
    assert done["status"] == "done", done
    files = done["result"]["files"]
    assert files["txt"].endswith("Meeting-notes.txt") and open(files["txt"]).read() == "hello from whisper again\n"
    assert "00:01:01,250 --> 00:01:02,000" in open(files["srt"]).read()
    assert json.load(open(files["json"]))["source"] == str(audio)
    assert client.post("/api/speech/jobs", headers=H, json={"path": str(tmp_path / "missing.wav")}).status_code == 404


def test_it_says_what_is_missing(client, monkeypatch):
    st = client.get("/api/speech", headers=H).json()
    assert st["installed"] and st["ready"] and [m["downloaded"] for m in st["models"]] == [True, False, False]
    assert client.put("/api/speech/model", headers=H, json={"model": "openai/whisper-tiny"}).status_code == 400
    monkeypatch.setattr(speech, "default_model", lambda: speech.MODELS[1]["id"])
    assert "isn't downloaded" in client.post("/api/speech/transcribe", headers=H, content=b"x").json()["detail"]
    monkeypatch.setattr(speech, "python", lambda: None)
    assert "isn't installed" in client.post("/api/speech/transcribe", headers=H, content=b"x").json()["detail"]
