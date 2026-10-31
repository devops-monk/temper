import sys

import pytest
from fastapi.testclient import TestClient

from temper_engine import tts
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
FAKE = """
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("voices"):
        print(json.dumps({"done": {"voices": ["af_heart", "af_nova", "bm_george", "ef_dora"]}}), flush=True)
        continue
    open(req["out"], "w").write(f"{req['format']}|{req['voice']}|{req['lang']}|{req['speed']}|{req['text']}")
    print(json.dumps({"done": {"duration": 1.25}}), flush=True)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The engine with Kokoro installed, played by a stand-in worker that writes what it was asked."""
    script = tmp_path / "fake_tts.py"
    script.write_text(FAKE)
    monkeypatch.setattr(tts, "python", lambda: sys.executable)
    monkeypatch.setattr(tts, "model_folder", lambda: str(tmp_path))
    monkeypatch.setattr(tts.worker, "command", lambda: [sys.executable, "-u", str(script), str(tmp_path)])
    monkeypatch.setattr(tts, "_voices", [])
    with TestClient(create_app(token="t")) as c:
        yield c
    tts.worker.stop()


def test_voices_are_described_by_language(client):
    out = client.get("/api/tts/voices", headers=H).json()
    george = next(v for v in out["voices"] if v["id"] == "bm_george")
    assert george == {"id": "bm_george", "name": "George", "language": "British English", "lang": "en-gb", "gender": "male"}
    assert out["voice"] == "af_heart"
    assert client.put("/api/tts/voice", headers=H, json={"voice": "ef_dora"}).json()["voice"] == "ef_dora"
    assert client.put("/api/tts/voice", headers=H, json={"voice": "nobody"}).status_code == 400


def test_speaking_returns_audio_in_the_voices_language(client):
    r = client.post("/api/tts/speak", headers=H, json={"text": " Hola ", "voice": "ef_dora", "speed": 1.5})
    assert r.headers["content-type"] == "audio/wav" and r.headers["x-duration"] == "1.25"
    assert r.text == "wav|ef_dora|es|1.5|Hola"
    assert client.post("/api/tts/speak", headers=H, json={"text": "  "}).status_code == 400
    assert client.post("/api/tts/speak", headers=H, json={"text": "x" * (tts.MAX_TEXT + 1)}).status_code == 413


def test_saved_clips_make_a_gallery(client):
    clip = client.post("/api/tts/speak", headers=H, json={"text": "Keep this", "save": True}).json()
    assert clip["voice"] == "af_heart" and clip["duration"] == 1.25
    assert [c["id"] for c in client.get("/api/tts/gallery", headers=H).json()["clips"]] == [clip["id"]]
    assert client.get(f"/api/tts/gallery/{clip['id']}/audio", headers=H).text.endswith("Keep this")
    assert client.delete(f"/api/tts/gallery/{clip['id']}", headers=H).json()["clips"] == []
    assert client.get("/api/tts/gallery/..%2Fsecret/audio", headers=H).status_code == 404


def test_the_openai_speech_api_maps_its_voices(client):
    r = client.post("/v1/audio/speech", headers=H, json={"input": "Hi", "voice": "nova", "response_format": "mp3"})
    assert r.headers["content-type"] == "audio/mpeg" and r.text.startswith("mp3|af_nova|en-us")
    assert client.post("/v1/audio/speech", headers=H, json={"input": "Hi", "voice": "shimmer"}).text.startswith("mp3|af_heart")  # none by that name
    assert client.post("/v1/audio/speech", headers=H, json={"input": "Hi", "response_format": "midi"}).status_code == 400


def test_it_says_what_is_missing(client, monkeypatch):
    monkeypatch.setattr(tts, "model_folder", lambda: None)
    assert "aren't downloaded" in client.post("/api/tts/speak", headers=H, json={"text": "hi"}).json()["detail"]
    monkeypatch.setattr(tts, "python", lambda: None)
    assert "isn't installed" in client.post("/api/tts/speak", headers=H, json={"text": "hi"}).json()["detail"]
