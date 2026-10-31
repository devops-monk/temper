"""Text to speech with Kokoro: reading chat replies aloud, a voice library, a gallery of saved clips, and
OpenAI's /v1/audio/speech.

Kokoro (82M parameters, Apache-2.0) runs on ONNX Runtime through kokoro-onnx, installed on request into
its own environment under engines/voice with PyAV for MP3, Opus, AAC and FLAC. Its model and its 54 voices
in nine languages are one Hugging Face download. tts_worker.py keeps the model loaded between requests,
on the same terms as the speech-to-text worker (speech.Worker).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import engines, home, prefs, storage
from .jobs import Job, registry
from .speech import Worker

router = APIRouter(tags=["speech"])

REPO = "fastrtc/kokoro-onnx"
FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
PACKAGES = ["kokoro-onnx>=0.4", "av>=12"]
DEFAULT_VOICE = "af_heart"
# A voice's first letter is its language (and the phonemizer's language code); its second, f or m.
LANGUAGES = {
    "a": ("American English", "en-us"), "b": ("British English", "en-gb"), "e": ("Spanish", "es"), "f": ("French", "fr-fr"), "h": ("Hindi", "hi"),
    "i": ("Italian", "it"), "j": ("Japanese", "ja"), "p": ("Brazilian Portuguese", "pt-br"), "z": ("Mandarin Chinese", "cmn"),
}
MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/L16"}
WORKER = Path(__file__).with_name("tts_worker.py")
MAX_TEXT = 20_000


def venv() -> Path:
    return home() / "engines" / "voice"


def python() -> str | None:
    p = engines._venv_python(venv())
    return str(p) if p.is_file() and (venv() / ".installed").is_file() else None


def model_folder() -> str | None:
    """The downloaded snapshot holding Kokoro's model and voices."""
    snapshots = storage.models_dir() / ("models--" + REPO.replace("/", "--")) / "snapshots"
    found = sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if snapshots.is_dir() else []
    return next((str(s) for s in found if all((s / f).is_file() for f in FILES)), None)


worker = Worker("voice", lambda: [python(), "-u", str(WORKER), model_folder()])
_voices: list[str] = []


def voice_info(voice: str) -> dict:
    language, lang = LANGUAGES.get(voice[:1], ("English", "en-us"))
    name = voice.split("_", 1)[-1].replace("_", " ").title()
    return {"id": voice, "name": name, "language": language, "lang": lang, "gender": {"f": "female", "m": "male"}.get(voice[1:2], "")}


def voices() -> list[str]:
    """Kokoro's voices, asked of the worker once and then remembered."""
    global _voices
    cache = home() / "voices.json"
    if not _voices:
        try:
            _voices = json.loads(cache.read_text())
        except (OSError, ValueError):
            _voices = worker.request({"voices": True})["voices"]
            cache.write_text(json.dumps(_voices))
    return _voices


def chosen_voice() -> str:
    return prefs.load().get("tts_voice") or DEFAULT_VOICE


def status() -> dict:
    return {"installed": python() is not None, "downloaded": model_folder() is not None, "ready": python() is not None and model_folder() is not None,
            "voice": chosen_voice(), "repo": REPO, "size": 353_746_785, "running": worker.running()}


def _check() -> None:
    if python() is None:
        raise HTTPException(409, "Text to speech isn't installed. Install it on the Voice page, or run `temper say --install`.")
    if model_folder() is None:
        raise HTTPException(409, f"Kokoro's voices aren't downloaded. Download them on the Voice page ({REPO}).")


def speak(text: str, out: str, voice: str | None = None, speed: float = 1.0, fmt: str = "wav") -> dict:
    """Reads `text` into the file `out`; returns its duration."""
    _check()
    voice = voice or chosen_voice()
    if voice not in voices():
        raise HTTPException(400, f"No voice called {voice}. `temper say --voices` lists them.")
    text = text.strip()
    if not text:
        raise HTTPException(400, "There's no text to read.")
    if len(text) > MAX_TEXT:
        raise HTTPException(413, f"That's {len(text):,} characters; up to {MAX_TEXT:,} are read at once.")
    try:
        return worker.request({"text": text, "voice": voice, "speed": speed, "lang": voice_info(voice)["lang"], "out": out, "format": fmt})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _audio(text: str, voice: str | None, speed: float, fmt: str) -> Response:
    with tempfile.NamedTemporaryFile(suffix="." + fmt, dir=home(), delete=False) as f:
        path = f.name
    try:
        done = speak(text, path, voice, speed, fmt)
        data = Path(path).read_bytes()
    finally:
        os.unlink(path)
    return Response(data, media_type=MEDIA[fmt], headers={"X-Duration": str(done.get("duration", ""))})


# --- The gallery -------------------------------------------------------------------------------------


def gallery_dir() -> Path:
    folder = home() / "voice"
    folder.mkdir(exist_ok=True)
    return folder


def gallery() -> list[dict]:
    items = []
    for meta in gallery_dir().glob("*.json"):
        try:
            items.append(json.loads(meta.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(items, key=lambda i: i["created"], reverse=True)


# --- Routes ------------------------------------------------------------------------------------------


class Speak(BaseModel):
    text: str
    voice: str | None = None
    speed: float = Field(1.0, ge=0.5, le=2.0)
    save: bool = False


class VoiceChoice(BaseModel):
    voice: str


class OpenAISpeech(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    model: str | None = None
    speed: float = Field(1.0, ge=0.25, le=4.0)
    response_format: str = "mp3"


@router.get("/api/tts")
def tts_status():
    return status()


@router.post("/api/tts/install")
def install():
    job = registry.create("engine-install", "Install text to speech (Kokoro)")
    registry.run_thread(job, lambda j: engines.install_venv(j, venv(), PACKAGES, "text to speech"))
    return job.summary()


@router.delete("/api/tts")
def remove():
    worker.stop()
    shutil.rmtree(venv(), ignore_errors=True)
    return status()


@router.get("/api/tts/voices")
def list_voices():
    _check()
    return {"voices": [voice_info(v) for v in voices()], "voice": chosen_voice()}


@router.put("/api/tts/voice")
def choose_voice(req: VoiceChoice):
    _check()
    if req.voice not in voices():
        raise HTTPException(400, f"No voice called {req.voice}.")
    prefs.update("tts_voice", req.voice)
    return status()


@router.post("/api/tts/speak")
def speak_route(req: Speak):
    """Audio of the text as WAV; with `save`, kept in the gallery and described instead."""
    if not req.save:
        return _audio(req.text, req.voice, req.speed, "wav")
    clip = uuid.uuid4().hex[:12]
    voice = req.voice or chosen_voice()
    done = speak(req.text, str(gallery_dir() / f"{clip}.wav"), voice, req.speed)
    item = {"id": clip, "text": req.text.strip(), "voice": voice, "speed": req.speed, "duration": done.get("duration"), "created": time.time(),
            "path": str(gallery_dir() / f"{clip}.wav")}
    (gallery_dir() / f"{clip}.json").write_text(json.dumps(item, ensure_ascii=False))
    return item


@router.get("/api/tts/gallery")
def list_gallery():
    return {"clips": gallery()}


@router.get("/api/tts/gallery/{clip}/audio")
def clip_audio(clip: str):
    path = gallery_dir() / f"{clip}.wav"
    if not clip.isalnum() or not path.is_file():
        raise HTTPException(404, "No such clip.")
    return FileResponse(path, media_type="audio/wav")


@router.delete("/api/tts/gallery/{clip}")
def delete_clip(clip: str):
    if not clip.isalnum():
        raise HTTPException(404, "No such clip.")
    for suffix in (".wav", ".json"):
        (gallery_dir() / f"{clip}{suffix}").unlink(missing_ok=True)
    return {"clips": gallery()}


@router.post("/v1/audio/speech")
def v1_speech(req: OpenAISpeech):
    """OpenAI's speech API. Kokoro's own voice names work, and so do OpenAI's where Kokoro has one by that name
    (alloy, nova, echo, onyx, fable...); any other plays in the voice picked on the Voice page."""
    if req.response_format not in MEDIA:
        raise HTTPException(400, f"response_format can be {', '.join(MEDIA)}.")
    _check()
    known = voices()
    voice = req.voice if req.voice in known else next((v for v in known if v.endswith("_" + req.voice.lower())), chosen_voice())
    return _audio(req.input, voice, max(0.5, min(2.0, req.speed)), req.response_format)
