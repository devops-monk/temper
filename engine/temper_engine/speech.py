"""Speech to text with Whisper: dictation in Chat, transcribing files, and the OpenAI audio API.

Whisper runs in its own environment under engines/speech, installed on request like the opt-in engines:
mlx-whisper on Apple Silicon (without the PyTorch it only needs for converting models), faster-whisper
elsewhere (CTranslate2, on an NVIDIA GPU when its libraries are there, else the CPU). speech_worker.py
runs in that environment and keeps the model loaded between requests; it's stopped after IDLE seconds
unused. Models are ordinary Hugging Face downloads, read from the models folder with no network.

Transcripts of files are saved under transcripts/ in Temper's home as text, SRT and WebVTT subtitles,
and JSON with each segment's times.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import engines, hardware, home, prefs
from .jobs import Job, registry

router = APIRouter(tags=["speech"])

BACKEND = "mlx" if engines.APPLE else "faster"
NAME = {"mlx": "Whisper on MLX", "faster": "faster-whisper"}[BACKEND]
# mlx-whisper's own requirements without torch, which only its model converter imports; PyAV decodes audio.
MLX_DEPS = ["mlx>=0.11", "numba", "numpy", "tqdm", "more-itertools", "tiktoken", "huggingface_hub", "scipy", "av>=12"]
PACKAGES = {"mlx": MLX_DEPS, "faster": ["faster-whisper>=1.0"]}[BACKEND]
NO_DEPS = {"mlx": ["mlx-whisper>=0.4"], "faster": []}[BACKEND]

MODELS = {
    "mlx": [
        {"id": "mlx-community/whisper-large-v3-turbo", "name": "Large v3 Turbo", "size": 1_610_000_000, "about": "Fast and accurate in about 100 languages."},
        {"id": "mlx-community/whisper-small-mlx", "name": "Small", "size": 480_000_000, "about": "Quickest; fine for clear English dictation."},
        {"id": "mlx-community/whisper-large-v3-mlx", "name": "Large v3", "size": 3_080_000_000, "about": "The most accurate, at about a third of Turbo's speed."},
    ],
    "faster": [
        {"id": "deepdml/faster-whisper-large-v3-turbo-ct2", "name": "Large v3 Turbo", "size": 1_620_000_000, "about": "Fast and accurate in about 100 languages; best with a GPU."},
        {"id": "Systran/faster-whisper-small", "name": "Small", "size": 490_000_000, "about": "Quick on a CPU; fine for clear English dictation."},
        {"id": "Systran/faster-whisper-large-v3", "name": "Large v3", "size": 3_090_000_000, "about": "The most accurate, and the slowest."},
    ],
}[BACKEND]
IDLE = 600.0
WORKER = Path(__file__).with_name("speech_worker.py")


def venv() -> Path:
    return home() / "engines" / "speech"


def python() -> str | None:
    p = engines._venv_python(venv())
    return str(p) if p.is_file() and (venv() / ".installed").is_file() else None


def _folder(model: str) -> str | None:
    return engines.local_folder(model)


def default_model() -> str:
    """The model picked in Settings; otherwise Turbo, or Small on a computer with no GPU for faster-whisper."""
    chosen = prefs.load().get("speech_model")
    if chosen:
        return chosen
    return MODELS[1]["id"] if BACKEND == "faster" and not hardware.nvidia_gpus() else MODELS[0]["id"]


def status() -> dict:
    chosen = default_model()
    return {
        "backend": NAME,
        "installed": python() is not None,
        "model": chosen,
        "ready": python() is not None and _folder(chosen) is not None,
        "models": [{**m, "downloaded": _folder(m["id"]) is not None} for m in MODELS],
        "running": worker.running(),
    }


def _not_ready(model: str) -> HTTPException | None:
    if python() is None:
        return HTTPException(409, "Speech to text isn't installed. Install it in Settings → Speech to text, or run `temper speech install`.")
    if _folder(model) is None:
        return HTTPException(409, f"The speech model {model} isn't downloaded. Download it in Settings → Speech to text.")
    return None


# --- The worker ------------------------------------------------------------------------------------


class Worker:
    """A worker script in its own environment, answering JSON-line requests one at a time; started on first
    use and stopped after IDLE seconds unused. `command()` gives its command line. Also used by tts.py."""

    def __init__(self, name: str, command) -> None:
        self.name, self.command = name, command
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.used = 0.0
        self._reaper: threading.Thread | None = None

    def _log(self) -> Path:
        return home() / f"{self.name}.log"

    def _start(self) -> None:
        log = open(self._log(), "ab")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        self.proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", bufsize=1, env=env)
        log.close()
        if json.loads(self._line()).get("ready") is not True:
            raise RuntimeError(f"The {self.name} worker didn't start.")
        if self._reaper is None:
            self._reaper = threading.Thread(target=self._reap, daemon=True)
            self._reaper.start()

    def _line(self) -> str:
        assert self.proc and self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            self.proc = None
            tail = self._log().read_text(errors="replace").strip().splitlines()[-3:]
            raise RuntimeError(f"The {self.name} worker stopped. " + " ".join(tail))
        return line

    def request(self, payload: dict, on_message=None) -> dict:
        """Sends one request; passes each message before the last to `on_message`, and returns the "done" one's value.
        Raises ValueError with the worker's "error"."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            try:
                while True:
                    msg = json.loads(self._line())
                    if "error" in msg:
                        raise ValueError(msg["error"])
                    if "done" in msg:
                        return msg["done"]
                    if on_message:
                        on_message(msg)
            finally:
                self.used = time.time()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        with self.lock:
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                self.proc = None

    def _reap(self) -> None:
        while True:
            time.sleep(30)
            if self.proc is not None and time.time() - self.used > IDLE and not self.lock.locked():
                self.stop()


worker = Worker("speech", lambda: [python(), "-u", str(WORKER), BACKEND])


def transcribe(audio: str, model: str | None = None, language: str | None = None, task: str = "transcribe", prompt: str | None = None, on_segment=None) -> dict:
    model = model or default_model()
    if error := _not_ready(model):
        raise error
    segments: list[dict] = []

    def got(msg: dict) -> None:
        if "segment" in msg:
            segments.append(msg["segment"])
            if on_segment:
                on_segment(msg["segment"], msg.get("progress"))

    done = worker.request({"audio": audio, "model": _folder(model), "language": language or None, "task": task, "prompt": prompt or None}, got)
    return {**done, "model": model, "segments": segments}


def _bytes(data: bytes, suffix: str, **kw) -> dict:
    """Transcribes audio sent in a request, from a temporary file."""
    if not data:
        raise HTTPException(400, "No audio was sent.")
    with tempfile.NamedTemporaryFile(suffix=suffix or ".audio", dir=home(), delete=False) as f:
        f.write(data)
    try:
        return transcribe(f.name, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        os.unlink(f.name)


# --- Subtitles ---------------------------------------------------------------------------------------


def _stamp(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def srt(segments: list[dict]) -> str:
    return "".join(f"{i}\n{_stamp(s['start'], ',')} --> {_stamp(s['end'], ',')}\n{s['text']}\n\n" for i, s in enumerate(segments, 1))


def vtt(segments: list[dict]) -> str:
    return "WEBVTT\n\n" + "".join(f"{_stamp(s['start'], '.')} --> {_stamp(s['end'], '.')}\n{s['text']}\n\n" for s in segments)


def save(source: str, result: dict) -> dict:
    """Writes a file's transcript in each format; returns their paths."""
    stem = re.sub(r"[^\w.-]+", "-", Path(source).stem).strip("-") or "audio"
    folder = home() / "transcripts" / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}"
    folder.mkdir(parents=True, exist_ok=True)
    files = {"txt": folder / f"{stem}.txt", "srt": folder / f"{stem}.srt", "vtt": folder / f"{stem}.vtt", "json": folder / f"{stem}.json"}
    files["txt"].write_text(result["text"] + "\n", encoding="utf-8")
    files["srt"].write_text(srt(result["segments"]), encoding="utf-8")
    files["vtt"].write_text(vtt(result["segments"]), encoding="utf-8")
    files["json"].write_text(json.dumps({"source": source, **result}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: str(v) for k, v in files.items()}


# --- Routes ------------------------------------------------------------------------------------------


@router.get("/api/speech")
def speech_status():
    return status()


@router.post("/api/speech/install")
def install():
    job = registry.create("engine-install", f"Install speech to text ({NAME})")

    def run(j: Job) -> None:
        engines.install_venv(j, venv(), PACKAGES, "speech to text", no_deps=NO_DEPS)

    registry.run_thread(job, run)
    return job.summary()


@router.delete("/api/speech")
def remove():
    worker.stop()
    shutil.rmtree(venv(), ignore_errors=True)
    return status()


class Choice(BaseModel):
    model: str


@router.put("/api/speech/model")
def choose(req: Choice):
    if req.model not in {m["id"] for m in MODELS}:
        raise HTTPException(400, f"Pick one of: {', '.join(m['id'] for m in MODELS)}.")
    prefs.update("speech_model", req.model)
    worker.stop()  # the next request loads the new one
    return status()


@router.post("/api/speech/warm")
def warm():
    """Loads the model ahead of a dictation (Chat calls this as recording starts), off the request."""
    model = default_model()
    if _not_ready(model) is None and not worker.running() and not worker.lock.locked():
        def run() -> None:
            silence = home() / "speech-warm.wav"
            import wave

            with wave.open(str(silence), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\0\0" * 8000)
            try:
                transcribe(str(silence), model)
            except (RuntimeError, ValueError):
                pass  # the real request reports it
            finally:
                silence.unlink(missing_ok=True)

        threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@router.post("/api/speech/transcribe")
async def dictate(request: Request, language: str | None = None, task: str = "transcribe", suffix: str = ".webm"):
    """The request's body is the audio (Chat's recording); the reply is its text."""
    data = await request.body()
    return await run_in_threadpool(_bytes, data, suffix if suffix.startswith(".") else "." + suffix, language=language, task=task)


class FileJob(BaseModel):
    path: str
    language: str | None = None
    task: str = "transcribe"
    model: str | None = None


@router.post("/api/speech/jobs")
def transcribe_file(req: FileJob):
    if not Path(req.path).is_file():
        raise HTTPException(404, f"No file at {req.path}.")
    if error := _not_ready(req.model or default_model()):
        raise error
    job = registry.create("transcribe", f"Transcribe {Path(req.path).name}")
    job.progress = 0.0

    def run(j: Job) -> None:
        def show(seg: dict, progress: float | None) -> None:
            j.log(f"[{_stamp(seg['start'], '.')[:8]}] {seg['text']}")
            if progress is not None:
                j.progress = progress
                j.emit("progress", value=progress)

        result = transcribe(req.path, req.model, req.language, req.task, on_segment=show)
        j.result.update(files=save(req.path, result), language=result["language"], duration=result["duration"], text=result["text"][:4000])

    registry.run_thread(job, run)
    return job.summary()


def _openai(task: str, file: UploadFile, model: str | None, language: str | None, prompt: str | None, response_format: str):
    known = {m["id"] for m in MODELS}
    data = file.file.read()
    result = _bytes(data, Path(file.filename or "").suffix, model=model if model in known else None, language=language, task=task, prompt=prompt)
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "srt":
        return PlainTextResponse(srt(result["segments"]))
    if response_format == "vtt":
        return PlainTextResponse(vtt(result["segments"]), media_type="text/vtt")
    if response_format == "verbose_json":
        segments = [{"id": i, **s} for i, s in enumerate(result["segments"])]
        return JSONResponse({"task": task, "language": result["language"], "duration": result["duration"], "text": result["text"], "segments": segments})
    return JSONResponse({"text": result["text"]})


@router.post("/v1/audio/transcriptions")
def v1_transcriptions(file: UploadFile = File(...), model: str | None = Form(None), language: str | None = Form(None), prompt: str | None = Form(None),
                      response_format: str = Form("json")):
    """OpenAI's transcription API; `model` may name one of MODELS, anything else (whisper-1) uses the chosen one."""
    return _openai("transcribe", file, model, language, prompt, response_format)


@router.post("/v1/audio/translations")
def v1_translations(file: UploadFile = File(...), model: str | None = Form(None), prompt: str | None = Form(None), response_format: str = Form("json")):
    """Speech in any language as English text."""
    return _openai("translate", file, model, None, prompt, response_format)
