"""Transcribes audio with Whisper, inside the speech environment (speech.py), one request at a time.

It's run by path with that environment's Python, so it imports nothing from temper_engine. Each line on
stdin is a request: {"audio": path, "model": folder, "language": code or null, "task": "transcribe" or
"translate", "prompt": text or null}. It answers with JSON lines: {"segment": {...}, "progress": 0..1} as
it goes, then {"done": {...}} or {"error": "..."}. The model stays loaded between requests.

`backend` is "mlx" (mlx-whisper, Apple Silicon) or "faster" (faster-whisper, CTranslate2 on CUDA or CPU).
Audio is decoded with PyAV, so any format FFmpeg reads works and no ffmpeg program is needed.
"""

from __future__ import annotations

import json
import sys

RATE = 16000
# mlx-whisper transcribes a whole array before returning; pieces this long give progress on long files.
PIECE = 10 * 60 * RATE


def decode(path: str):
    """The audio as 16 kHz mono float32 samples."""
    import av
    import numpy as np

    out = []
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("That file has no audio in it.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
        for frame in container.decode(stream):
            out.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        out.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def segment(start: float, end: float, text: str) -> dict:
    return {"start": round(start, 2), "end": round(end, 2), "text": text.strip()}


def run_mlx(req: dict, audio) -> dict:
    import mlx_whisper

    duration = len(audio) / RATE
    texts, language, prompt = [], req.get("language"), req.get("prompt")
    for offset in range(0, max(len(audio), 1), PIECE):
        piece = audio[offset:offset + PIECE]
        if len(piece) < RATE // 10:  # a last sliver shorter than 0.1 s
            break
        r = mlx_whisper.transcribe(piece, path_or_hf_repo=req["model"], language=language, task=req.get("task") or "transcribe", initial_prompt=prompt, verbose=None,
                                   condition_on_previous_text=True)
        language = language or r.get("language")
        base = offset / RATE
        for s in r.get("segments") or []:
            if s["text"].strip():
                send({"segment": segment(base + s["start"], base + s["end"], s["text"]), "progress": min(1.0, (base + s["end"]) / duration) if duration else 1.0})
        texts.append(r.get("text", "").strip())
        prompt = texts[-1][-200:] or prompt  # carries names and spelling across pieces
    return {"text": " ".join(t for t in texts if t), "language": language, "duration": round(duration, 2)}


_faster: dict = {}


def run_faster(req: dict, audio) -> dict:
    from faster_whisper import WhisperModel

    model = _faster.get(req["model"])
    if model is None:
        try:
            model = WhisperModel(req["model"], device="auto", compute_type="default")
        except Exception:  # a GPU without the CUDA libraries CTranslate2 needs: the CPU still works
            model = WhisperModel(req["model"], device="cpu", compute_type="int8")
        _faster.clear()
        _faster[req["model"]] = model
    segments, info = model.transcribe(audio, language=req.get("language"), task=req.get("task") or "transcribe", initial_prompt=req.get("prompt"), vad_filter=True)
    texts = []
    for s in segments:
        texts.append(s.text.strip())
        send({"segment": segment(s.start, s.end, s.text), "progress": min(1.0, s.end / info.duration) if info.duration else 1.0})
    return {"text": " ".join(t for t in texts if t), "language": info.language, "duration": round(info.duration, 2)}


def main() -> None:
    backend = sys.argv[1]
    send({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            audio = decode(req["audio"])
            send({"done": (run_mlx if backend == "mlx" else run_faster)(req, audio)})
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)})


if __name__ == "__main__":
    main()
