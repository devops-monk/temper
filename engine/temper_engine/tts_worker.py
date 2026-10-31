"""Reads text aloud with Kokoro, inside the voice environment (tts.py), one request at a time.

Run by path with that environment's Python, so it imports nothing from temper_engine. The first argument is
the folder holding kokoro-v1.0.onnx and voices-v1.0.bin. Each line on stdin is a request:
{"text", "voice", "speed", "lang", "out", "format"} writes the audio to `out` and answers {"done": {"duration"}};
{"voices": true} answers {"done": {"voices": [...]}}. Errors answer {"error": "..."}. The model stays loaded.
"""

from __future__ import annotations

import json
import sys
import wave


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def write(samples, rate: int, out: str, fmt: str) -> None:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    if fmt == "pcm":  # raw 16-bit little-endian, as OpenAI's API gives it
        with open(out, "wb") as f:
            f.write(pcm.tobytes())
        return
    if fmt == "wav":
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        return
    import av

    codec = {"mp3": "mp3", "opus": "libopus", "aac": "aac", "flac": "flac"}[fmt]
    container = {"opus": "ogg", "aac": "adts"}.get(fmt, fmt)
    with av.open(out, "w", format=container) as c:
        rate_out = 48000 if fmt == "opus" else rate
        stream = c.add_stream(codec, rate=rate_out)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = rate
        resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=rate_out)
        for f in resampler.resample(frame) + resampler.resample(None):
            for packet in stream.encode(f):
                c.mux(packet)
        for packet in stream.encode(None):
            c.mux(packet)


def main() -> None:
    from pathlib import Path

    from kokoro_onnx import Kokoro

    folder = Path(sys.argv[1])
    kokoro = Kokoro(str(folder / "kokoro-v1.0.onnx"), str(folder / "voices-v1.0.bin"))
    send({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("voices"):
                send({"done": {"voices": sorted(kokoro.get_voices())}})
                continue
            text = (req.get("text") or "").strip()
            if not text:
                raise ValueError("There's no text to read.")
            samples, rate = kokoro.create(text, voice=req["voice"], speed=float(req.get("speed") or 1.0), lang=req.get("lang") or "en-us")
            write(samples, rate, req["out"], req.get("format") or "wav")
            send({"done": {"duration": round(len(samples) / rate, 2)}})
        except Exception as e:
            send({"error": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
