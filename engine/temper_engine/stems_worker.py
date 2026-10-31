"""Splits a song into stems with Demucs, inside the audio environment (stems.py).

Run by path with that environment's Python, so it imports nothing from temper_engine:
`python stems_worker.py <audio file> <output folder>`. It prints job events as `@@{json}` lines (the
format jobs.run_command reads): progress as it goes, then a result naming each stem's WAV file. With
vocals among the stems it also writes "instrumental", everything else mixed back together.

Audio is decoded with PyAV, so any format FFmpeg reads works. It runs on CUDA when there is a GPU and
on the CPU otherwise: PyTorch's Metal backend crashes inside Demucs on some macOS versions, and a crash
there can't be caught.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path


def emit(event: str, **data) -> None:
    print("@@" + json.dumps({"event": event, **data}), flush=True)


def decode(path: str, rate: int, channels: int):
    """The audio as float32 samples shaped (channels, frames) at `rate`."""
    import av
    import numpy as np

    layout = "stereo" if channels == 2 else "mono"
    out = []
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("That file has no audio in it.")
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        for frame in container.decode(stream):
            out.extend(f.to_ndarray() for f in resampler.resample(frame))
        out.extend(f.to_ndarray() for f in resampler.resample(None))
    if not out:
        raise ValueError("That file has no audio in it.")
    return np.concatenate(out, axis=1).astype("float32")


def save(samples, rate: int, path: Path) -> None:
    import numpy as np

    pcm = (np.clip(samples.T, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(samples.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main() -> None:
    source, folder = sys.argv[1], Path(sys.argv[2])
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from demucs.api import Separator

        emit("phase", name="Loading Demucs")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def progress(info: dict) -> None:
            if info["state"] == "end" and info["audio_length"]:
                share = (info["model_idx_in_bag"] + min(1.0, info["segment_offset"] / info["audio_length"])) / info["models"]
                emit("progress", value=round(min(0.99, share), 3))

        separator = Separator("htdemucs", device=device, callback=progress)
        emit("phase", name=f"Reading {Path(source).name}")
        audio = decode(source, separator.samplerate, separator.audio_channels)
        emit("phase", name=f"Separating on the {'GPU' if device == 'cuda' else 'CPU'}")
        _, stems = separator.separate_tensor(torch.from_numpy(audio), separator.samplerate)
        written = {}
        for name, wav in stems.items():
            written[name] = str(folder / f"{name}.wav")
            save(wav.cpu().numpy(), separator.samplerate, Path(written[name]))
        if "vocals" in stems:
            rest = sum(wav for name, wav in stems.items() if name != "vocals")
            written["instrumental"] = str(folder / "instrumental.wav")
            save(rest.cpu().numpy(), separator.samplerate, Path(written["instrumental"]))
        emit("progress", value=1.0)
        emit("result", stems=written, folder=str(folder), duration=round(audio.shape[1] / separator.samplerate, 2))
    except Exception as e:
        emit("error", message=str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}")
        for leftover in (*folder.glob("*.wav"), folder / "source.json"):
            leftover.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
