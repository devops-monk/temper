"""Converting a safetensors checkpoint into a GGUF that chat can run, with llama.cpp's converter.

Chat runs GGUFs through llama-server, so a Hugging Face folder, a pre-quantized checkpoint (FP8,
NVFP4, MXFP4, compressed-tensors, GPTQ) or a Mistral-format folder becomes chat-ready by converting
it. The converter dequantizes what it can read; Temper then quantizes the result and adds it to the
library. The source folder is never changed: when its tokenizer config needs rewriting for the
converter, the converter reads a linked copy of the folder instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home, library, llamacpp
from .jobs import ANSI, Job, registry

router = APIRouter(prefix="/api/models/convert", tags=["models"])

QUANTS = {"q4_k_m": "Q4_K_M", "q8_0": "Q8_0", "f16": "F16"}
WRITING = re.compile(r"Writing:\s+(\d+)%")  # the converter's tqdm bar
QUANTIZING = re.compile(r"\[\s*(\d+)/\s*(\d+)\]")  # llama-quantize's "[ 12/ 291]" per tensor
# How much bigger a 16-bit copy is than the checkpoint, for the disk check.
GROWTH = {"FP8": 2, "INT8": 2, "FP4": 4, "NVFP4": 4, "MXFP4": 4, "INT4": 4, "4-bit": 4, "BitNet": 8}


class ConvertRequest(BaseModel):
    path: str
    quant: str = "q4_k_m"


def _f16_size(found: dict) -> int:
    label = (found.get("quantization") or {}).get("label", "")
    return found["size"] * next((g for key, g in GROWTH.items() if key in label), 1)


def _why_not(found: dict) -> str:
    if note := (found.get("quantization") or {}).get("note"):
        return note
    return {
        "gguf": "This is already a GGUF; load it from the list.",
        "mlx": "MLX checkpoints can't be converted to GGUF. Convert the original Hugging Face model instead.",
    }.get(found["kind"]) or " ".join(found.get("warnings") or []) or "Temper can't convert this model."


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name).strip("-") or "model"


def _converter_input(folder: Path, scratch: Path) -> Path:
    """`folder` itself, or a linked copy with a tokenizer config the converter's transformers 4 can read."""
    config_file = folder / "tokenizer_config.json"
    config = library._read_json(config_file) if config_file.is_file() else {}
    fixed = llamacpp.downgraded_tokenizer_config(config)
    if fixed is None:
        return folder
    shadow = scratch / "source"
    shadow.mkdir()
    for item in folder.iterdir():
        if item.name == "tokenizer_config.json":
            continue
        target = shadow / item.name
        try:
            os.symlink(item.resolve(), target, target_is_directory=item.is_dir())
        except OSError:  # Windows without Developer Mode can't make links
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, target)
    (shadow / "tokenizer_config.json").write_text(json.dumps(fixed, indent=2, ensure_ascii=False), encoding="utf-8")
    return shadow


def _run(job: Job, cmd: list[str], env: dict[str, str] | None, progress: tuple[float, float], pattern: re.Pattern) -> None:
    """Runs a tool, logging its output and mapping its own progress onto `progress` (start, end)."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace")
    job.process = proc
    start, end = progress
    last = ""
    assert proc.stdout
    for raw in proc.stdout:
        line = ANSI.sub("", raw.rstrip())
        if m := pattern.search(line):
            done = int(m.group(1)) / (int(m.group(2)) if m.lastindex == 2 else 100)
            job.progress = start + (end - start) * done
            job.emit("progress", value=job.progress)
            continue  # progress bars would flood the log
        if line.strip():
            job.log(line)
            last = line.strip()
    if proc.wait() != 0 and not job.cancel_requested:
        raise RuntimeError(_explain(last) or f"{Path(cmd[0]).name} exited with code {proc.returncode}")


def _explain(last: str) -> str:
    """The converter's last line, turned into something a person can act on where it's a known case."""
    if "NotImplementedError" in last and "Architecture" in last:
        return f"llama.cpp {llamacpp.TAG} can't convert this architecture yet ({last.split(':', 1)[-1].strip()})."
    if "NotImplementedError" in last:
        return f"llama.cpp's converter can't read this checkpoint: {last.split(':', 1)[-1].strip()}"
    return last[:400]


def _convert(job: Job, found: dict, quant: str, out: Path) -> None:
    source = Path(found["path"])
    name = found["name"]
    label = QUANTS[quant]
    job.emit("phase", name="Installing llama.cpp")
    llamacpp.install(lambda message, _p: job.log(message) if not message.startswith(("llama-", "cudart")) else None)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{_safe(name.split('/')[-1])}-{label}.gguf"
    # Q8_0 and F16 come straight out of the converter; Q4_K_M needs llama-quantize on a 16-bit file.
    direct = quant != "q4_k_m"
    intermediate = final if direct else out / f".{final.stem}-F16.gguf"
    with tempfile.TemporaryDirectory(dir=out, prefix=".convert-") as scratch:
        job.emit("phase", name="Converting to GGUF")
        cmd, env = llamacpp.convert_command(_converter_input(source, Path(scratch)), intermediate, quant if direct else "f16", mistral_format=found["kind"] == "mistral")
        _run(job, cmd, env, (0.0, 1.0 if direct else 0.6), WRITING)
    if job.cancel_requested:
        intermediate.unlink(missing_ok=True)
        return
    if not direct:
        job.emit("phase", name=f"Quantizing to {label}")
        try:
            _run(job, llamacpp.quantize_command(intermediate, final, quant), None, (0.6, 1.0), QUANTIZING)
        finally:
            intermediate.unlink(missing_ok=True)
        if job.cancel_requested:
            final.unlink(missing_ok=True)
            return
    entry = library.add_entry(str(final), name=f"{name} {label}")
    job.result.update(gguf=str(final), name=entry["name"])


@router.post("")
def start(req: ConvertRequest):
    if req.quant not in QUANTS:
        raise HTTPException(400, f"Choose one of {', '.join(QUANTS.values())}.")
    found = library.inspect(req.path)
    if not found.get("convertible"):
        raise HTTPException(400, _why_not(found))
    out = home() / "converted" / _safe(found["name"])
    for job in registry.jobs.values():
        if job.kind == "convert" and job.status == "running" and job.result.get("source") == found["path"]:
            raise HTTPException(409, f"{found['name']} is already being converted.")
    need = _f16_size(found) * (1 if req.quant == "f16" else 1.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if need > free:
        raise HTTPException(400, f"Converting needs about {need / 1e9:.0f} GB free on the disk Temper keeps its files on; it has {free / 1e9:.0f} GB.")
    job = registry.create("convert", f"Convert {found['name']} to {QUANTS[req.quant]} GGUF")
    job.progress = 0.0
    job.result.update(source=found["path"], output_dir=str(out))
    registry.run_thread(job, lambda j: _convert(j, found, req.quant, out))
    return job.summary()
