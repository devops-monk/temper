"""Export worker process: `python -m temper_engine.export_worker <export.json>`."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .jobs import emit_from_worker as emit


def export_unsloth(cfg: dict, out: Path) -> None:
    from unsloth import FastLanguageModel

    emit("phase", name="Loading adapter")
    model, tokenizer = FastLanguageModel.from_pretrained(cfg["adapter"], load_in_4bit="merged_4bit" in cfg["formats"])
    steps = len(cfg["formats"])
    for i, fmt in enumerate(cfg["formats"]):
        emit("phase", name=f"Saving {fmt}")
        if fmt == "gguf":
            model.save_pretrained_gguf(str(out / "gguf"), tokenizer, quantization_method=cfg["gguf_quants"])
        else:
            model.save_pretrained_merged(str(out / fmt), tokenizer, save_method=fmt)
        emit("progress", value=(i + 1) / steps)


def export_mlx(cfg: dict, out: Path) -> None:
    unsupported = set(cfg["formats"]) - {"lora", "merged_16bit", "gguf"}
    if unsupported:
        raise RuntimeError(f"Not available on Apple Silicon yet: {', '.join(sorted(unsupported))}")
    if "lora" in cfg["formats"]:
        shutil.copytree(cfg["adapter"], out / "lora")
    wants_gguf = "gguf" in cfg["formats"]
    if "merged_16bit" in cfg["formats"] or wants_gguf:
        emit("phase", name="Fusing adapter into the base model")
        cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", cfg["base_model"], "--adapter-path", cfg["adapter"],
               "--save-path", str(out / "merged_16bit"), "--dequantize"]
        if wants_gguf:
            # MLX writes fp16 GGUF only; quantize with llama.cpp's llama-quantize afterwards.
            cmd += ["--export-gguf", "--gguf-path", str(out / "gguf" / "model-f16.gguf")]
        subprocess.run(cmd, check=True)
        if wants_gguf:
            quantize_gguf(out / "gguf" / "model-f16.gguf", [q for q in cfg["gguf_quants"] if q != "f16"])
    emit("progress", value=1.0)


def quantize_gguf(src: Path, quants: list[str]) -> None:
    tool = shutil.which("llama-quantize")
    if not tool:
        if quants:
            print("llama-quantize not found; kept the f16 GGUF only. Install llama.cpp to get smaller quants.", flush=True)
        return
    for q in quants:
        emit("phase", name=f"Quantizing {q}")
        subprocess.run([tool, str(src), str(src.with_name(f"model-{q}.gguf")), q.upper()], check=True)


def main() -> None:
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    out = Path(cfg["output_dir"])
    try:
        (export_mlx if cfg["backend"] == "mlx" else export_unsloth)(cfg, out)
        emit("result", output_dir=str(out), gguf=sorted(str(p) for p in out.rglob("*.gguf")))
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
