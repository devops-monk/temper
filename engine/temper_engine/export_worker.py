"""Export worker process: `python -m temper_engine.export_worker <export.json>`.

GGUF quants can be calibrated with an importance matrix (llama-imatrix) measured on the run's own
training data, so the weights that matter most for what the model was taught keep the most
precision. That's what makes the IQ quants usable, and it helps the low-bit K quants too.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from . import llamacpp
from .jobs import emit_from_worker as emit
from .llamacpp import run_streamed


def export_gpu(cfg: dict, out: Path) -> None:
    from unsloth import FastLanguageModel

    emit("phase", name="Loading adapter")
    model, tokenizer = FastLanguageModel.from_pretrained(cfg["adapter"], load_in_4bit="merged_4bit" in cfg["formats"])
    steps = len(cfg["formats"])
    for i, fmt in enumerate(cfg["formats"]):
        emit("phase", name=f"Saving {fmt}")
        if fmt == "gguf" and cfg.get("imatrix"):
            # Calibrated quants come from llama.cpp's own tools, from the merged 16-bit weights.
            merged = out / "merged_for_gguf"
            model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
            export_gguf(merged, out / "gguf", cfg)
            shutil.rmtree(merged)
        elif fmt == "gguf":
            model.save_pretrained_gguf(str(out / "gguf"), tokenizer, quantization_method=cfg["gguf_quants"])
        else:
            model.save_pretrained_merged(str(out / fmt), tokenizer, save_method=fmt)
        emit("progress", value=(i + 1) / steps)


def export_mlx(cfg: dict, out: Path) -> None:
    unsupported = set(cfg["formats"]) - {"lora", "merged_16bit", "gguf", "mlx"}
    if unsupported:
        raise RuntimeError(f"Not available on Apple Silicon yet: {', '.join(sorted(unsupported))}")
    steps = ["lora" in cfg["formats"], "mlx" in cfg["formats"], "merged_16bit" in cfg["formats"] or "gguf" in cfg["formats"], "gguf" in cfg["formats"]]
    done = 0

    def advance() -> None:
        nonlocal done
        done += 1
        emit("progress", value=done / max(sum(steps), 1))

    if "lora" in cfg["formats"]:
        shutil.copytree(cfg["adapter"], out / "lora")
        advance()
    if "mlx" in cfg["formats"]:
        emit("phase", name="Merging into an MLX model")
        # Without --dequantize the merged model keeps the base model's quantization (often 4-bit): small, and what
        # mlx-lm, LM Studio and Temper's MLX engine run.
        run_streamed([sys.executable, "-m", "mlx_lm", "fuse", "--model", cfg["base_model"], "--adapter-path", cfg["adapter"], "--save-path", str(out / "mlx")])
        advance()
    if "merged_16bit" in cfg["formats"] or "gguf" in cfg["formats"]:
        merged = out / "merged_16bit"
        emit("phase", name="Merging the adapter into the base model")
        # --dequantize writes 16-bit weights even when the base model is 4-bit, which GGUF conversion needs.
        run_streamed([sys.executable, "-m", "mlx_lm", "fuse", "--model", cfg["base_model"], "--adapter-path", cfg["adapter"],
                      "--save-path", str(merged), "--dequantize"])
        advance()
        if "gguf" in cfg["formats"]:
            export_gguf(merged, out / "gguf", cfg)
            advance()
            if "merged_16bit" not in cfg["formats"]:
                shutil.rmtree(merged)  # only needed as the converter's input


def export_gguf(model_dir: Path, dest: Path, cfg: dict) -> None:
    """Hugging Face folder → GGUF with llama.cpp's converter, then llama-quantize for each quant.

    mlx-lm's own GGUF export handles only Llama, Mistral and Mixtral; llama.cpp's converter covers
    nearly every architecture it can run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    name = cfg.get("run_id") or "model"
    downgrade_tokenizer_config(model_dir)
    emit("phase", name="Installing llama.cpp")
    llamacpp.install(lambda message, _p: print(message, flush=True) if not message.startswith(("llama-", "cudart")) else None)
    emit("phase", name="Converting to GGUF")
    f16 = dest / f"{name}-F16.gguf"
    cmd, env = llamacpp.convert_command(model_dir, f16)
    run_streamed(cmd, env)
    quants = [q for q in cfg["gguf_quants"] if q != "f16"]
    matrix = None
    if cfg.get("imatrix") and quants:
        emit("phase", name="Measuring which weights matter most")
        text = dest / "calibration.txt"
        text.write_text(calibration_text(cfg, model_dir), encoding="utf-8")
        matrix = dest / "imatrix.gguf"
        run_streamed(llamacpp.imatrix_command(f16, text, matrix))
        text.unlink()
    for quant in quants:
        emit("phase", name=f"Quantizing to {quant.upper()}")
        label = f"imatrix-{quant.upper()}" if matrix else quant.upper()
        run_streamed(llamacpp.quantize_command(f16, dest / f"{name}-{label}.gguf", quant, matrix))
    if "f16" not in cfg["gguf_quants"]:
        f16.unlink()


CALIBRATION_CHARS = 200_000  # about 50k tokens: enough for a stable importance matrix, a minute or two to measure


def calibration_text(cfg: dict, model_dir: Path) -> str:
    """The run's training rows as the model sees them (its chat template), for llama-imatrix to measure on."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    run = Path(cfg["adapter"]).parent
    rows_file = run / "data" / "train.jsonl"  # MLX runs keep their prepared rows
    if rows_file.is_file():
        rows = [json.loads(line) for line in rows_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        from .train_worker import conversations, load_rows

        run_cfg = json.loads((run / "config.json").read_text())
        rows = [{"messages": c} if isinstance(c, list) else {"text": c} for c in conversations(run_cfg, load_rows(run_cfg))[1]]
    parts, size = [], 0
    for row in rows:
        try:
            text = row["text"] if "text" in row else tok.apply_chat_template(row["messages"], tokenize=False)
        except Exception:  # noqa: BLE001 — a row the template can't render is just left out
            continue
        parts.append(text)
        size += len(text)
        if size >= CALIBRATION_CHARS:
            break
    if not parts:
        raise RuntimeError("The run's training data couldn't be read for calibration. Export without the importance matrix.")
    return "\n\n".join(parts)


def downgrade_tokenizer_config(model_dir: Path) -> None:
    """Rewrite a transformers 5 tokenizer config in place so the converter can read it (see llamacpp)."""
    path = model_dir / "tokenizer_config.json"
    if not path.is_file():
        return
    config = llamacpp.downgraded_tokenizer_config(json.loads(path.read_text(encoding="utf-8")))
    if config is not None:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    out = Path(cfg["output_dir"])
    try:
        (export_mlx if cfg["backend"] == "mlx" else export_gpu)(cfg, out)
        emit("result", output_dir=str(out), gguf=sorted(str(p) for p in out.rglob("*.gguf")))
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
