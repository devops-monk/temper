"""Export worker process: `python -m temper_engine.export_worker <export.json>`."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from . import llamacpp
from .jobs import emit_from_worker as emit
from .llamacpp import run_streamed


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
    steps = ["lora" in cfg["formats"], "merged_16bit" in cfg["formats"] or "gguf" in cfg["formats"], "gguf" in cfg["formats"]]
    done = 0

    def advance() -> None:
        nonlocal done
        done += 1
        emit("progress", value=done / max(sum(steps), 1))

    if "lora" in cfg["formats"]:
        shutil.copytree(cfg["adapter"], out / "lora")
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
    for quant in [q for q in cfg["gguf_quants"] if q != "f16"]:
        emit("phase", name=f"Quantizing to {quant.upper()}")
        run_streamed(llamacpp.quantize_command(f16, dest / f"{name}-{quant.upper()}.gguf", quant))
    if "f16" not in cfg["gguf_quants"]:
        f16.unlink()


def downgrade_tokenizer_config(model_dir: Path) -> None:
    """Rewrite a transformers 5 tokenizer config so the converter's pinned transformers 4 can read it.

    v5 saves `extra_special_tokens` as a list of strings; v4 expects a dict there and keeps the list
    under `additional_special_tokens`.
    """
    path = model_dir / "tokenizer_config.json"
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return
    del config["extra_special_tokens"]
    config["additional_special_tokens"] = list(dict.fromkeys(config.get("additional_special_tokens", []) + extra))
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


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
