"""Training worker process: `python -m temper_engine.train_worker <config.json>`.

Prints `@@{json}` lines (see jobs.emit_from_worker) that the engine turns into live UI events, and
keeps the same events in the run's events.jsonl, so a run's metrics, dataset and outcome outlast the
engine (training history, comparing runs). status.json says how the run ended.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, chat_render, formats
from .jobs import emit_from_worker

RECORDED = {"phase", "dataset", "provenance", "metrics", "eval", "sample", "best", "result", "error", "checkpoint"}
PATIENCE = 3  # evaluations without a better validation loss before stopping early, when asked
SAMPLES = 2  # held-out prompts answered at each evaluation
RUN: Path | None = None  # the run's folder, once main() knows it


def emit(event: str, **data) -> None:
    emit_from_worker(event, **data)
    if RUN is not None and event in RECORDED:
        with open(RUN / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "t": time.time(), **data}, default=str) + "\n")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(cfg: dict, data: Path | None = None, texts: list[str] | None = None) -> None:
    """Where the run's model and data came from, exactly: the base model's revision and the training data's hash."""
    import hashlib

    from huggingface_hub import try_to_load_from_cache

    base = cfg.get("source_id") or cfg["base_model"]
    found = try_to_load_from_cache(cfg["base_model"], "config.json") if not Path(cfg["base_model"]).exists() else None
    revision = re.search(r"snapshots[/\\]([0-9a-f]{40})", found).group(1) if isinstance(found, str) and "snapshots" in found else None
    digest = sha256(data) if data else hashlib.sha256("\n".join(texts or []).encode()).hexdigest()
    libraries = {}
    for name in ("mlx_lm", "trl", "transformers", "peft", "torch"):
        try:
            libraries[name] = __import__(name).__version__
        except (ImportError, AttributeError):
            pass
    emit("provenance", base_model=base, base_revision=revision, dataset=cfg["dataset"], dataset_split=cfg.get("dataset_split"), dataset_sha256=digest, temper=__version__, libraries=libraries)

# ---------------------------------------------------------------- dataset helpers

def load_rows(cfg: dict, source: str | None = None) -> list[dict]:
    from datasets import load_dataset

    src = source or cfg["dataset"]
    path = Path(src).expanduser()
    if path.exists():
        kind = {".csv": "csv", ".parquet": "parquet"}.get(path.suffix.lower(), "json")
        ds = load_dataset(kind, data_files=str(path), split="train")
    else:
        ds = load_dataset(src, cfg.get("dataset_config") or None, split=cfg.get("dataset_split") or "train")
    if cfg.get("max_samples") and source is None:
        ds = ds.shuffle(seed=cfg["seed"]).select(range(min(cfg["max_samples"], len(ds))))
    return list(ds)


def detect_format(row: dict) -> str:
    fmt = formats.detect_format(list(row))
    if fmt is None:
        raise ValueError(f"Could not detect the dataset format from columns {sorted(row)}. Prepare it on the Datasets page first.")
    return fmt


def to_messages(row: dict, fmt: str) -> list[dict] | str:
    return formats.to_messages(row, fmt)


def conversations(cfg: dict, rows: list[dict]) -> tuple[str, list]:
    """Normalise rows, skipping any that can't be used (e.g. no assistant reply)."""
    fmt = cfg["dataset_format"] if cfg["dataset_format"] != "auto" else detect_format(rows[0])
    out = []
    for row in rows:
        try:
            out.append(to_messages(row, fmt))
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    if not out:
        raise ValueError(f"No usable rows in {fmt} format.")
    return fmt, out


# ---------------------------------------------------------------- PyTorch GPUs (CUDA / ROCm / XPU)


def train_gpu(cfg: dict, out: Path) -> None:
    emit("phase", name="Loading model")
    from unsloth import FastLanguageModel  # must be imported before transformers/trl

    import inspect

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    full = cfg["method"] == "full"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["method"] == "qlora",
        full_finetuning=full,
    )
    if not full:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] + (["embed_tokens", "lm_head"] if cfg.get("train_embeddings") else []),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg["seed"],
        )

    emit("phase", name="Preparing dataset")
    def chat_text(conv: list[dict] | str) -> str:
        if isinstance(conv, str):
            return conv
        try:
            return "".join(chat_render.render(tokenizer, conv))
        except ValueError:  # doesn't end with an assistant reply
            return tokenizer.apply_chat_template(conv, tokenize=False)

    render = lambda convs: [chat_text(c) for c in convs]  # noqa: E731
    fmt, convs = conversations(cfg, load_rows(cfg))
    texts = render(convs)
    dataset = Dataset.from_dict({"text": texts})
    eval_dataset, eval_convs = None, []
    if cfg.get("eval_dataset"):
        _, eval_convs = conversations(cfg, load_rows(cfg, cfg["eval_dataset"]))
        eval_dataset = Dataset.from_dict({"text": render(eval_convs)})
    emit("dataset", rows=len(texts), format=fmt, sample=texts[0][:1500], eval_rows=len(eval_dataset) if eval_dataset else 0)
    provenance(cfg, texts=texts)

    class Reporter(TrainerCallback):
        def __init__(self) -> None:
            self.started = time.time()

        def on_train_begin(self, args, state, control, **kw):
            emit("phase", name="Training", total_steps=state.max_steps)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "eval_loss" in logs:
                emit("eval", step=state.global_step, val_loss=logs["eval_loss"])
            if not logs or "loss" not in logs:
                return
            elapsed = time.time() - self.started
            eta = elapsed / max(state.global_step, 1) * (state.max_steps - state.global_step)
            emit(
                "metrics",
                step=state.global_step,
                total_steps=state.max_steps,
                loss=logs.get("loss"),
                lr=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
                epoch=logs.get("epoch"),
                eta=eta,
            )
            emit("progress", value=state.global_step / max(state.max_steps, 1))

    prompts = sample_prompts(eval_convs) if cfg.get("sample_replies", True) else []

    class Samples(TrainerCallback):
        """Answers a few held-out prompts at each evaluation."""

        def on_evaluate(self, args, state, control, model=None, **kw):
            if not prompts or model is None:
                return
            import torch

            was = model.training
            model.eval()
            try:
                for p in prompts:
                    text = tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
                    ids = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**ids, max_new_tokens=160, do_sample=False)
                    reply = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                    emit("sample", step=state.global_step, prompt=str(p["messages"][-1].get("content"))[:600], reply=reply.strip(), expected=p["expected"])
            finally:
                model.train(was)

    args = dict(
        output_dir=str(out / "checkpoints"),
        dataset_text_field="text",
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"] or -1,
        learning_rate=cfg["learning_rate"],
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=cfg["seed"],
        report_to="none",
        save_strategy="steps",
        save_steps=save_every(cfg, cfg["max_steps"] or 500),
        save_total_limit=10,
    )
    # TRL renamed max_seq_length to max_length; pass whichever this version accepts.
    params = inspect.signature(SFTConfig.__init__).parameters
    args["max_length" if "max_length" in params else "max_seq_length"] = cfg["max_seq_length"]
    if eval_dataset is not None:
        # transformers renamed evaluation_strategy to eval_strategy.
        args["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
        args["eval_steps"] = eval_every(cfg["max_steps"] or 100)
    callbacks = [Reporter(), Samples()]
    if eval_dataset is not None and cfg.get("early_stopping"):
        from transformers import EarlyStoppingCallback

        # Early stopping keeps the best checkpoint, so one is saved at every evaluation.
        args.update(load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, save_steps=args["eval_steps"])
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=PATIENCE))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**args),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from") or None)
    state = trainer.state
    if cfg.get("early_stopping") and state.best_model_checkpoint:
        best = int(state.best_model_checkpoint.rstrip("/\\").rsplit("-", 1)[-1])
        emit("best", step=best, val_loss=state.best_metric, stopped_at=state.global_step if state.global_step < state.max_steps else None)

    emit("phase", name="Saving")
    target = out / ("model" if full else "adapter")
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    emit("result", adapter=str(target), base_model=cfg["base_model"])


# ---------------------------------------------------------------- MLX (Apple Silicon)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# mlx-lm >= 0.30 prints a table: "  12    1.205 ▲    133    9.1k" and "   1    val 1.623   59.44s".
MLX_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▲▼]?\s+([\d.]+)\s+[\d.]+k?\s*$")
MLX_VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")
# Older mlx-lm: "Iter 10: Train loss 2.345, Learning Rate 1.000e-05, ..., Tokens/sec 512.3"
MLX_ITER = re.compile(r"Iter (\d+): Train loss ([\d.]+)(?:, Learning Rate ([\deE.+-]+))?(?:.*Tokens/sec ([\d.]+))?")
MLX_ITER_VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+)")


def parse_mlx_line(line: str) -> dict | None:
    """Turns one line of `mlx_lm lora` output into a metrics dict, or None if it isn't one."""
    line = ANSI.sub("", line)
    if m := MLX_VAL.match(line) or MLX_ITER_VAL.search(line):
        return {"step": int(m.group(1)), "val_loss": float(m.group(2))}
    if m := MLX_ROW.match(line):
        return {"step": int(m.group(1)), "loss": float(m.group(2)), "tokens_per_sec": float(m.group(3))}
    if m := MLX_ITER.search(line):
        out = {"step": int(m.group(1)), "loss": float(m.group(2))}
        if m.group(3):
            out["lr"] = float(m.group(3))
        if m.group(4):
            out["tokens_per_sec"] = float(m.group(4))
        return out
    return None


def sample_prompts(convs: list) -> list[dict]:
    """A few held-out conversations as prompts (everything before the last reply), and the reply the data expects."""
    out = []
    for c in convs:
        if isinstance(c, list) and len(c) > 1 and c[-1].get("role") == "assistant" and not any(m.get("tool_calls") for m in c):
            out.append({"messages": c[:-1], "expected": str(c[-1].get("content") or "")[:600]})
        if len(out) == SAMPLES:
            break
    return out


def eval_every(steps: int) -> int:
    """Steps between evaluations: about ten over the run."""
    return max(1, steps // 10)


def save_every(cfg: dict, steps: int) -> int:
    """Steps between checkpoints: as set, or five over the run (at least every 10 steps)."""
    return cfg.get("save_every") or max(10, steps // 5)


def train_mlx(cfg: dict, out: Path) -> None:
    emit("phase", name="Preparing dataset")
    fmt, convs = conversations(cfg, load_rows(cfg))
    records = [formats.record(c) for c in convs]
    random.Random(cfg["seed"]).shuffle(records)
    if cfg.get("eval_dataset"):
        valid = [formats.record(c) for c in conversations(cfg, load_rows(cfg, cfg["eval_dataset"]))[1]]
        n_valid = 0
    else:
        n_valid = max(1, len(records) // 20)  # hold out 5% to report validation loss
        valid = records[:n_valid]
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    for name, chunk in (("valid", valid), ("train", records[n_valid:] or records)):
        with open(data_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk)
    emit("dataset", rows=len(records) - n_valid, format=fmt, sample=json.dumps(records[0], ensure_ascii=False)[:1500], eval_rows=len(valid))
    provenance(cfg, data=data_dir / "train.jsonl")
    prompts = sample_prompts([r.get("messages") for r in valid]) if cfg.get("sample_replies", True) else []
    if prompts:
        (out / "samples.json").write_text(json.dumps(prompts, ensure_ascii=False))

    steps_per_epoch = max(1, (len(records) - n_valid) // cfg["batch_size"])
    iters = cfg["max_steps"] or int(steps_per_epoch * cfg["epochs"])
    adapter = out / "adapter"
    # mlx-lm takes LoRA shape only from a config file (JSON is valid YAML). Its `scale` multiplies the
    # update directly; alpha / rank is the standard LoRA scaling, matching the GPU path.
    lora_cfg = out / "lora_config.yaml"
    lora_cfg.write_text(json.dumps({
        "lora_parameters": {"rank": cfg["lora_r"], "scale": cfg["lora_alpha"] / cfg["lora_r"], "dropout": cfg["lora_dropout"]},
    }))
    cmd = [
        # mlx-lm's trainer, with chat rows rendered as the model sees them at inference.
        sys.executable, "-m", "temper_engine.mlx_lora",
        "--model", cfg["base_model"],
        "--train",
        "--config", str(lora_cfg),
        "--data", str(data_dir),
        "--iters", str(iters),
        "--batch-size", str(cfg["batch_size"]),
        "--grad-accumulation-steps", str(cfg["grad_accum"]),
        # Recompute activations instead of storing them: much less unified memory for a little speed.
        "--grad-checkpoint",
        "--learning-rate", str(cfg["learning_rate"]),
        "--max-seq-length", str(cfg["max_seq_length"]),
        "--adapter-path", str(adapter),
        "--steps-per-report", "1",
        "--save-every", str(save_every(cfg, iters)),
        "--steps-per-eval", str(eval_every(iters)),
        "--seed", str(cfg["seed"]),
    ]
    if cfg.get("resume_from"):
        cmd += ["--resume-adapter-file", str(cfg["resume_from"])]
    if fmt != "text":
        cmd.append("--mask-prompt")  # loss on the replies only, not on the prompts
    if cfg["method"] == "full":
        cmd += ["--fine-tune-type", "full", "--num-layers", "-1"]  # every layer, not mlx-lm's default last 16
    env = dict(os.environ)
    if cfg.get("train_embeddings"):
        env["TEMPER_TRAIN_EMBEDDINGS"] = "1"
    if prompts:
        env["TEMPER_SAMPLES"] = str(out / "samples.json")
    if cfg.get("early_stopping"):
        env["TEMPER_EARLY_STOP"] = str(PATIENCE)
    emit("phase", name="Training", total_steps=iters)
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    assert proc.stdout
    last = ""
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("TEMPER {"):  # samples and early stopping, from mlx_lora
            try:
                data = json.loads(line[7:])
                emit(data.pop("event"), **data)
                continue
            except (ValueError, KeyError):
                pass
        print(line, flush=True)
        last = line.strip() or last
        metrics = parse_mlx_line(line)
        if metrics and "loss" in metrics:
            step = metrics["step"]
            elapsed = time.time() - started
            emit("metrics", total_steps=iters, eta=elapsed / step * (iters - step), **metrics)
            emit("progress", value=step / iters)
        elif metrics:
            emit("eval", **metrics)
    if proc.wait() != 0:
        raise RuntimeError(f"mlx_lm lora failed: {last[:300]}")
    emit("result", adapter=str(adapter), base_model=cfg["base_model"])


def main() -> None:
    global RUN
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    out = RUN = cfg_path.parent
    started = time.time()

    def ended(status: str, error: str | None = None) -> None:
        (out / "status.json").write_text(json.dumps({"status": status, "error": error, "started": started, "finished": time.time()}))

    try:
        (train_mlx if cfg["backend"] == "mlx" else train_gpu)(cfg, out)
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        ended("failed", f"{type(exc).__name__}: {exc}")
        raise
    ended("done")


if __name__ == "__main__":
    main()
