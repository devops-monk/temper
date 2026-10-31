"""Fine-tuning API: start runs, and keep their history. The heavy lifting happens in train_worker.

Each run is a folder in runs/: config.json (what was asked), events.jsonl (metrics, evaluation,
dataset and provenance as they happened), status.json (how it ended), and adapter/ with the
result and its checkpoints. From these a run can be looked back on, compared with others, run again,
resumed from a checkpoint, or chatted with at any checkpoint.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import chat_render, home, preference, sources
from .hardware import probe
from .jobs import registry

router = APIRouter(prefix="/api/train", tags=["training"])


def cached_snapshot(repo_id: str) -> str:
    """The cached copy of a full model. With ModelScope as the source, workers can't fetch from Hugging Face."""
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(repo_id, "config.json")
    if isinstance(config, str):
        folder = Path(config).parent
        if any(folder.glob("*.safetensors")) or any(folder.glob("*.bin")):
            return str(folder)
    raise HTTPException(409, f"Download {repo_id} from ModelScope first: open it under Models → Find models and choose Download all.")


class TrainConfig(BaseModel):
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    # The repo id when base_model was swapped for its cached folder (ModelScope).
    source_id: str | None = None
    dataset: str = "mlabonne/FineTome-100k"
    dataset_split: str = "train"
    # Hugging Face subset, for datasets that have several (e.g. openai/gsm8k "main").
    dataset_config: str | None = None
    # auto: detect messages / conversations / instruction+output / text columns.
    dataset_format: Literal["auto", "messages", "sharegpt", "alpaca", "text"] = "auto"
    max_samples: int | None = Field(default=2000, ge=1)
    method: Literal["lora", "qlora", "full"] = "qlora"
    # sft: learn to write the replies. dpo, orpo, simpo: learn which of two replies is better; kto: from
    # single replies marked good or bad (preference.py).
    objective: Literal["sft", "dpo", "orpo", "simpo", "kto"] = "sft"
    pref_beta: float | None = Field(default=None, gt=0)  # the objective's beta; None for its usual value
    # Continued pretraining: train the token embeddings and output head too, for new vocabulary or a new language.
    train_embeddings: bool = False
    max_seq_length: int = Field(default=2048, ge=128, le=262144)
    lora_r: int = Field(default=16, ge=1, le=512)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: float = Field(default=1, gt=0)
    max_steps: int = Field(default=60, ge=0, description="0 trains for the full epoch count")
    batch_size: int = Field(default=2, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    warmup_steps: int = Field(default=5, ge=0)
    seed: int = 3407
    name: str | None = None
    # Held-out rows for validation loss. A prepared `x.jsonl` with a sibling `x.eval.jsonl` uses it automatically.
    eval_dataset: str | None = None
    # Stop once validation loss hasn't improved for a few evaluations, keeping the best adapter.
    early_stopping: bool = False
    # Answer a couple of held-out prompts at each evaluation, to see replies change as it trains.
    sample_replies: bool = True
    save_every: int = Field(default=0, ge=0, description="steps between checkpoints; 0 saves five over the run")
    resume_from: str | None = None  # a checkpoint to carry on from
    start_step: int = 0  # the step of that checkpoint, so the run's steps line up with the run it carries on
    # Set by the engine.
    backend: str | None = None
    output_dir: str | None = None


def quantized(model: str) -> bool:
    """Whether an MLX model's weights are quantized (4- or 8-bit), as far as its config.json on this computer says."""
    from huggingface_hub import try_to_load_from_cache

    path = Path(model).expanduser()
    found = path / "config.json" if path.is_dir() else try_to_load_from_cache(model, "config.json")
    try:
        config = json.loads(Path(found).read_text()) if isinstance(found, (str, Path)) and Path(found).is_file() else {}
    except (OSError, ValueError):
        config = {}
    return bool(config.get("quantization") or config.get("quantization_config")) or bool(re.search(r"[-_](?:[2-8]bit|q[2-8])\b", model, re.I))


def runs_dir() -> Path:
    path = home() / "runs"
    path.mkdir(exist_ok=True)
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.split("/")[-1]).strip("-").lower()[:40]


@router.post("")
def start(cfg: TrainConfig):
    info = probe()
    trainer = info["capabilities"]["trainer"]
    if trainer is None:
        if info["backend"] in ("metal", "cuda", "rocm"):
            hint = "Add training support in Settings."
        else:
            hint = "Training needs an NVIDIA, AMD or Intel GPU, or an Apple Silicon Mac."
        raise HTTPException(409, f"No trainer is available on this machine. {hint}")
    if trainer == "mlx" and (cfg.method == "full" or cfg.train_embeddings) and quantized(cfg.base_model):
        what = "Full fine-tuning" if cfg.method == "full" else "Training the embeddings"
        raise HTTPException(400, f"{what} changes every weight, which a 4- or 8-bit model can't store. Choose the 16-bit version of the model (without 4bit or 8bit in its name), or LoRA.")

    if sources.modelscope() and not Path(cfg.base_model).expanduser().exists():
        cfg.source_id, cfg.base_model = cfg.base_model, cached_snapshot(cfg.base_model)

    dataset = Path(cfg.dataset).expanduser()
    if cfg.eval_dataset is None and dataset.suffix == ".jsonl" and dataset.with_suffix(".eval.jsonl").is_file():
        cfg.eval_dataset = str(dataset.with_suffix(".eval.jsonl"))
    if cfg.objective != "sft" and (cfg.method == "full" or cfg.train_embeddings):
        raise HTTPException(400, "Preference training trains LoRA adapters. Choose LoRA or QLoRA, without training the embeddings.")
    if cfg.early_stopping and trainer != "mlx" and not cfg.eval_dataset:
        # MLX holds out a slice of the data by itself; the GPU trainer evaluates only on a separate set.
        raise HTTPException(400, "Stopping early needs a held-out set to measure against. Prepare the dataset on the Datasets page (it sets one aside), or give an evaluation file.")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + _slug(cfg.name or cfg.source_id or cfg.base_model)
    # Two runs of the same name started in the same second get -2, -3 and so on.
    out, n = runs_dir() / run_id, 1
    while out.exists():
        n += 1
        out = runs_dir() / f"{run_id}-{n}"
    run_id = out.name
    out.mkdir(parents=True)
    cfg.backend, cfg.output_dir = trainer, str(out)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))

    job = registry.create("train", cfg.name or f"{(cfg.source_id or cfg.base_model).split('/')[-1]} on {cfg.dataset.split('/')[-1]}")
    job.result["run_id"] = run_id
    job.progress = 0.0
    registry.run_worker(job, "temper_engine.train_worker", str(out / "config.json"))
    return job.summary()


def _run_dir(run_id: str) -> Path:
    d = runs_dir() / run_id
    if not re.fullmatch(r"[\w.-]+", run_id) or not (d / "config.json").is_file():
        raise HTTPException(404, f"There's no training run {run_id}.")
    return d


def events(d: Path) -> list[dict]:
    try:
        lines = (d / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def checkpoints(d: Path) -> list[dict]:
    """The checkpoints a run saved, by step: MLX's numbered adapter files, or the GPU trainer's checkpoint folders."""
    found = []
    for f in (d / "adapter").glob("[0-9]*_adapters.safetensors"):
        found.append({"step": int(f.name.split("_")[0]), "path": str(f), "bytes": f.stat().st_size})
    for c in (d / "checkpoints").glob("checkpoint-*"):
        if c.is_dir() and c.name.split("-")[-1].isdigit():
            found.append({"step": int(c.name.split("-")[-1]), "path": str(c), "bytes": sum(f.stat().st_size for f in c.rglob("*") if f.is_file())})
    return sorted(found, key=lambda c: c["step"])


def _status(d: Path) -> str:
    running = any(j.kind == "train" and j.status == "running" and j.result.get("run_id") == d.name for j in registry.jobs.values())
    if running:
        return "running"
    try:
        return json.loads((d / "status.json").read_text())["status"]
    except (OSError, ValueError, KeyError):
        return "done" if (d / "adapter" / "adapters.safetensors").is_file() or (d / "adapter" / "adapter_config.json").is_file() else "stopped"


def summary(d: Path) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    evs = events(d)
    metrics = [e for e in evs if e["event"] == "metrics"]
    evals = [e for e in evs if e["event"] == "eval" and e.get("val_loss") is not None]
    prov = next((e for e in evs if e["event"] == "provenance"), {})
    return {
        "id": d.name,
        "name": cfg.get("name"),
        "base_model": cfg.get("source_id") or cfg.get("base_model"),
        "dataset": cfg.get("dataset"),
        "method": cfg.get("method"),
        "objective": cfg.get("objective") or "sft",
        "backend": cfg.get("backend"),
        "adapter": str(d / "adapter") if (d / "adapter").is_dir() else None,
        "path": str(d),
        "created": (d / "config.json").stat().st_mtime,
        "status": _status(d),
        "steps": metrics[-1]["step"] if metrics else 0,
        "total_steps": metrics[-1].get("total_steps") if metrics else None,
        "loss": metrics[-1].get("loss") if metrics else None,
        "val_loss": evals[-1]["val_loss"] if evals else None,
        "dataset_sha256": prov.get("dataset_sha256"),
        "checkpoints": len(checkpoints(d)),
    }


@router.get("/runs")
def list_runs():
    return [summary(d) for d in sorted(runs_dir().iterdir(), reverse=True) if (d / "config.json").is_file()]


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """A run in full: what it was asked, its metrics and evaluations step by step, where its data came from, its checkpoints."""
    d = _run_dir(run_id)
    evs = events(d)
    offset = json.loads((d / "config.json").read_text()).get("start_step") or 0
    return {
        **summary(d),
        "config": json.loads((d / "config.json").read_text()),
        "start_step": offset,
        "metrics": [{k: e.get(k) for k in ("loss", "lr", "tokens_per_sec", "grad_norm")} | {"step": (e.get("step") or 0) + offset} for e in evs if e["event"] == "metrics"],
        "evals": [{"step": (e.get("step") or 0) + offset, "val_loss": e.get("val_loss")} for e in evs if e["event"] == "eval"],
        "dataset_info": next(({k: v for k, v in e.items() if k not in ("event", "t", "sample")} for e in evs if e["event"] == "dataset"), None),
        "provenance": next(({k: v for k, v in e.items() if k not in ("event", "t")} for e in evs if e["event"] == "provenance"), None),
        "error": next((e.get("message") for e in evs if e["event"] == "error"), None),
        "best": next(({k: e.get(k) for k in ("step", "val_loss", "stopped_at")} for e in evs if e["event"] == "best"), None),
        "samples": [{k: e.get(k) for k in ("step", "prompt", "reply", "expected")} for e in evs if e["event"] == "sample"],
        "checkpoint_list": checkpoints(d),
    }


def _again(cfg: dict, **changes) -> TrainConfig:
    """A run's config, ready to start once more (with `changes`)."""
    keep = {k: v for k, v in cfg.items() if k not in ("backend", "output_dir")}
    if keep.get("source_id"):
        keep["base_model"], keep["source_id"] = keep["source_id"], None
    return TrainConfig(**{**keep, **changes})


@router.post("/runs/{run_id}/rerun")
def rerun(run_id: str):
    """Starts a new run with exactly the same settings."""
    cfg = json.loads((_run_dir(run_id) / "config.json").read_text())
    return start(_again(cfg, resume_from=None, start_step=0))


class Resume(BaseModel):
    step: int


def _checkpoint(d: Path, step: int) -> dict:
    found = next((c for c in checkpoints(d) if c["step"] == step), None)
    if not found:
        raise HTTPException(404, f"The run has no checkpoint at step {step}.")
    return found


@router.post("/runs/{run_id}/resume")
def resume(run_id: str, req: Resume):
    """Carries a run on from one of its checkpoints, as a new run, for the steps it had left."""
    d = _run_dir(run_id)
    cfg = json.loads((d / "config.json").read_text())
    point = _checkpoint(d, req.step)
    left = max((cfg.get("max_steps") or 0) - req.step, 1) if cfg.get("max_steps") else 0
    name = f"{cfg.get('name') or d.name} from step {req.step}"
    return start(_again(cfg, resume_from=point["path"], start_step=(cfg.get("start_step") or 0) + req.step, max_steps=left, name=name))


@router.post("/runs/{run_id}/checkpoints/{step}/adapter")
def checkpoint_adapter(run_id: str, step: int):
    """An adapter folder for one checkpoint, to chat with it: GPU checkpoints are one already; MLX's get the run's adapter config beside them."""
    d = _run_dir(run_id)
    point = _checkpoint(d, step)
    path = Path(point["path"])
    if path.is_dir():
        return {"adapter": str(path)}
    folder = d / "checkpoints" / f"step-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    config = d / "adapter" / "adapter_config.json"
    if not config.is_file():
        raise HTTPException(409, "The run hasn't written its adapter config yet; try again in a moment.")
    shutil.copy2(config, folder / "adapter_config.json")
    target = folder / "adapters.safetensors"
    if not target.exists():
        try:
            target.hardlink_to(path)
        except OSError:
            shutil.copy2(path, target)
    return {"adapter": str(folder)}


# --- Token preview --------------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _tokenizer(base_model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise HTTPException(409, "Seeing tokens needs training support. Add it in Settings.") from None
    try:
        return AutoTokenizer.from_pretrained(base_model)
    except (OSError, ValueError) as e:
        raise HTTPException(422, f"Couldn't load {base_model}'s tokenizer: {str(e).splitlines()[0]}") from e


def tokens_for(tokenizer, conv: list[dict] | str) -> tuple[list[int], int]:
    """A row's token ids as training sees them, and how many at the start are the prompt (masked from the loss).

    The same split as the trainers: everything before the last reply is prompt; plain text has none.
    """
    if isinstance(conv, str):
        prompt, completion = "", conv
    else:
        try:
            prompt, completion = chat_render.render(tokenizer, conv)
        except ValueError:  # doesn't end with a reply: trained on in full
            prompt, completion = "", tokenizer.apply_chat_template(conv, tokenize=False)
    head = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    return head + tokenizer.encode(completion, add_special_tokens=False), len(head)


class TokenPreview(BaseModel):
    base_model: str
    dataset: str
    dataset_split: str = "train"
    dataset_config: str | None = None
    dataset_format: str = "auto"
    max_seq_length: int = 2048
    row: int = Field(default=0, ge=0, le=99)


@router.post("/preview")
def preview_tokens(req: TokenPreview):
    """One row of the dataset exactly as the model will see it: its tokens, which are masked, and where it's cut."""
    from . import datasets, formats

    rows, _ = datasets._rows(req.dataset, req.dataset_split, req.row + 2, req.dataset_config)
    if not rows:
        raise HTTPException(404, "The dataset has no rows.")
    row = rows[min(req.row, len(rows) - 1)]
    if preference.is_preference(row):
        fmt = "preference"  # shown with its chosen (or only) reply
    else:
        fmt = req.dataset_format if req.dataset_format != "auto" else formats.detect_format(list(row))
    if fmt is None:
        raise HTTPException(422, f"Couldn't tell the dataset's format from its columns ({', '.join(sorted(row))}). Prepare it on the Datasets page first.")
    try:
        if fmt == "preference":
            prompt, chosen, _ = preference.to_example(row) if preference.is_unpaired(row) else preference.to_pair(row)
            conv = prompt + [{"role": "assistant", "content": chosen}]
        else:
            conv = formats.to_messages(row, fmt)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise HTTPException(422, f"Row {req.row + 1} can't be used: {e}") from e
    tok = _tokenizer(req.base_model)
    if not isinstance(conv, str) and not tok.chat_template:
        raise HTTPException(422, f"{req.base_model} has no chat template, so it can't be trained on conversations. Choose an instruct model, or prepare the data as plain text.")
    ids, masked = tokens_for(tok, conv)
    shown = ids[:4000]
    return {
        "tokens": [{"id": i, "text": tok.decode([i])} for i in shown],
        "count": len(ids),
        "masked": masked,
        "max_seq_length": req.max_seq_length,
        "format": fmt,
        "row": min(req.row, len(rows) - 1),
        "more": len(rows) > req.row + 1,  # whether there's a next row
    }
