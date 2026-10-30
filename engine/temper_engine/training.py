"""Fine-tuning API: start runs, list past runs. The heavy lifting happens in train_worker."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home, sources
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

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + _slug(cfg.name or cfg.source_id or cfg.base_model)
    out = runs_dir() / run_id
    out.mkdir(parents=True)
    cfg.backend, cfg.output_dir = trainer, str(out)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))

    job = registry.create("train", cfg.name or f"{(cfg.source_id or cfg.base_model).split('/')[-1]} on {cfg.dataset.split('/')[-1]}")
    job.result["run_id"] = run_id
    job.progress = 0.0
    registry.run_worker(job, "temper_engine.train_worker", str(out / "config.json"))
    return job.summary()


@router.get("/runs")
def list_runs():
    runs = []
    for d in sorted(runs_dir().iterdir(), reverse=True):
        cfg_file = d / "config.json"
        if not cfg_file.is_file():
            continue
        cfg = json.loads(cfg_file.read_text())
        runs.append({
            "id": d.name,
            "base_model": cfg.get("source_id") or cfg.get("base_model"),
            "dataset": cfg.get("dataset"),
            "backend": cfg.get("backend"),
            "adapter": str(d / "adapter") if (d / "adapter").is_dir() else None,
            "path": str(d),
        })
    return runs
