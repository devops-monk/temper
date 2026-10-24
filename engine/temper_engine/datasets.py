"""Datasets: preview, map columns to chat turns, and prepare (clean, dedupe, split) for training."""

from __future__ import annotations

import csv
import json
import re
import time
from itertools import islice
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import formats, home
from .formats import detect_format
from .jobs import registry

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

VIEWER = "https://datasets-server.huggingface.co"


def _local(path: Path, limit: int) -> list[dict]:
    suffix = path.suffix.lower()
    with open(path, encoding="utf-8") as f:
        if suffix == ".csv":
            return list(islice(csv.DictReader(f), limit))
        if suffix == ".jsonl":
            return [json.loads(line) for line in islice((l for l in f if l.strip()), limit)]
        if suffix == ".json":
            data = json.load(f)
            return data[:limit] if isinstance(data, list) else [data]
    raise HTTPException(400, "Supported local files: .csv, .json, .jsonl")


def _hub(dataset: str, split: str, limit: int, config: str | None = None) -> tuple[list[dict], dict]:
    """First rows of one split, plus which subset ("config") and split were used and which exist."""
    with httpx.Client(timeout=20) as client:
        splits = client.get(f"{VIEWER}/splits", params={"dataset": dataset})
        if splits.status_code != 200:
            raise HTTPException(404, f"Could not open dataset {dataset} on Hugging Face")
        options = splits.json().get("splits", [])
        configs = list(dict.fromkeys(s["config"] for s in options))
        in_config = [s for s in options if s["config"] == (config or (configs[0] if configs else None))]
        chosen = next((s for s in in_config if s["split"] == split), in_config[0] if in_config else None)
        if not chosen:
            raise HTTPException(404, f"{dataset} has no viewable splits")
        rows = client.get(f"{VIEWER}/first-rows", params={"dataset": dataset, "config": chosen["config"], "split": chosen["split"]})
        if rows.status_code != 200:
            raise HTTPException(502, f"The dataset viewer could not read {dataset}")
        info = {"config": chosen["config"], "configs": configs, "split": chosen["split"], "splits": [s["split"] for s in in_config]}
        return [r["row"] for r in rows.json().get("rows", [])][:limit], info


def _rows(source: str, split: str, limit: int, config: str | None) -> tuple[list[dict], dict]:
    path = Path(source).expanduser()
    return (_local(path, limit), {}) if path.is_file() else _hub(source, split, limit, config)


@router.get("/preview")
def preview(source: str, split: str = "train", limit: int = 50, config: str | None = None):
    path = Path(source).expanduser()
    rows, info = _rows(source, split, limit, config)
    columns = list(rows[0].keys()) if rows else []
    fmt = detect_format(columns)
    suggestion = None if fmt else formats.guess_mapping(columns)
    return {
        "source": source,
        "columns": columns,
        "rows": rows,
        "format": fmt,
        "local": path.is_file(),
        "suggested": {"format": suggestion[0], "mapping": suggestion[1]} if suggestion else None,
        **info,
    }


class Mapping(BaseModel):
    source: str
    split: str = "train"
    config: str | None = None
    format: str
    mapping: dict[str, str] = Field(default_factory=dict)
    system_prompt: str | None = None


def _check_mapping(req: Mapping) -> None:
    if req.format not in formats.FORMATS:
        raise HTTPException(400, f"Unknown format {req.format}")
    req.mapping = {k: v for k, v in req.mapping.items() if v}
    req.system_prompt = (req.system_prompt or "").strip() or None


@router.post("/render")
def render(req: Mapping):
    """The first rows as the model will see them, for a live preview while mapping columns."""
    _check_mapping(req)
    rows, _ = _rows(req.source, req.split, 20, req.config)
    out = []
    for row in rows:
        try:
            out.append({"conversation": formats.to_messages(row, req.format, req.mapping, req.system_prompt)})
        except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
            out.append({"error": str(exc) or type(exc).__name__})
    return {"rows": out, "usable": sum(1 for r in out if "conversation" in r)}


class Prepare(Mapping):
    dedupe: bool = True
    min_chars: int = Field(0, ge=0)
    max_chars: int = Field(0, ge=0)
    eval_fraction: float = Field(0.05, ge=0, le=0.5)
    shuffle: bool = True
    seed: int = 3407
    max_rows: int = Field(0, ge=0)
    name: str | None = None


@router.post("/prepare")
def prepare(req: Prepare):
    _check_mapping(req)
    stem = re.sub(r"[^a-z0-9]+", "-", (req.name or Path(req.source).stem or req.source.split("/")[-1]).lower()).strip("-")[:40] or "dataset"
    out_dir = home() / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{stem}-prepared-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    cfg = {
        "source": req.source,
        "split": req.split,
        "config": req.config,
        "format": req.format,
        "mapping": req.mapping,
        "system_prompt": req.system_prompt,
        "output": str(output),
        "options": req.model_dump(include={"dedupe", "min_chars", "max_chars", "eval_fraction", "shuffle", "seed", "max_rows"}),
    }
    work = home() / "prepare"
    work.mkdir(exist_ok=True)
    cfg_path = work / f"{output.stem}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    job = registry.create("prepare", f"Prepare {req.source.split('/')[-1]}")
    job.result["output"] = str(output)
    registry.run_worker(job, "temper_engine.prepare_worker", str(cfg_path))
    return job.summary()
