"""Dataset preview: first rows of a Hugging Face dataset or a local CSV / JSON / JSONL file."""

from __future__ import annotations

import csv
import json
from itertools import islice
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

VIEWER = "https://datasets-server.huggingface.co"


def detect_format(columns: list[str]) -> str | None:
    cols = set(columns)
    if "messages" in cols:
        return "messages"
    if "conversations" in cols:
        return "sharegpt"
    if {"instruction", "output"} <= cols:
        return "alpaca"
    if "text" in cols:
        return "text"
    return None


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


def _hub(dataset: str, split: str, limit: int) -> list[dict]:
    with httpx.Client(timeout=20) as client:
        splits = client.get(f"{VIEWER}/splits", params={"dataset": dataset})
        if splits.status_code != 200:
            raise HTTPException(404, f"Could not open dataset {dataset} on Hugging Face")
        options = splits.json().get("splits", [])
        chosen = next((s for s in options if s["split"] == split), options[0] if options else None)
        if not chosen:
            raise HTTPException(404, f"{dataset} has no viewable splits")
        rows = client.get(f"{VIEWER}/first-rows", params={"dataset": dataset, "config": chosen["config"], "split": chosen["split"]})
        if rows.status_code != 200:
            raise HTTPException(502, f"The dataset viewer could not read {dataset}")
        return [r["row"] for r in rows.json().get("rows", [])][:limit]


@router.get("/preview")
def preview(source: str, split: str = "train", limit: int = 50):
    path = Path(source).expanduser()
    rows = _local(path, limit) if path.is_file() else _hub(source, split, limit)
    columns = list(rows[0].keys()) if rows else []
    return {"source": source, "columns": columns, "rows": rows, "format": detect_format(columns), "local": path.is_file()}
