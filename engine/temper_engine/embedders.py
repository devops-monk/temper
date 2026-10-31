"""Embeddings: OpenAI's /v1/embeddings, and fine-tuning an embedding model on your own pairs.

/v1/embeddings answers with the embedding models Temper has: the GGUF ones knowledge bases use (run by
llama.cpp, knowledge.py), embedding models tuned here, and any sentence-transformers model downloaded from
Hugging Face (run by the warm worker in the classifier environment, classifiers.py). A request that names
none of them, such as OpenAI's own model names, gets the knowledge bases' embedding model (or, without
one, a general sentence-transformers model that's downloaded).

Tuning teaches an embedding model what "close" means for your data, from pairs: a question and the passage
that answers it, a search and the product it should find (embedder_worker.py). It's measured on held-out
pairs before and after, so you can see whether it helped. The pairs come from a CSV, JSON or JSONL file
with two columns, found by name (anchor/query/question… and positive/passage/answer…) or given. Tuned
models are folders under embedders/ in Temper's home.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import shutil
import struct
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import classifiers, home, knowledge
from .jobs import registry

router = APIRouter(tags=["embeddings"])

WORKER = Path(__file__).with_name("embedder_worker.py")
ANCHOR_COLUMNS = ("anchor", "query", "question", "prompt", "input", "sentence1", "text1", "search")
POSITIVE_COLUMNS = ("positive", "passage", "answer", "document", "context", "response", "output", "sentence2", "text2", "product")
MAX_PAIRS = 100_000
MAX_INPUTS = 2048


def folder() -> Path:
    path = home() / "embedders"
    path.mkdir(exist_ok=True)
    return path


def info(name: str) -> dict:
    path = folder() / name
    if not re.fullmatch(r"[a-z0-9-]+", name) or not (path / "embedder.json").is_file():
        raise HTTPException(404, f"There's no tuned embedding model called {name}.")
    return json.loads((path / "embedder.json").read_text()) | {"path": str(path / "model")}


def tuned() -> list[dict]:
    out = []
    for d in folder().iterdir():
        if (d / "embedder.json").is_file():
            try:
                out.append(info(d.name))
            except (HTTPException, ValueError, OSError):
                continue
    return sorted(out, key=lambda e: e["created"], reverse=True)


# --- Which model answers ---------------------------------------------------------------------------


def resolve(model: str | None) -> tuple[str, str, str]:
    """(kind, what to run, name to report): ("gguf", embedder id, ...) or ("st", folder, ...)."""
    if model:
        for eid, e in knowledge.EMBEDDERS.items():
            if model in (eid, e["repo"], e["name"]):
                return "gguf", eid, eid
        name = model.lower()
        if re.fullmatch(r"[a-z0-9-]+", name) and (folder() / name / "embedder.json").is_file():
            return "st", str(folder() / name / "model"), name
        found = classifiers.encoder_folder(model)
        if found:
            return "st", found, model
    eid = knowledge.embedder_id()
    if knowledge.embedder_path(eid):
        return "gguf", eid, eid
    fallback = next((e for e in knowledge.EMBEDDERS if knowledge.embedder_path(e)), None)
    if fallback:
        return "gguf", fallback, fallback
    general = next((e["id"] for e in classifiers.ENCODERS if classifiers.encoder_folder(e["id"])), None) if classifiers.python() else None
    if general:
        return "st", classifiers.encoder_folder(general), general
    raise HTTPException(404, "There's no embedding model on this computer. Download one in Settings → Knowledge bases, or name a sentence-transformers model you've downloaded.")


def embed(texts: list[str], model: str | None = None) -> tuple[list[list[float]], str]:
    kind, target, name = resolve(model)
    if kind == "gguf":
        try:
            return knowledge.embedder.embed(texts, target).tolist(), name
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
    if classifiers.python() is None:
        raise HTTPException(409, "Running sentence-transformers models needs the classifier environment. Install it on the Train page (Classifier).")
    try:
        return classifiers.worker.request({"embed": target, "texts": texts})["vectors"], name
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = "float"
    dimensions: int | None = Field(None, ge=1)


@router.post("/v1/embeddings")
def v1_embeddings(req: EmbeddingsRequest):
    """OpenAI's embeddings API. `dimensions` keeps the first N numbers of each vector, made unit length again."""
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts or len(texts) > MAX_INPUTS:
        raise HTTPException(400, f"Send between 1 and {MAX_INPUTS} inputs.")
    if not all(isinstance(t, str) for t in texts):
        raise HTTPException(400, "Inputs are text; token arrays aren't supported.")
    vectors, name = embed(texts, req.model)
    data = []
    for i, v in enumerate(vectors):
        if req.dimensions and req.dimensions < len(v):
            v = v[: req.dimensions]
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            v = [x / norm for x in v]
        value = base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode() if req.encoding_format == "base64" else v
        data.append({"object": "embedding", "index": i, "embedding": value})
    tokens = sum(max(1, len(t) // 4) for t in texts)  # an estimate: the servers don't count
    return {"object": "list", "data": data, "model": name, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


# --- Tuning ------------------------------------------------------------------------------------------


def pairs(path: Path, anchor_column: str | None, positive_column: str | None) -> tuple[list[list[str]], str, str]:
    rows = classifiers.read_rows(path)
    if not rows:
        raise HTTPException(400, f"{path.name} has no rows.")
    a_col = classifiers._column(rows, anchor_column, ANCHOR_COLUMNS, "anchors (the questions or searches)")
    p_col = classifiers._column(rows, positive_column, POSITIVE_COLUMNS, "positives (what each should find)")
    if a_col == p_col:
        raise HTTPException(400, "The anchor and positive columns have to be different.")
    out = [[str(r[a_col]).strip(), str(r[p_col]).strip()] for r in rows if r.get(a_col) and r.get(p_col) and str(r[a_col]).strip() and str(r[p_col]).strip()]
    return out[:MAX_PAIRS], a_col, p_col


class TuneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    dataset: str
    anchor_column: str | None = None
    positive_column: str | None = None
    base: str = classifiers.ENCODERS[0]["id"]
    epochs: int = Field(1, ge=1, le=20)
    eval_share: float = Field(0.1, ge=0, le=0.5)


class Peek(BaseModel):
    dataset: str
    anchor_column: str | None = None
    positive_column: str | None = None


@router.get("/api/embedders")
def embedders_status():
    return {"installed": classifiers.python() is not None, "bases": classifiers.status()["encoders"], "tuned": tuned()}


@router.post("/api/embedders/peek")
def peek(req: Peek):
    path = Path(req.dataset).expanduser()
    rows = classifiers.read_rows(path)
    columns = list(rows[0]) if rows else []
    try:
        found, a_col, p_col = pairs(path, req.anchor_column, req.positive_column)
    except HTTPException as e:
        return {"columns": columns, "rows": len(rows), "problem": e.detail}
    return {"columns": columns, "rows": len(rows), "anchor_column": a_col, "positive_column": p_col, "pairs": len(found), "sample": found[:3],
            "problem": None if len(found) >= 10 else "Give at least 10 pairs."}


@router.post("/api/embedders")
def tune(req: TuneRequest):
    if classifiers.python() is None:
        raise HTTPException(409, "Tuning embedding models needs the classifier environment. Install it on the Train page (Classifier).")
    base = classifiers.encoder_folder(req.base)
    if base is None:
        raise HTTPException(409, f"Download the base model first ({req.base}).")
    name = classifiers._slug(req.name)
    out = folder() / name
    if (out / "embedder.json").is_file():
        raise HTTPException(409, f"There's already a tuned model called {name}. Delete it or pick another name.")
    path = Path(req.dataset).expanduser()
    found, _, _ = pairs(path, req.anchor_column, req.positive_column)
    if len(found) < 10:
        raise HTTPException(400, f"Give at least 10 pairs; {path.name} has {len(found)}.")
    rng = random.Random(42)
    rng.shuffle(found)
    n = min(len(found) - 8, round(len(found) * req.eval_share))
    held, train = found[:n], found[n:]
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    plan = {"name": name, "base": base, "base_id": req.base, "dataset": str(path), "created": time.time(), "epochs": req.epochs, "out": str(out), "train": train, "eval": held}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False))
    job = registry.create("embedder", f"Embedding model: {name}")
    job.progress = 0.0
    registry.run_command(job, [classifiers.python(), "-I", "-u", str(WORKER), str(out / "plan.json")], env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    return job.summary() | {"train": len(train), "eval": len(held)}


@router.delete("/api/embedders/{name}")
def delete(name: str):
    info(name)
    classifiers.worker.stop()  # it may hold this model
    shutil.rmtree(folder() / name)
    return embedders_status()
