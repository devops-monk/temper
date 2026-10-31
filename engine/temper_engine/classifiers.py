"""Classifiers: small models that sort text into your labels, trained on your examples in a minute.

A classifier is a sentence encoder (a small embedding model) with a classification head trained on
top: logistic regression on the encoder's vectors, optionally after tuning the encoder on pairs of
examples (classifier_worker.py). It answers in milliseconds with a confidence for each label, so it suits
decisions made many times over (sorting tickets, flagging spam) better than asking a chat model.

The training data is a CSV, JSON or JSONL file with a text column and a label column, found by name
(text, sentence, content… and label, category, class…) or given. A share of each label is held out to
measure the classifier; a prepared `x.jsonl` with an `x.eval.jsonl` beside it uses that instead.

PyTorch and sentence-transformers install on request into their own environment under
engines/classifier. Classifiers are folders under classifiers/ in Temper's home. A saved decision can
answer with one (decisions.py), and apps reach them at POST /v1/classifiers/{name}.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import engines, home, storage
from .jobs import registry
from .speech import Worker

router = APIRouter(tags=["classifiers"])

PACKAGES = [engines.TORCH, "sentence-transformers>=3", "scikit-learn", "datasets", "accelerate"]
WORKER = Path(__file__).with_name("classifier_worker.py")
TEXT_COLUMNS = ("text", "sentence", "content", "input", "message", "body", "review", "question", "prompt", "document")
LABEL_COLUMNS = ("label", "labels", "category", "class", "intent", "sentiment", "topic", "target", "answer", "output")
MAX_ROWS = 50_000
MAX_LABELS = 100
_FILES = ["*.json", "*.txt", "*.model", "model.safetensors", "1_Pooling/*", "2_Normalize/*"]

ENCODERS = [
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "name": "MiniLM L6", "size": 91_000_000, "about": "English; small and quick. A good first choice."},
    {"id": "BAAI/bge-small-en-v1.5", "name": "BGE small", "size": 134_000_000, "about": "English; a little slower, often a little better."},
    {"id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "name": "Multilingual MiniLM", "size": 471_000_000, "about": "Over 50 languages."},
]


def venv() -> Path:
    return home() / "engines" / "classifier"


def python() -> str | None:
    p = engines._venv_python(venv())
    return str(p) if p.is_file() and (venv() / ".installed").is_file() else None


def encoder_folder(encoder: str) -> str | None:
    """A sentence-transformers model on this computer: a folder, or a downloaded Hugging Face repo."""
    path = Path(encoder).expanduser()
    if path.is_dir():
        return str(path) if (path / "config.json").is_file() else None
    snapshots = storage.models_dir() / ("models--" + encoder.replace("/", "--")) / "snapshots"
    found = sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if snapshots.is_dir() else []
    return next((str(s) for s in found if (s / "config.json").is_file() and (any(s.glob("*.safetensors")) or (s / "pytorch_model.bin").is_file())), None)


def folder() -> Path:
    path = home() / "classifiers"
    path.mkdir(exist_ok=True)
    return path


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]
    if not slug:
        raise HTTPException(400, "Give the classifier a name.")
    return slug


def info(name: str) -> dict:
    path = folder() / name
    if not re.fullmatch(r"[a-z0-9-]+", name) or not (path / "classifier.json").is_file():
        raise HTTPException(404, f"There's no classifier called {name}.")
    data = json.loads((path / "classifier.json").read_text())
    return {k: v for k, v in data.items() if k != "head"} | {"path": str(path)}


def trained() -> list[dict]:
    out = []
    for d in folder().iterdir():
        if (d / "classifier.json").is_file():
            try:
                out.append(info(d.name))
            except (HTTPException, ValueError, OSError):
                continue
    return sorted(out, key=lambda c: c["created"], reverse=True)


def status() -> dict:
    return {"installed": python() is not None, "running": worker.running(),
            "encoders": [e | {"downloaded": encoder_folder(e["id"]) is not None} for e in ENCODERS], "classifiers": trained()}


worker = Worker("classifier", lambda: [python(), "-I", "-u", str(WORKER), "serve"])


# --- Reading examples ------------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise HTTPException(404, f"No file at {path}.")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        elif suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), []) if isinstance(data, dict) else []
        else:
            raise HTTPException(400, "Use a CSV, JSON or JSONL file.")
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"{path.name} couldn't be read: {e}") from e
    return [r for r in rows[: MAX_ROWS * 2] if isinstance(r, dict)]


def _column(rows: list[dict], given: str | None, names: tuple[str, ...], what: str) -> str:
    columns = list(rows[0]) if rows else []
    if given:
        if given not in columns:
            raise HTTPException(400, f"There's no column called {given}. The columns are: {', '.join(columns)}.")
        return given
    folded = {c.casefold(): c for c in columns}
    found = next((folded[n] for n in names if n in folded), None)
    if found is None:
        raise HTTPException(400, f"Which column holds the {what}? The columns are: {', '.join(columns)}.")
    return found


def examples(path: Path, text_column: str | None = None, label_column: str | None = None) -> tuple[list[list[str]], str, str]:
    """[text, label] pairs from a file, with the columns they came from. Rows missing either are skipped."""
    rows = read_rows(path)
    if not rows:
        raise HTTPException(400, f"{path.name} has no rows.")
    text_col = _column(rows, text_column, TEXT_COLUMNS, "text")
    label_col = _column(rows, label_column, LABEL_COLUMNS, "labels")
    out = []
    for r in rows:
        text, label = r.get(text_col), r.get(label_col)
        if isinstance(label, bool):
            label = "yes" if label else "no"
        if isinstance(text, str) and text.strip() and label is not None and str(label).strip():
            out.append([text.strip(), str(label).strip()])
    return out[:MAX_ROWS], text_col, label_col


def split(rows: list[list[str]], share: float, seed: int = 42) -> tuple[list, list]:
    """Holds out about `share` of each label, keeping at least one example of it for training."""
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    train, held = [], []
    for group in by_label.values():
        group = group[:]
        rng.shuffle(group)
        n = min(len(group) - 1, round(len(group) * share))
        held += group[:n]
        train += group[n:]
    rng.shuffle(train)
    return train, held


def problem(counts: Counter, label_col: str) -> str | None:
    """Why these labels can't train a classifier, if they can't."""
    if len(counts) < 2:
        return f"A classifier needs at least two labels; {label_col} has {len(counts)}."
    if len(counts) > MAX_LABELS:
        return f"{label_col} has {len(counts):,} different values; up to {MAX_LABELS} labels work. Is it the right column?"
    few = [k for k, v in counts.items() if v < 2]
    if few:
        return f"Give each label at least two examples; {', '.join(few[:5])}{'…' if len(few) > 5 else ''} {'has' if len(few) == 1 else 'have'} only one."
    return None


def plan_data(req: "TrainRequest") -> dict:
    path = Path(req.dataset).expanduser()
    rows, text_col, label_col = examples(path, req.text_column, req.label_column)
    counts = Counter(label for _, label in rows)
    if why := problem(counts, label_col):
        raise HTTPException(400, why)
    evals = path.with_name(path.name.removesuffix(".jsonl") + ".eval.jsonl") if path.suffix == ".jsonl" else None
    if evals and evals.is_file() and evals != path:
        held = [r for r in examples(evals, text_col, label_col)[0] if r[1] in counts]
        train = rows
    else:
        train, held = split(rows, req.eval_share)
    return {"train": train, "eval": held, "labels": sorted(counts), "counts": dict(counts), "text_column": text_col, "label_column": label_col}


# --- Routes ------------------------------------------------------------------------------------------


class TrainRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    dataset: str
    text_column: str | None = None
    label_column: str | None = None
    encoder: str = ENCODERS[0]["id"]
    fine_tune: bool = False
    eval_share: float = Field(0.2, ge=0, le=0.5)


class Peek(BaseModel):
    dataset: str
    text_column: str | None = None
    label_column: str | None = None


class Choice(BaseModel):
    encoder: str


class Classify(BaseModel):
    text: str | None = None
    texts: list[str] | None = Field(None, max_length=1000)


@router.get("/api/classifiers")
def classifiers_status():
    return status()


@router.post("/api/classifiers/install")
def install():
    job = registry.create("engine-install", "Install classifier training (sentence-transformers)")
    registry.run_thread(job, lambda j: engines.install_venv(j, venv(), PACKAGES, "classifier training"))
    return job.summary()


@router.delete("/api/classifiers/install")
def uninstall():
    worker.stop()
    shutil.rmtree(venv(), ignore_errors=True)
    return status()


@router.post("/api/classifiers/encoders/download")
def download_encoder(req: Choice):
    from .hub import DownloadRequest
    from .hub import download as hub_download

    return hub_download(DownloadRequest(repo_id=req.encoder, patterns=_FILES))


@router.post("/api/classifiers/peek")
def peek(req: Peek):
    """What training on a file would use: its columns, the ones picked, and how many examples of each label."""
    path = Path(req.dataset).expanduser()
    rows = read_rows(path)
    columns = list(rows[0]) if rows else []
    try:
        found, text_col, label_col = examples(path, req.text_column, req.label_column)
    except HTTPException as e:
        return {"columns": columns, "rows": len(rows), "problem": e.detail}
    counts = Counter(label for _, label in found)
    return {"columns": columns, "rows": len(rows), "text_column": text_col, "label_column": label_col, "examples": len(found), "labels": dict(counts.most_common()),
            "problem": problem(counts, label_col)}


@router.post("/api/classifiers")
def train(req: TrainRequest):
    if python() is None:
        raise HTTPException(409, "Classifier training isn't installed. Install it on the Train page (Classifier), or run `temper classify --install`.")
    encoder = encoder_folder(req.encoder)
    if encoder is None:
        raise HTTPException(409, f"Download the encoder first ({req.encoder}).")
    name = _slug(req.name)
    out = folder() / name
    if (out / "classifier.json").is_file():
        raise HTTPException(409, f"There's already a classifier called {name}. Delete it or pick another name.")
    data = plan_data(req)
    shutil.rmtree(out, ignore_errors=True)  # what a failed run left
    out.mkdir()
    plan = {"name": name, "encoder": encoder, "encoder_id": req.encoder, "dataset": str(Path(req.dataset).expanduser()), "fine_tune": req.fine_tune,
            "created": time.time(), "out": str(out), **data}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False))
    job = registry.create("classifier", f"Classifier: {name}")
    job.progress = 0.0
    registry.run_command(job, [python(), "-I", "-u", str(WORKER), "train", str(out / "plan.json")], env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    return job.summary() | {"labels": data["counts"], "train": len(data["train"]), "eval": len(data["eval"])}


@router.delete("/api/classifiers/{name}")
def delete(name: str):
    info(name)
    worker.stop()  # it may hold this one's encoder
    shutil.rmtree(folder() / name)
    return status()


def classify(name: str, texts: list[str]) -> list[dict]:
    path = info(name)["path"]
    if python() is None:
        raise HTTPException(409, "Classifier support isn't installed. Install it on the Train page (Classifier).")
    if not texts:
        return []
    try:
        return worker.request({"classifier": path, "texts": texts})["results"]
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/v1/classifiers/{name}")
def classify_route(name: str, req: Classify):
    """Each text's label, how sure the classifier is, and its score for every label."""
    if req.texts is not None:
        return {"classifier": name, "results": classify(name, req.texts)}
    if req.text is None:
        raise HTTPException(400, "Send `text`, or `texts` for several.")
    return {"classifier": name, **classify(name, [req.text])[0]}
