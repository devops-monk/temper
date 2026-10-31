"""Trains and runs text classifiers, inside the classifier environment (classifiers.py).

Run by path with that environment's Python, so it imports nothing from temper_engine.

`python classifier_worker.py train <plan.json>` trains one. The plan names the encoder (a
sentence-transformers folder), the labels, and the training and held-out rows as [text, label] pairs. The
encoder turns each text into a vector and a logistic-regression head learns the labels from them; with
`fine_tune`, the encoder is first tuned on pairs of examples (same label: alike, different: unlike), the
SetFit recipe, which helps most when there are only a few examples per label. It writes classifier.json
(the head's weights, the labels, and how it did on the held-out rows) and, when tuned, encoder/. It prints
job events as `@@{json}` lines (the format jobs.run_command reads).

`python classifier_worker.py serve` answers JSON lines on stdin, {"classifier": folder, "texts": [...]},
with {"done": {"results": [{"label", "confidence", "scores"}]}} or {"error": "..."}, keeping the
classifiers it has loaded. It also embeds for /v1/embeddings (embedders.py): {"embed": folder, "texts"}
gets {"done": {"vectors": [...]}}, unit length.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BATCH = 64


def emit(event: str, **data) -> None:
    print("@@" + json.dumps({"event": event, **data}), flush=True)


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def encode(model, texts: list[str], on_share=None):
    import numpy as np

    out = []
    for start in range(0, len(texts), BATCH):
        out.append(model.encode(texts[start : start + BATCH], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
        if on_share:
            on_share(min(1.0, (start + BATCH) / max(len(texts), 1)))
    return np.concatenate(out) if out else np.zeros((0, model.get_sentence_embedding_dimension()))


def pairs(rows: list[list[str]], seed: int, limit: int = 4000) -> list[tuple[str, str, float]]:
    """SetFit's pairs: for each example, one with the same label and one with another, a few rounds over."""
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = {}
    for text, label in rows:
        by_label.setdefault(label, []).append(text)
    out = []
    for _ in range(20):
        for text, label in rows:
            same = by_label[label]
            others = [t for other, texts in by_label.items() if other != label for t in texts]
            if len(same) > 1:
                out.append((text, rng.choice([t for t in same if t is not text] or same), 1.0))
            if others:
                out.append((text, rng.choice(others), 0.0))
            if len(out) >= limit:
                return out
    return out


def tune(model, rows: list[list[str]], seed: int) -> int:
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader

    examples = [InputExample(texts=[a, b], label=score) for a, b, score in pairs(rows, seed)]
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    model.fit(train_objectives=[(loader, losses.CosineSimilarityLoss(model))], epochs=1, warmup_steps=max(1, len(loader) // 10), show_progress_bar=False)
    return len(examples)


def probabilities(head: dict, vectors):
    """The head's probability for each label, as scikit-learn's predict_proba would give them."""
    import numpy as np

    coef, intercept = np.asarray(head["coef"]), np.asarray(head["intercept"])
    if coef.shape[0] == 1:  # two labels: one weight vector, for the second
        p = 1 / (1 + np.exp(-(vectors @ coef[0] + intercept[0])))
        return np.stack([1 - p, p], axis=1)
    logits = vectors @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def train(plan_path: str) -> None:
    plan = json.loads(Path(plan_path).read_text())
    out = Path(plan["out"])
    emit("phase", name="Loading the encoder")
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    model = SentenceTransformer(plan["encoder"], device=device())
    train_rows, eval_rows, labels = plan["train"], plan["eval"], plan["labels"]
    tuned = 0
    if plan.get("fine_tune"):
        emit("phase", name="Tuning the encoder on pairs of examples")
        emit("progress", value=0.05)
        tuned = tune(model, train_rows, plan.get("seed", 42))
        model.save(str(out / "encoder"))
    emit("phase", name=f"Reading {len(train_rows) + len(eval_rows):,} examples")
    texts = [t for t, _ in train_rows] + [t for t, _ in eval_rows]
    base = 0.5 if plan.get("fine_tune") else 0.05
    vectors = encode(model, texts, lambda share: emit("progress", value=round(base + (0.9 - base) * share, 3)))
    x_train, x_eval = vectors[: len(train_rows)], vectors[len(train_rows) :]
    emit("phase", name="Training the classifier head")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(x_train, [label for _, label in train_rows])
    head = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "classes": [str(c) for c in clf.classes_]}
    metrics = None
    if eval_rows:
        truth = [label for _, label in eval_rows]
        probs = probabilities(head, x_eval)
        guess = [head["classes"][i] for i in probs.argmax(axis=1)]
        p, r, f, support = precision_recall_fscore_support(truth, guess, labels=labels, zero_division=0)
        metrics = {
            "accuracy": round(float(accuracy_score(truth, guess)), 4),
            "macro_f1": round(float(f1_score(truth, guess, labels=labels, average="macro", zero_division=0)), 4),
            "labels": {label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(support[i])} for i, label in enumerate(labels)},
            "confusion": confusion_matrix(truth, guess, labels=labels).tolist(),
            "mistakes": [{"text": t[:300], "label": y, "guess": g} for (t, y), g in zip(eval_rows, guess, strict=True) if y != g][:20],
        }
    info = {**{k: plan[k] for k in ("name", "labels", "dataset", "encoder_id", "created")}, "fine_tuned": bool(plan.get("fine_tune")), "pairs": tuned,
            "encoder": "encoder" if plan.get("fine_tune") else plan["encoder"], "dimensions": int(vectors.shape[1]), "train_examples": len(train_rows),
            "eval_examples": len(eval_rows), "metrics": metrics, "head": head}
    (out / "classifier.json").write_text(json.dumps(info, ensure_ascii=False))
    emit("progress", value=1.0)
    emit("result", name=plan["name"], metrics=metrics and {k: metrics[k] for k in ("accuracy", "macro_f1")})


_loaded: dict[str, tuple] = {}


def classify(req: dict) -> dict:
    from sentence_transformers import SentenceTransformer

    folder = Path(req["classifier"])
    key = str(folder)
    info = json.loads((folder / "classifier.json").read_text())
    if key not in _loaded or _loaded[key][0] != info["created"]:
        encoder = folder / info["encoder"] if info["encoder"] == "encoder" else info["encoder"]
        _loaded[key] = (info["created"], SentenceTransformer(str(encoder), device=device()), info["head"])
    _, model, head = _loaded[key]
    probs = probabilities(head, encode(model, req["texts"]))
    results = []
    for row in probs:
        scores = {c: round(float(p), 4) for c, p in sorted(zip(head["classes"], row, strict=True), key=lambda x: -x[1])}
        label = next(iter(scores))
        results.append({"label": label, "confidence": scores[label], "scores": scores})
    return {"results": results}


_encoders: dict = {}


def embed(req: dict) -> dict:
    from sentence_transformers import SentenceTransformer

    if req["embed"] not in _encoders:
        _encoders[req["embed"]] = SentenceTransformer(req["embed"], device=device())
    return {"vectors": encode(_encoders[req["embed"]], req["texts"]).tolist()}


def serve() -> None:
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            print(json.dumps({"done": embed(req) if "embed" in req else classify(req)}), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)


def main() -> None:
    if sys.argv[1] == "serve":
        return serve()
    try:
        train(sys.argv[2])
    except Exception as e:
        emit("error", message=str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
