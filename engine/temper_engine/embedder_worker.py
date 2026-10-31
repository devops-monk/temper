"""Fine-tunes a sentence-embedding model on pairs, inside the classifier environment (classifiers.py).

Run by path with that environment's Python, so it imports nothing from temper_engine:
`python embedder_worker.py <plan.json>`. The plan names the base model (a sentence-transformers folder)
and the training and held-out pairs, each [anchor, positive]: a question and the passage that answers it,
a search and the product it should find. Training pulls each anchor towards its positive and away from
the other positives in its batch (multiple-negatives ranking loss), so no negatives are needed.

Before and after, it measures retrieval on the held-out pairs: each anchor searches all their positives,
plus some training positives as distractors, and recall@1, recall@5 and MRR@10 say how often the right
one comes first or near the top. It saves the tuned model to model/ and embedder.json beside it, and
prints job events as `@@{json}` lines (the format jobs.run_command reads).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DISTRACTORS = 500


def emit(event: str, **data) -> None:
    print("@@" + json.dumps({"event": event, **data}), flush=True)


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def retrieval(model, held: list[list[str]], corpus_extra: list[str]) -> dict | None:
    """How well each held-out anchor finds its own positive among all the candidates."""
    import numpy as np

    if not held:
        return None
    corpus = list(dict.fromkeys([p for _, p in held] + corpus_extra))
    index = {p: i for i, p in enumerate(corpus)}
    queries = model.encode([a for a, _ in held], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    docs = model.encode(corpus, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    order = np.argsort(-(queries @ docs.T), axis=1)
    ranks = [int(np.where(order[i] == index[p])[0][0]) + 1 for i, (_, p) in enumerate(held)]
    return {"recall_at_1": round(sum(r == 1 for r in ranks) / len(ranks), 4), "recall_at_5": round(sum(r <= 5 for r in ranks) / len(ranks), 4),
            "mrr_at_10": round(sum(1 / r for r in ranks if r <= 10) / len(ranks), 4), "candidates": len(corpus)}


def main() -> None:
    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(plan["out"])
    try:
        emit("phase", name="Loading the base model")
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader

        model = SentenceTransformer(plan["base"], device=device())
        train, held = plan["train"], plan["eval"]
        extra = random.Random(42).sample([p for _, p in train], min(DISTRACTORS, len(train)))
        emit("phase", name="Measuring the base model")
        before = retrieval(model, held, extra)
        emit("progress", value=0.1)
        emit("phase", name=f"Training on {len(train):,} pairs")
        loader = DataLoader([InputExample(texts=[a, p]) for a, p in train], shuffle=True, batch_size=int(plan.get("batch_size") or 16))
        steps = len(loader) * int(plan.get("epochs") or 1)

        model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))], epochs=int(plan.get("epochs") or 1), warmup_steps=max(1, steps // 10),
                  show_progress_bar=False)
        emit("progress", value=0.85)
        emit("phase", name="Measuring the tuned model")
        after = retrieval(model, held, extra)
        model.save(str(out / "model"))
        info = {k: plan[k] for k in ("name", "base_id", "dataset", "created")}
        info.update(train_pairs=len(train), eval_pairs=len(held), epochs=int(plan.get("epochs") or 1), before=before, after=after,
                    dimensions=int(model.get_sentence_embedding_dimension()))
        (out / "embedder.json").write_text(json.dumps(info, ensure_ascii=False))
        emit("progress", value=1.0)
        emit("result", name=plan["name"], before=before, after=after)
    except Exception as e:
        emit("error", message=str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
