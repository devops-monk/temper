"""Preference training on Apple Silicon: DPO, ORPO and SimPO as LoRA adapters, with MLX.

mlx-lm trains only by next-token loss, so this is a small trainer of its own built from mlx-lm's
pieces (its model loading, LoRA layers and gradient checkpointing). It saves adapters exactly as
mlx-lm does, so chatting with a run, its checkpoints and export work the same as for fine-tuning.

Each pair is two sequences, prompt + chosen and prompt + rejected, rendered as the model sees them
at inference (chat_render). Only reply tokens count. DPO compares against the starting model: LoRA
starts as a no-op (its B matrices are zero), so the starting model's log-likelihoods are measured
once, before the first step.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from . import chat_render
from .preference import OBJECTIVES

LORA_LAYERS = 16  # the last 16 layers, as mlx-lm fine-tuning does
VAL_BATCHES = 25

Pair = tuple[list[dict], str, str]


def encode(tokenizer, prompt: list[dict], reply: str, limit: int) -> tuple[list[int], int]:
    """Token ids of prompt + reply, cut to `limit`, and where the reply starts."""
    head, tail = chat_render.render(tokenizer, prompt + [{"role": "assistant", "content": reply}])
    start = tokenizer.encode(head, add_special_tokens=False)
    ids = (start + tokenizer.encode(tail, add_special_tokens=False))[:limit]
    return ids, min(len(start), len(ids) - 1)


def batch(items: list[tuple[list[int], int]]) -> tuple[mx.array, mx.array, mx.array]:
    """Padded ids, lengths and reply starts for a list of encoded sequences."""
    width = max(len(ids) for ids, _ in items)
    ids = mx.array([ids + [0] * (width - len(ids)) for ids, _ in items])
    return ids, mx.array([len(i) for i, _ in items]), mx.array([s for _, s in items])


def reply_logps(model, ids: mx.array, lengths: mx.array, starts: mx.array) -> tuple[mx.array, mx.array]:
    """Each sequence's summed log-likelihood over its reply tokens, and how many there are."""
    logits = model(ids[:, :-1]).astype(mx.float32)
    targets = ids[:, 1:]
    picked = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)
    pos = mx.arange(targets.shape[1])[None]
    mask = (pos >= starts[:, None] - 1) & (pos < lengths[:, None] - 1)
    return (picked * mask).sum(-1), mask.sum(-1)


def log_sigmoid(x: mx.array) -> mx.array:
    return -mx.logaddexp(0, -x)


def objective_loss(objective: str, beta: float, gamma: float, c, r, nc, nr, ref_c=None, ref_r=None):
    """(loss, margin per pair) for chosen and rejected log-likelihoods `c`, `r` over `nc`, `nr` tokens."""
    if objective == "dpo":
        margin = beta * ((c - ref_c) - (r - ref_r))
        return -log_sigmoid(margin).mean(), margin
    avg_c, avg_r = c / mx.maximum(nc, 1), r / mx.maximum(nr, 1)
    if objective == "simpo":
        margin = beta * (avg_c - avg_r) - gamma
        return -log_sigmoid(margin).mean(), margin
    # ORPO: next-token loss on the chosen reply, plus the log odds ratio of chosen over rejected.
    odds = lambda a: a - mx.log1p(-mx.exp(mx.minimum(a, -1e-6)))  # noqa: E731
    margin = odds(avg_c) - odds(avg_r)
    return -avg_c.mean() - beta * log_sigmoid(margin).mean(), margin


def train(cfg: dict, out: Path, train_pairs: list[Pair], valid_pairs: list[Pair], emit: Callable[..., None], steps: int, eval_every: int, save_every: int) -> None:
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    objective = cfg["objective"]
    params = OBJECTIVES[objective]
    beta = cfg.get("pref_beta") or params["beta"]
    gamma = params.get("gamma", 0.0)

    emit("phase", name="Loading model")
    model, tokenizer = load(cfg["base_model"])
    model.freeze()
    lora = {"rank": cfg["lora_r"], "scale": cfg["lora_alpha"] / cfg["lora_r"], "dropout": cfg["lora_dropout"]}
    layers = min(LORA_LAYERS, len(model.layers))
    linear_to_lora_layers(model, layers, lora)
    if cfg.get("resume_from"):
        model.load_weights(str(cfg["resume_from"]), strict=False)
    grad_checkpoint(model.layers[0])

    emit("phase", name="Preparing pairs")
    limit = cfg["max_seq_length"]
    enc = lambda ps: [(encode(tokenizer, p, c, limit), encode(tokenizer, p, r, limit)) for p, c, r in ps]  # noqa: E731
    train_set, valid_set = enc(train_pairs), enc(valid_pairs)
    size = cfg["batch_size"]

    ref: dict[tuple[str, int], tuple[float, float]] = {}
    if objective == "dpo":
        # The starting model's likelihoods: LoRA is still a no-op here.
        emit("phase", name="Measuring the starting model")
        for name, items in (("train", train_set), ("valid", valid_set)):
            for i in range(0, len(items), size):
                chunk = items[i : i + size]
                total, _ = reply_logps(model, *batch([c for c, _ in chunk] + [r for _, r in chunk]))
                mx.eval(total)
                k = len(chunk)
                for j in range(k):
                    ref[(name, i + j)] = (total[j].item(), total[k + j].item())

    def loss_fn(model, ids, lengths, starts, ref_c, ref_r):
        total, n = reply_logps(model, ids, lengths, starts)
        k = total.shape[0] // 2
        loss, margin = objective_loss(objective, beta, gamma, total[:k], total[k:], n[:k], n[k:], ref_c, ref_r)
        return loss, margin

    grad_fn = nn.value_and_grad(model, loss_fn)

    def inputs(name: str, items, idx: list[int]):
        chunk = [items[i] for i in idx]
        ids, lengths, starts = batch([c for c, _ in chunk] + [r for _, r in chunk])
        refs = [ref.get((name, i), (0.0, 0.0)) for i in idx]
        return ids, lengths, starts, mx.array([a for a, _ in refs]), mx.array([b for _, b in refs])

    warmup = min(cfg["warmup_steps"], max(steps - 1, 0))
    lr = cfg["learning_rate"]
    decay = optim.linear_schedule(lr, lr * 0.1, max(steps - warmup, 1))
    schedule = optim.join_schedules([optim.linear_schedule(lr * 0.01, lr, warmup), decay], [warmup]) if warmup else decay
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)

    rng = random.Random(cfg["seed"])
    order: list[int] = []

    def next_idx() -> list[int]:
        nonlocal order
        if len(order) < size:
            fresh = list(range(len(train_set)))
            rng.shuffle(fresh)
            order += fresh
        idx, order = order[:size], order[size:]
        return idx

    def evaluate(step: int) -> None:
        if not valid_set:
            return
        losses, wins, count = [], 0, 0
        for b, i in enumerate(range(0, len(valid_set), size)):
            if b == VAL_BATCHES:
                break
            idx = list(range(i, min(i + size, len(valid_set))))
            loss, margin = loss_fn(model, *inputs("valid", valid_set, idx))
            mx.eval(loss, margin)
            losses.append(loss.item() * len(idx))
            wins += int((margin > 0).sum().item())
            count += len(idx)
        emit("eval", step=step, val_loss=sum(losses) / count, val_accuracy=wins / count)

    adapter = out / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": layers, "lora_parameters": lora, "model": cfg["base_model"],
        "objective": objective, "beta": beta, **({"gamma": gamma} if objective == "simpo" else {}),
    }, indent=2))
    save = lambda path: mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))  # noqa: E731

    emit("phase", name="Training", total_steps=steps)
    model.train()
    started = time.time()
    tokens = 0
    for step in range(1, steps + 1):
        grads, loss_sum, wins, margins = None, 0.0, 0, 0.0
        for _ in range(cfg["grad_accum"]):
            batch_in = inputs("train", train_set, next_idx())
            tokens += int(batch_in[1].sum().item())
            (loss, margin), g = grad_fn(model, *batch_in)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            mx.eval(loss, margin, grads)
            loss_sum += loss.item()
            wins += int((margin > 0).sum().item())
            margins += margin.sum().item()
        accum = cfg["grad_accum"]
        opt.update(model, tree_map(lambda g: g / accum, grads))
        mx.eval(model.parameters(), opt.state)
        n = size * accum
        elapsed = time.time() - started
        emit("metrics", step=step, total_steps=steps, loss=round(loss_sum / accum, 4), reward_accuracy=wins / n, margin=round(margins / n, 4),
             lr=float(schedule(step)), tokens_per_sec=round(tokens / max(elapsed, 1e-6)), eta=elapsed / step * (steps - step))
        emit("progress", value=step / steps)
        if step == 1 or step % eval_every == 0 or step == steps:
            model.eval()
            evaluate(step)
            model.train()
        if step % save_every == 0 and step != steps:
            save(adapter / f"{step:07d}_adapters.safetensors")
    save(adapter / "adapters.safetensors")
    emit("result", adapter=str(adapter), base_model=cfg["base_model"])
