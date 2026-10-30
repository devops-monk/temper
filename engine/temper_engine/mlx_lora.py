"""`python -m temper_engine.mlx_lora <mlx_lm lora args>`: mlx-lm's LoRA trainer with Temper's chat rendering.

mlx-lm tokenizes chat rows with the template's history form; this swaps in `chat_render.render` so
training text matches inference, and masks the prompt so loss covers only the reply.

With TEMPER_TRAIN_EMBEDDINGS set, the token embeddings and output head are trained too (mlx-lm
leaves them frozen): continued pretraining that teaches new words or a new language.

At each evaluation, with TEMPER_SAMPLES naming a JSON file of prompts, the model answers them and
the replies are printed as `TEMPER {"event": "sample", ...}` lines for the worker to pass on. With
TEMPER_EARLY_STOP set to a count, training stops once validation loss hasn't improved for that many
evaluations, and the adapter saved is the one from the best evaluation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from mlx_lm import lora
from mlx_lm.tuner import datasets
from mlx_lm.tuner.callbacks import TrainingCallback

from .chat_render import render

_original = datasets.ChatDataset.process


def _process(self, d):
    messages = d[self.chat_key]
    if d.get("tools") or not messages or messages[-1].get("role") != "assistant":
        return _original(self, d)
    prompt, completion = render(self.tokenizer, messages)
    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
    tokens = prompt_ids + self.tokenizer.encode(completion, add_special_tokens=False)
    return (tokens, len(prompt_ids) if self.mask_prompt else 0)


datasets.ChatDataset.process = _process

_train_model = lora.train_model


def _train_model_with_embeddings(args, model, *rest, **kw):
    # train_model freezes the whole model first, then unfreezes what it trains; unfreezing the
    # embeddings and head straight after that first freeze keeps them trainable.
    freeze = model.freeze

    def freeze_then_unfreeze_embeddings(*a, **k):
        done = freeze(*a, **k)
        inner = getattr(model, "model", model)
        for part in (getattr(inner, "embed_tokens", None), getattr(model, "lm_head", None)):
            if part is not None:
                part.unfreeze()
        return done

    model.freeze = freeze_then_unfreeze_embeddings
    return _train_model(args, model, *rest, **kw)


if os.environ.get("TEMPER_TRAIN_EMBEDDINGS"):
    lora.train_model = _train_model_with_embeddings


def say(event: str, **data) -> None:
    print("TEMPER " + json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


class EarlyStop(Exception):
    pass


class Watch(TrainingCallback):
    """Sample replies and early stopping, at each evaluation; passes reports on to mlx-lm's own callback."""

    def __init__(self, model, args, val_dataset, inner: TrainingCallback | None) -> None:
        self.model, self.args, self.inner = model, args, inner
        self.tokenizer = getattr(getattr(val_dataset, "_data", val_dataset), "tokenizer", None)
        path = os.environ.get("TEMPER_SAMPLES")
        self.prompts = json.loads(Path(path).read_text()) if path else []
        self.patience = int(os.environ.get("TEMPER_EARLY_STOP") or 0)
        self.best, self.best_step, self.bad = float("inf"), None, 0
        self.best_file = Path(args.adapter_file).with_name("best_adapters.safetensors")

    def on_train_loss_report(self, info: dict) -> None:
        if self.inner:
            self.inner.on_train_loss_report(info)

    def on_val_loss_report(self, info: dict) -> None:
        if self.inner:
            self.inner.on_val_loss_report(info)
        step = info["iteration"] + 1  # the step mlx-lm prints beside it
        self.samples(step)
        if not self.patience:
            return
        if info["val_loss"] < self.best - 1e-4:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            self.best, self.best_step, self.bad = info["val_loss"], step, 0
            mx.save_safetensors(str(self.best_file), dict(tree_flatten(self.model.trainable_parameters())))
        else:
            self.bad += 1
            if self.bad >= self.patience:
                raise EarlyStop(step)

    def samples(self, step: int) -> None:
        if not self.prompts or self.tokenizer is None:
            return
        from mlx_lm import generate

        self.model.eval()
        try:
            for p in self.prompts:
                text = self.tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
                reply = generate(self.model, self.tokenizer, text, max_tokens=p.get("max_tokens", 160))
                say("sample", step=step, prompt=p["messages"][-1]["content"][:600], reply=reply.strip(), expected=p.get("expected"))
        finally:
            self.model.train()

    def keep_best(self, stopped_at: int | None) -> None:
        """Puts the best evaluation's weights in place as the run's adapter."""
        if self.best_step is None or not self.best_file.is_file():
            return
        shutil.copyfile(self.best_file, self.args.adapter_file)
        self.best_file.unlink()
        say("best", step=self.best_step, val_loss=self.best, stopped_at=stopped_at)


_train = lora.train


def _train_watched(*args, **kw):
    watch = Watch(kw["model"], kw["args"], kw.get("val_dataset"), kw.get("training_callback"))
    kw["training_callback"] = watch
    try:
        _train(*args, **kw)
    except EarlyStop as stop:
        watch.keep_best(stop.args[0])
        return
    watch.keep_best(None)


lora.train = _train_watched

if __name__ == "__main__":
    lora.main()
