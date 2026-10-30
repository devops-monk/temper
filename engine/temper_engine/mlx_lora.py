"""`python -m temper_engine.mlx_lora <mlx_lm lora args>`: mlx-lm's LoRA trainer with Temper's chat rendering.

mlx-lm tokenizes chat rows with the template's history form; this swaps in `chat_render.render` so
training text matches inference, and masks the prompt so loss covers only the reply.

With TEMPER_TRAIN_EMBEDDINGS set, the token embeddings and output head are trained too (mlx-lm
leaves them frozen): continued pretraining that teaches new words or a new language.
"""

from __future__ import annotations

import os

from mlx_lm import lora
from mlx_lm.tuner import datasets

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

if __name__ == "__main__":
    lora.main()
