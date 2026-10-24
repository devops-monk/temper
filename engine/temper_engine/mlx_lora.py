"""`python -m temper_engine.mlx_lora <mlx_lm lora args>`: mlx-lm's LoRA trainer with Temper's chat rendering.

mlx-lm tokenizes chat rows with the template's history form; this swaps in `chat_render.render` so
training text matches inference, and masks the prompt so loss covers only the reply.
"""

from __future__ import annotations

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

if __name__ == "__main__":
    lora.main()
