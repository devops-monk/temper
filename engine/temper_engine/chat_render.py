"""Render a conversation for training exactly as the model will see it at inference.

Some chat templates format an assistant turn differently in history than after the generation
prompt. Qwen3's, for example, inserts an empty `<think></think>` block into every assistant turn it
renders, but its generation prompt ends at `<|im_start|>assistant\\n`. Training on the history form
teaches the model to start replies with tokens it never sees at inference, and it then opens every
answer with a stray token. So: prompt = the template with the generation prompt, completion = the
reply plus whatever the template puts after it (e.g. `<|im_end|>\\n`).
"""

from __future__ import annotations


def render(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """(prompt, completion) for a conversation that ends with an assistant reply."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the last turn must be the assistant's")
    reply = messages[-1]["content"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    end = full.rfind(reply)
    suffix = full[end + len(reply):] if end >= 0 else ""
    return prompt, reply + suffix
