"""Render a conversation for training exactly as the model will see it at inference.

Some chat templates format an assistant turn differently in history than after the generation
prompt. Qwen3's, for example, inserts an empty `<think></think>` block into every assistant turn it
renders, but its generation prompt ends at `<|im_start|>assistant\\n`. Training on the history form
teaches the model to start replies with tokens it never sees at inference, and it then opens every
answer with a stray token. So: prompt = the template with the generation prompt, completion = the
reply plus whatever the template puts after it (e.g. `<|im_end|>\\n`).

A run can replace the model's template (TEMPLATES, or Jinja of your own): to train a base model on
conversations, or to change the format an instruct model uses. The run keeps it beside its adapter
(TEMPLATE_FILE), so chatting with the run and its exports use the same one.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATE_FILE = "chat_template.jinja"
_SYSTEM = "{%- set system = messages[0]['content'] if messages and messages[0]['role'] == 'system' else '' %}"
TEMPLATES = {
    "chatml": {
        "label": "ChatML",
        "about": "<|im_start|> and <|im_end|> around each turn, as Qwen and many fine-tunes use.",
        "template": "{%- for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                    "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    },
    "llama3": {
        "label": "Llama 3",
        "about": "Header and end-of-turn tokens, as Llama 3 and its fine-tunes use.",
        "template": "{{- bos_token }}{%- for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] | trim }}<|eot_id|>{% endfor %}"
                    "{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}",
    },
    "gemma": {
        "label": "Gemma",
        "about": "User and model turns, with the system prompt joined to the first user turn, as Gemma uses.",
        "template": "{{- bos_token }}" + _SYSTEM + "{%- for m in messages if m['role'] != 'system' %}<start_of_turn>{{ 'model' if m['role'] == 'assistant' else 'user' }}\n"
                    "{% if loop.first and system %}{{ system }}\n\n{% endif %}{{ m['content'] | trim }}<end_of_turn>\n{% endfor %}"
                    "{%- if add_generation_prompt %}<start_of_turn>model\n{% endif %}",
    },
    "alpaca": {
        "label": "Instruction / Response",
        "about": "Plain-text ### Instruction and ### Response headings, which any model can learn, base models included.",
        "template": _SYSTEM + "{%- if system %}{{ system }}\n\n{% endif %}{%- for m in messages if m['role'] != 'system' %}"
                    "{%- if m['role'] == 'assistant' %}### Response:\n{{ m['content'] }}{{ eos_token }}\n\n{% else %}### Instruction:\n{{ m['content'] }}\n\n{% endif %}{% endfor %}"
                    "{%- if add_generation_prompt %}### Response:\n{% endif %}",
    },
}


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


def template(value: str | None) -> str | None:
    """The Jinja text for a template given by name (TEMPLATES), as a .jinja file, or as Jinja itself.

    Raises ValueError for one that isn't Jinja or can't render a conversation with its reply in it."""
    if not value or not value.strip():
        return None
    value = value.strip()
    if value in TEMPLATES:
        return TEMPLATES[value]["template"]
    if value.endswith(".jinja") and "\n" not in value:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"There's no template file at {value}.")
        value = path.read_text()
    if "{{" not in value and "{%" not in value:
        raise ValueError(f"Choose one of {', '.join(TEMPLATES)}, or give a Jinja chat template.")
    check(value)
    return value


def check(text: str) -> None:
    """Renders a sample conversation as transformers would, to catch a broken template before training."""
    from jinja2 import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def fail(message):
        raise TemplateError(message)

    env.globals["raise_exception"] = fail
    sample = [{"role": "system", "content": "SYSTEM-1"}, {"role": "user", "content": "USER-2"}, {"role": "assistant", "content": "REPLY-3"}]
    try:
        out = env.from_string(text).render(messages=sample, add_generation_prompt=False, bos_token="", eos_token="")
        env.from_string(text).render(messages=sample[:-1], add_generation_prompt=True, bos_token="", eos_token="")
    except TemplateError as e:
        raise ValueError(f"The chat template doesn't work: {e}") from e
    if "USER-2" not in out or "REPLY-3" not in out:
        raise ValueError("The chat template leaves out the messages: it should write each one's content.")


def use(tokenizer, text: str | None) -> None:
    """Replace the tokenizer's chat template (inside mlx-lm's wrapper, if it's one)."""
    if text:
        # Only mlx-lm's wrapper: a Hugging Face fast tokenizer's own _tokenizer is the Rust one.
        inner = tokenizer._tokenizer if type(tokenizer).__name__ == "TokenizerWrapper" else tokenizer
        inner.chat_template = text


def save(adapter: Path, text: str | None) -> None:
    if text:
        (Path(adapter) / TEMPLATE_FILE).write_text(text, encoding="utf-8")


def saved(adapter: str | Path | None) -> str | None:
    """The template a run trained with, if it replaced the model's."""
    path = Path(adapter) / TEMPLATE_FILE if adapter else None
    return path.read_text(encoding="utf-8") if path and path.is_file() else None
