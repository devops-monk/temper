"""Dataset formats: detect them, map columns to chat turns, clean and split. No third-party imports.

Supported shapes, each with a default column the user can remap:
  messages  — OpenAI-style list of {role, content}
  sharegpt  — list of {from, value}
  alpaca    — instruction (+ optional input) → output, optional system column
  text      — plain text for continued pretraining
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

FORMATS = ("messages", "sharegpt", "alpaca", "text")
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system", "model": "assistant", "bot": "assistant"}
# Default column for each role in each format; a mapping overrides any of them.
DEFAULT_COLUMNS = {
    "messages": {"messages": "messages"},
    "sharegpt": {"conversations": "conversations"},
    "alpaca": {"instruction": "instruction", "input": "input", "output": "output", "system": "system"},
    "text": {"text": "text"},
}
REQUIRED = {"messages": ["messages"], "sharegpt": ["conversations"], "alpaca": ["instruction", "output"], "text": ["text"]}


def detect_format(columns: list[str]) -> str | None:
    cols = set(columns)
    if "messages" in cols:
        return "messages"
    if "conversations" in cols:
        return "sharegpt"
    if {"instruction", "output"} <= cols:
        return "alpaca"
    if "text" in cols:
        return "text"
    return None


def guess_mapping(columns: list[str]) -> tuple[str, dict[str, str]] | None:
    """A format and mapping for common column names that `detect_format` doesn't know."""
    cols = {c.lower(): c for c in columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)  # noqa: E731
    chat = pick("chat", "conversation", "dialogue", "dialog", "turns")
    if chat:
        return "messages", {"messages": chat}
    prompt = pick("instruction", "prompt", "question", "query", "input_text", "problem", "user")
    answer = pick("output", "response", "answer", "completion", "solution", "target", "assistant", "chosen")
    if prompt and answer and prompt != answer:
        mapping = {"instruction": prompt, "output": answer}
        if context := pick("input", "context"):
            if context not in (prompt, answer):
                mapping["input"] = context
        if system := pick("system", "system_prompt"):
            mapping["system"] = system
        return "alpaca", mapping
    body = pick("content", "body", "document", "passage")
    return ("text", {"text": body}) if body else None


def _turns(value) -> list:
    if isinstance(value, str):  # CSV stores lists as JSON text
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("expected a list of turns")
    return value


def to_messages(row: dict, fmt: str, mapping: dict[str, str] | None = None, system_prompt: str | None = None) -> list[dict] | str:
    """One row as chat messages (or plain text for `text`). Raises ValueError for a row it can't use."""
    cols = {**DEFAULT_COLUMNS[fmt], **(mapping or {})}
    if fmt == "text":
        text = row.get(cols["text"])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"no text in column “{cols['text']}”")
        return text
    if fmt == "alpaca":
        instruction, output = row.get(cols["instruction"]), row.get(cols["output"])
        if instruction in (None, "") or output in (None, ""):
            raise ValueError("empty instruction or output")
        prompt = str(instruction)
        if cols.get("input") and row.get(cols["input"]):
            prompt += f"\n\n{row[cols['input']]}"
        out = [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(output)}]
        system = row.get(cols["system"]) if cols.get("system") else None
        if system:
            out.insert(0, {"role": "system", "content": str(system)})
    else:
        key = cols["messages"] if fmt == "messages" else cols["conversations"]
        out = []
        for turn in _turns(row.get(key)):
            role = ROLE_MAP.get(str(turn.get("role") or turn.get("from") or "").lower(), "user")
            content = turn.get("content") if "content" in turn else turn.get("value", "")
            out.append({"role": role, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
        if not any(t["role"] == "assistant" and t["content"].strip() for t in out):
            raise ValueError("no assistant reply")
    if system_prompt and (not out or out[0]["role"] != "system"):
        out.insert(0, {"role": "system", "content": system_prompt})
    return out


def length(conv: list[dict] | str) -> int:
    return len(conv) if isinstance(conv, str) else sum(len(t["content"]) for t in conv)


def fingerprint(conv: list[dict] | str) -> str:
    """Same text after case and whitespace folding → same fingerprint."""
    text = conv if isinstance(conv, str) else "\x1e".join(f"{t['role']}\x1f{t['content']}" for t in conv)
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


@dataclass
class CleanOptions:
    dedupe: bool = True
    min_chars: int = 0
    max_chars: int = 0  # 0 = no limit
    eval_fraction: float = 0.0
    shuffle: bool = True
    seed: int = 3407
    max_rows: int = 0  # 0 = keep all


@dataclass
class CleanResult:
    train: list = field(default_factory=list)
    eval: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def clean(rows: list[dict], fmt: str, mapping: dict[str, str] | None, system_prompt: str | None, opts: CleanOptions) -> CleanResult:
    stats = {"rows_in": len(rows), "malformed": 0, "duplicates": 0, "too_short": 0, "too_long": 0}
    seen: set[str] = set()
    kept = []
    for row in rows:
        try:
            conv = to_messages(row, fmt, mapping, system_prompt)
        except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
            stats["malformed"] += 1
            continue
        size = length(conv)
        if size < opts.min_chars:
            stats["too_short"] += 1
            continue
        if opts.max_chars and size > opts.max_chars:
            stats["too_long"] += 1
            continue
        if opts.dedupe:
            key = fingerprint(conv)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
        kept.append(conv)
    if opts.shuffle:
        random.Random(opts.seed).shuffle(kept)
    if opts.max_rows:
        kept = kept[: opts.max_rows]
    n_eval = round(len(kept) * opts.eval_fraction) if opts.eval_fraction > 0 else 0
    if opts.eval_fraction > 0 and len(kept) > 1:
        n_eval = min(max(n_eval, 1), len(kept) - 1)
    result = CleanResult(train=kept[n_eval:], eval=kept[:n_eval])
    result.stats = {**stats, "train": len(result.train), "eval": len(result.eval)}
    return result


def record(conv: list[dict] | str) -> dict:
    """The JSONL row Temper writes for a prepared dataset."""
    return {"text": conv} if isinstance(conv, str) else {"messages": conv}
