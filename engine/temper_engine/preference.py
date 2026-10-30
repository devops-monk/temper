"""Preference training: teach a model which of two replies to a prompt is better.

Rows hold a prompt, a chosen reply and a rejected one, in either common shape:
  {"prompt": "...", "chosen": "...", "rejected": "..."}                  (prompt as text or turns)
  {"chosen": [turns ending in the good reply], "rejected": [... the bad one]}

Objectives (OBJECTIVES), all trained as LoRA adapters:
  dpo    Direct Preference Optimization: raise the chosen reply's likelihood relative to the starting
         model, and lower the rejected one's. `beta` sets how far it may move from the start.
  orpo   Odds Ratio Preference Optimization: ordinary fine-tuning on the chosen reply plus a penalty
         on the odds of the rejected one. Needs no reference model.
  simpo  Simple Preference Optimization: compares the replies' average per-token likelihood, with a
         target margin `gamma`. Needs no reference model.

No third-party imports: the trainers (train_worker on MLX or GPU) do the maths.
"""

from __future__ import annotations

from .formats import ROLE_MAP

OBJECTIVES = {
    "dpo": {"beta": 0.1},
    "orpo": {"beta": 0.1},
    "simpo": {"beta": 2.0, "gamma": 0.5},
}
PROMPT_COLUMNS = ("prompt", "question", "instruction", "input")


def is_preference(columns) -> bool:
    return {"chosen", "rejected"} <= set(columns)


def _turns(value) -> list[dict]:
    out = []
    for t in value:
        if not isinstance(t, dict):
            raise ValueError("a turn isn't an object")
        role = ROLE_MAP.get(str(t.get("role") or t.get("from") or "").lower())
        if role is None:
            raise ValueError(f"unknown role {t.get('role') or t.get('from')!r}")
        out.append({"role": role, "content": str(t.get("content") if "content" in t else t.get("value") or "")})
    return out


def _prompt(row: dict) -> list[dict]:
    value = next((row[c] for c in PROMPT_COLUMNS if row.get(c)), None)
    if value is None:
        raise ValueError("no prompt")
    turns = _turns(value) if isinstance(value, list) else [{"role": "user", "content": str(value)}]
    if row.get("system") and not any(t["role"] == "system" for t in turns):
        turns.insert(0, {"role": "system", "content": str(row["system"])})
    return turns


def _reply(value) -> tuple[list[dict], str]:
    """A reply given as text, or as turns ending in it (and then, the turns before it)."""
    if isinstance(value, list):
        turns = _turns(value)
        if not turns or turns[-1]["role"] != "assistant":
            raise ValueError("the conversation doesn't end with a reply")
        return turns[:-1], turns[-1]["content"]
    if isinstance(value, dict):
        return _reply([value])
    return [], str(value)


def to_pair(row: dict) -> tuple[list[dict], str, str]:
    """(prompt turns, chosen reply, rejected reply) from a preference row."""
    before, chosen = _reply(row["chosen"])
    _, rejected = _reply(row["rejected"])
    prompt = before if any(t["role"] == "user" for t in before) else _prompt(row)
    if not chosen.strip() or not rejected.strip():
        raise ValueError("a reply is empty")
    return prompt, chosen, rejected


def pairs(rows: list[dict]) -> list[tuple[list[dict], str, str]]:
    """The usable pairs, skipping rows that can't be read (and ones where both replies are the same)."""
    out = []
    for row in rows:
        try:
            p = to_pair(row)
        except (ValueError, TypeError, KeyError):
            continue
        if p[1] != p[2]:
            out.append(p)
    if not out:
        raise ValueError("No usable preference rows: each needs a prompt, a chosen reply and a different rejected one.")
    return out
