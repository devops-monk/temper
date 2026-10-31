"""Reinforcement learning: the prompts a model practises on, and the rewards that score its replies.

The model writes several replies to each prompt; each reply gets a reward, and training makes replies that
scored above their group's average likelier (GRPO and its variants, mlx_grpo.py and train_worker.py).

Rows need a prompt, in a prompt/question/instruction column (or as messages ending with the user's turn),
and whatever the rewards check, usually the expected answer. Rewards (REWARDS), added together:
  answer   1 when the reply's final answer matches the row's expected one (an answer/solution column; for
           worked solutions, the part after ####). The final answer is the last \\boxed{...}, the text after
           "answer:" or ####, or else the last number in the reply.
  format   1 when the reply matches a regular expression (`reward_pattern`), such as an answer tag.
  python   your own function, `def reward(prompt, completion, row) -> float`, given the prompt as text,
           the reply and the dataset row.

No third-party imports: the trainers import this by path or as part of the engine.
"""

from __future__ import annotations

import ast
import re
from typing import Callable

from .preference import _prompt, _turns

REWARDS = {
    "answer": "Correct answer: 1 when the reply's final answer matches the expected one.",
    "format": "Format: 1 when the reply matches a pattern.",
    "python": "Your own Python function.",
}
OBJECTIVES = {
    "grpo": "GRPO: advantages scaled by each group's spread, loss averaged per reply.",
    "dr_grpo": "Dr. GRPO: unscaled advantages and a fixed length normaliser, so long wrong replies aren't favoured.",
    "dapo": "DAPO: loss averaged over every token in the batch, so long replies count fully; groups that all scored the same are skipped.",
}
ANSWER_COLUMNS = ("answer", "solution", "expected", "target", "ground_truth", "output", "label")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def is_rl(objective: str) -> bool:
    return objective in OBJECTIVES


def to_prompt(row: dict) -> list[dict]:
    """The turns the model replies to: messages up to the last user turn, or a prompt column."""
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list) and messages:
        turns = _turns(messages)
        while turns and turns[-1]["role"] == "assistant":
            turns.pop()
        if turns and turns[-1]["role"] == "user":
            return turns
    return _prompt(row)


def prompts(rows: list[dict]) -> list[tuple[list[dict], dict]]:
    """(prompt turns, row) for each row that has a prompt."""
    out = []
    for row in rows:
        try:
            out.append((to_prompt(row), row))
        except (ValueError, TypeError, KeyError):
            continue
    if not out:
        raise ValueError("No usable rows: each needs a prompt (a prompt, question or instruction column, or messages ending with the user's turn).")
    return out


def expected(row: dict) -> str | None:
    value = next((row[c] for c in ANSWER_COLUMNS if row.get(c) not in (None, "")), None)
    if value is None:
        return None
    text = str(value)
    return text.split("####")[-1] if "####" in text else text


def final_answer(text: str) -> str:
    """The answer a reply ends on, normalised for comparison."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    marked = list(re.finditer(r"answer(?: is)?\s*[:=]?\s*(.+)", text, re.I))
    if boxed:
        part = boxed[-1]
    elif "####" in text:
        part = text.split("####")[-1]
    elif marked:
        part = marked[-1].group(1)
    else:
        part = None  # no marked answer: the reply's last number, or all of it
    numbers = NUMBER.findall(text if part is None else part)
    if numbers:
        number = float((numbers[-1] if part is None else numbers[0]).replace(",", ""))
        return str(int(number)) if number.is_integer() else str(number)
    return re.sub(r"\s+", " ", text if part is None else part).strip().strip(".$").lower()


def score_answer(completion: str, row: dict) -> float:
    want = expected(row)
    return float(want is not None and final_answer(completion) == final_answer(want))


def load_python(code: str) -> Callable[[str, str, dict], float]:
    """Your reward function from its source; ValueError if it doesn't define `reward`."""
    check_python(code)
    space: dict = {}
    exec(compile(code, "<reward>", "exec"), space)  # the person's own code, run in their training worker
    return space["reward"]


def check_python(code: str) -> None:
    """Parses (without running) a reward function's source."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"The reward function doesn't parse: line {e.lineno}: {e.msg}") from e
    if not any(isinstance(n, ast.FunctionDef) and n.name == "reward" for n in tree.body):
        raise ValueError("Define a function called reward(prompt, completion, row) that returns a number.")


def check(cfg: dict) -> None:
    """Catches a reward setup that can't work before a run starts."""
    chosen = cfg.get("rewards") or []
    if not chosen:
        raise ValueError("Choose at least one reward.")
    unknown = set(chosen) - set(REWARDS)
    if unknown:
        raise ValueError(f"Unknown reward {', '.join(sorted(unknown))}.")
    if "format" in chosen:
        if not cfg.get("reward_pattern"):
            raise ValueError("The format reward needs a pattern to match.")
        try:
            re.compile(cfg["reward_pattern"])
        except re.error as e:
            raise ValueError(f"The format pattern isn't a regular expression: {e}") from e
    if "python" in chosen:
        if not (cfg.get("reward_code") or "").strip():
            raise ValueError("Write the reward function, or turn off the Python reward.")
        check_python(cfg["reward_code"])


def scorer(cfg: dict) -> Callable[[list[dict], str, dict], float]:
    """A function giving a reply's total reward: (prompt turns, reply, row) -> float."""
    check(cfg)
    chosen = cfg["rewards"]
    pattern = re.compile(cfg["reward_pattern"], re.S) if "format" in chosen else None
    own = load_python(cfg["reward_code"]) if "python" in chosen else None

    def score(prompt: list[dict], completion: str, row: dict) -> float:
        total = 0.0
        if "answer" in chosen:
            total += score_answer(completion, row)
        if pattern is not None:
            total += float(bool(pattern.search(completion)))
        if own is not None:
            text = "\n\n".join(t["content"] for t in prompt if t["role"] == "user")
            total += float(own(text, completion, row))
        return total

    return score
