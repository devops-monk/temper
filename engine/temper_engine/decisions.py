"""Decisions: typed answers from a model, for apps that need a value rather than prose.

A decision asks a question about some text and answers yes or no, one of a fixed set of choices, a
whole-number score on a scale, or a list of labels from a set, with a one-sentence reason. The model is
held to that shape with a JSON schema where its server supports structured output (llama.cpp, vLLM,
SGLang, Ollama); elsewhere the answer is read out of the reply, and a reply that can't be read is an
error rather than a guess. Answers are asked for at temperature 0, so the same text gets the same answer.

Saved decisions are named questions (`is-spam`, `ticket-priority`) kept in decisions.json, so an app
sends only the text: POST /v1/decisions/{name}. Ad hoc ones go to POST /v1/decide. Both are under /v1,
so a named API key opens them. Any model the engine can route to answers: chat's (the default), one
loaded alongside, or one on a linked device (`model@device`). Or a classifier trained on examples
(classifiers.py) answers instead, with its confidence, much faster: then the answer is one of its labels
(or yes/no, when its labels are yes and no).
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from . import home

router = APIRouter(tags=["decisions"])

Kind = Literal["yes_no", "choice", "score", "labels"]
MAX_INPUTS = 100


class Decision(BaseModel):
    """What to decide; the text it's about comes with each request."""

    question: str = Field(min_length=1, max_length=2000)
    kind: Kind = "yes_no"
    choices: list[str] = Field(default_factory=list, max_length=50)
    scale: tuple[int, int] = (1, 10)
    model: str | None = None
    instructions: str | None = Field(None, max_length=4000)  # how to decide: definitions, examples, edge cases
    classifier: str | None = None  # a trained classifier answers instead of a model

    @model_validator(mode="after")
    def _shape(self):
        self.choices = list(dict.fromkeys(c.strip() for c in self.choices if c.strip()))
        if self.classifier:
            if self.kind not in ("yes_no", "choice"):
                raise ValueError("A classifier answers yes or no, or one of its labels.")
            return self
        if self.kind in ("choice", "labels") and len(self.choices) < 2:
            raise ValueError("Give at least two choices.")
        if self.kind == "score" and self.scale[0] >= self.scale[1]:
            raise ValueError("The scale's low end has to be below its high end.")
        return self


def schema(d: Decision) -> dict:
    answer = {
        "yes_no": {"type": "string", "enum": ["yes", "no"]},
        "choice": {"type": "string", "enum": d.choices},
        "score": {"type": "integer", "minimum": d.scale[0], "maximum": d.scale[1]},
        "labels": {"type": "array", "items": {"type": "string", "enum": d.choices}, "uniqueItems": True},
    }[d.kind]
    # The reason comes first so the model works it out before it commits to an answer.
    return {"type": "object", "properties": {"reason": {"type": "string"}, "answer": answer}, "required": ["reason", "answer"], "additionalProperties": False}


def _ask(d: Decision) -> str:
    return {
        "yes_no": 'Answer "yes" or "no".',
        "choice": "Answer with exactly one of: " + ", ".join(json.dumps(c, ensure_ascii=False) for c in d.choices) + ".",
        "score": f"Answer with a whole number from {d.scale[0]} (lowest) to {d.scale[1]} (highest).",
        "labels": "Answer with a list of every one of these labels that applies, or an empty list: " + ", ".join(json.dumps(c, ensure_ascii=False) for c in d.choices) + ".",
    }[d.kind]


def messages(d: Decision, text: str | None) -> list[dict]:
    system = (
        "You make one decision and reply with JSON only: "
        '{"reason": "<one short sentence>", "answer": <the answer>}. ' + _ask(d) + (f"\n\nHow to decide:\n{d.instructions.strip()}" if d.instructions else "")
    )
    user = f"{d.question.strip()}\n\n<text>\n{text}\n</text>" if text else d.question.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _match(word: str, choices: list[str]) -> str | None:
    folded = word.strip().strip("\"'.").casefold()
    return next((c for c in choices if c.casefold() == folded), None)


def read(d: Decision, reply: str) -> dict:
    """The typed answer in a reply, or a ValueError saying why there isn't one."""
    found = re.search(r"\{.*\}", reply, re.S)
    data = None
    if found:
        try:
            data = json.loads(found.group(0))
        except ValueError:
            data = None
    raw = data.get("answer") if isinstance(data, dict) else reply
    reason = str(data.get("reason") or "").strip() if isinstance(data, dict) else ""
    if d.kind == "yes_no":
        word = raw if isinstance(raw, str) else ("yes" if raw is True else "no" if raw is False else "")
        hit = re.match(r"\W*(yes|no)\b", word, re.I)
        if not hit:
            raise ValueError(f"The model didn't answer yes or no: {reply[:200]}")
        answer: object = hit.group(1).lower() == "yes"
    elif d.kind == "choice":
        answer = _match(str(raw), d.choices) if raw is not None else None
        if answer is None:
            raise ValueError(f"The model's answer isn't one of the choices: {reply[:200]}")
    elif d.kind == "score":
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            number = raw
        else:
            digits = re.search(r"-?\d+", str(raw))
            number = int(digits.group(0)) if digits else None
        if number is None or not d.scale[0] <= number <= d.scale[1]:
            raise ValueError(f"The model's answer isn't a score from {d.scale[0]} to {d.scale[1]}: {reply[:200]}")
        answer = int(number)
    else:
        items = raw if isinstance(raw, list) else re.split(r"[,\n]", str(raw)) if raw else []
        matched = [_match(str(i), d.choices) for i in items]
        if any(m is None for m in matched):
            raise ValueError(f"The model's answer has labels that aren't in the set: {reply[:200]}")
        answer = list(dict.fromkeys(matched))
    return {"answer": answer, "reason": reason}


def decide(d: Decision, text: str | None) -> dict:
    from .runtime import complete

    started = time.monotonic()
    fmt = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": schema(d)}}
    try:
        reply = complete(messages(d, text), max_tokens=300, temperature=0, model=d.model, response_format=fmt)
    except ValueError:
        try:  # a server that refuses structured output: ask plainly and read the reply
            reply = complete(messages(d, text), max_tokens=300, temperature=0, model=d.model)
        except ValueError as e:
            raise HTTPException(502, str(e)) from e
    try:
        out = read(d, reply)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {**out, "seconds": round(time.monotonic() - started, 2)}


YES, NO = {"yes", "true", "1", "y"}, {"no", "false", "0", "n"}


def by_classifier(d: Decision, texts: list[str]) -> list[dict]:
    """The classifier's answers, as decisions: its label (or yes/no) and how sure it is."""
    from .classifiers import classify

    started = time.monotonic()
    results = classify(d.classifier, texts)
    seconds = round((time.monotonic() - started) / max(len(texts), 1), 3)
    out = []
    for r in results:
        label = r["label"]
        if d.kind == "yes_no":
            if label.casefold() not in YES | NO:
                raise HTTPException(400, f"{d.classifier}'s labels aren't yes and no; make this decision a choice instead.")
            answer: object = label.casefold() in YES
        else:
            answer = label
        out.append({"answer": answer, "reason": f"{d.classifier} is {r['confidence']:.0%} sure.", "confidence": r["confidence"], "scores": r["scores"], "seconds": seconds})
    return out


def _run(d: Decision, text: str | None, texts: list[str] | None) -> dict:
    if texts is not None and len(texts) > MAX_INPUTS:
        raise HTTPException(413, f"Up to {MAX_INPUTS} texts at once.")
    if d.classifier:
        if texts is None and not text:
            raise HTTPException(400, "A classifier needs the `text` to decide about.")
        found = by_classifier(d, texts if texts is not None else [text])
        return {"kind": d.kind, "results": found} if texts is not None else {"kind": d.kind, **found[0]}
    if texts is not None:
        return {"kind": d.kind, "results": [decide(d, t) for t in texts]}
    return {"kind": d.kind, **decide(d, text)}


# --- Saved decisions -------------------------------------------------------------------------------


def _file():
    return home() / "decisions.json"


def saved() -> dict[str, dict]:
    try:
        data = json.loads(_file().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise HTTPException(400, "Name it with lowercase letters, digits, - and _ (such as is-spam).")
    return name


def get(name: str) -> Decision:
    found = saved().get(name)
    if found is None:
        raise HTTPException(404, f"There's no saved decision called {name}.")
    return Decision(**found)


# --- Routes ----------------------------------------------------------------------------------------


class DecideRequest(Decision):
    text: str | None = Field(None, max_length=100_000)
    texts: list[str] | None = None


class Input(BaseModel):
    text: str | None = Field(None, max_length=100_000)
    texts: list[str] | None = None
    model: str | None = None


@router.post("/v1/decide")
def decide_route(req: DecideRequest):
    """Answers a question about `text` (or each of `texts`) as yes/no, a choice, a score or labels."""
    return _run(Decision(**req.model_dump(exclude={"text", "texts"})), req.text, req.texts)


@router.post("/v1/decisions/{name}")
def run_saved(name: str, req: Input):
    d = get(name)
    if req.model:
        d.model = req.model
    return {"decision": name, **_run(d, req.text, req.texts)}


@router.get("/api/decisions")
def list_saved():
    return {"decisions": [{"name": n, **v} for n, v in sorted(saved().items())]}


@router.put("/api/decisions/{name}")
def save(name: str, d: Decision):
    data = saved()
    data[_name(name)] = d.model_dump()
    _file().write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return list_saved()


@router.delete("/api/decisions/{name}")
def delete(name: str):
    data = saved()
    data.pop(name, None)
    _file().write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return list_saved()
