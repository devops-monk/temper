"""Auto-compaction: keeps a long chat inside the loaded model's context by summarising its oldest turns.

Before each reply the chat page sends the whole thread here. The prompt is counted with the model's
own chat template and tokenizer (llama-server's /apply-template and /tokenize, or the model folder's
tokenizer for other engines). If it leaves too
little room for the reply, the oldest turns are summarised by the same model, a chunk at a time when
they're longer than its context ("rolling"), and the summary goes into the system prompt in their
place. The chat keeps every message on screen; it stores the summary and how many messages it covers.
"""

from __future__ import annotations

import functools

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import attachments
from .engines import translate
from .instructions import with_instructions
from .runtime import RuntimeManager, manager

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Tokens a picture or a sound takes varies by model (64 for SmolVLM, over 1000 for some); this errs high.
MEDIA_TOKENS = {"image_url": 768, "input_audio": 750}
SUMMARY_TOKENS = 600
# Recent turns kept word for word after compacting, as a share of the context.
KEEP_SHARE = 0.4
CHARS_PER_TOKEN = 3.5

SUMMARISE = (
    "Summarise the conversation below so it can continue without it. Keep facts, names, numbers, decisions, "
    "what the user wants and prefers, open questions, and any code or text that will be needed again. "
    "Write plain prose in the third person, under 300 words."
)


class Summary(BaseModel):
    upto: int  # messages[:upto] are summarised
    text: str


class Prepare(BaseModel):
    messages: list[dict]  # the thread: user and assistant turns, content as text or parts
    system: str = ""
    summary: Summary | None = None
    max_tokens: int = 4096
    compact: bool = True
    # Only count: says whether it fits, without summarising or refusing, so the app can say it's compacting.
    check: bool = False
    # The chat's project folder, whose AGENTS.md files join the global instructions (instructions.py).
    project: str | None = None
    instructions: bool = True


def _context(server: RuntimeManager) -> int:
    return server.context or 8192


def _post(server: RuntimeManager, path: str, body: dict, timeout: float = 60) -> dict:
    try:
        r = httpx.post(f"{server.base}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"The model's server didn't answer: {e}") from e


def count(server: RuntimeManager, messages: list[dict]) -> int:
    """Prompt tokens for `messages`, as the model will see them."""
    if not messages:
        return 0
    media = 0
    plain = []
    for m in attachments.expand(messages):
        content = m.get("content")
        if isinstance(content, list):
            media += sum(MEDIA_TOKENS.get(p.get("type"), 0) for p in content)
            content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
        plain.append({"role": m["role"], "content": content or ""})
    if server.engine != "llama.cpp":
        return _count_local(server, plain) + media
    prompt = _post(server, "/apply-template", {"messages": plain})["prompt"]
    return len(_post(server, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]) + media


@functools.lru_cache(maxsize=4)
def _tokenizer(folder: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(folder)


def _count_local(server: RuntimeManager, messages: list[dict]) -> int:
    """Other engines have no tokenize endpoint, so the model folder's own tokenizer counts (no PyTorch needed)."""
    try:
        ids = _tokenizer(server.folder).apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:  # no transformers, or a template that wants something else: roughly, so compaction still kicks in
        return int(sum(len(str(m["content"])) for m in messages) / CHARS_PER_TOKEN) + 8 * len(messages)
    ids = ids.get("input_ids", ids) if hasattr(ids, "get") else ids
    return len(ids[0] if ids and isinstance(ids[0], list) else ids)


def _sendable(messages: list[dict]) -> list[dict]:
    """The turns the model sees: replies that failed (marked "error") are shown in the chat but never sent."""
    return [{k: v for k, v in m.items() if k != "error"} for m in messages if not m.get("error")]


def _with_system(system: str, summary: str | None) -> list[dict]:
    text = "\n\n".join(t for t in (system.strip(), f"Summary of the earlier conversation:\n{summary}" if summary else "") if t)
    return [{"role": "system", "content": text}] if text else []


def _transcript(messages: list[dict]) -> list[str]:
    """Each turn as plain text for summarising; attached documents are included, pictures and sounds named."""
    lines = []
    for m in attachments.expand(messages):
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(p["text"] if p.get("type") == "text" else "[a picture]" if p.get("type") == "image_url" else "[a sound]" for p in content)
        lines.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {content}")
    return lines


def _chunks(turns: list[str], budget: int) -> list[str]:
    """Whole turns packed into chunks of up to `budget` characters; a longer turn is cut into pieces."""
    pieces = [t[i : i + budget] for t in turns for i in range(0, len(t), budget)]
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + len(piece) + 2 <= budget:
            chunks[-1] += "\n\n" + piece
        else:
            chunks.append(piece)
    return chunks


def summarise(server: RuntimeManager, earlier: str | None, messages: list[dict]) -> str:
    """Folds `messages` into the `earlier` summary, a context-sized chunk at a time."""
    budget = max(int((_context(server) - SUMMARY_TOKENS - 400) * CHARS_PER_TOKEN), 4000)  # small contexts still get whole turns
    summary = earlier
    for chunk in _chunks(_transcript(messages), budget):
        before = f"Summary so far:\n{summary}\n\nThe conversation continues:\n" if summary else ""
        # Instructions before and after the transcript, so the model summarises it rather than answering its last question.
        body = {
            "messages": [
                {"role": "system", "content": SUMMARISE},
                {"role": "user", "content": f"{before}<conversation>\n{chunk}\n</conversation>\n\nDon't reply to the conversation. Write only its summary, as instructed."},
            ],
            "max_tokens": SUMMARY_TOKENS,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},  # templates without the switch ignore it
        }
        reply = _post(server, "/v1/chat/completions", translate(server.engine, body), timeout=600)
        summary = (reply["choices"][0]["message"].get("content") or "").strip() or summary
    if not summary:
        raise HTTPException(502, "The model returned an empty summary.")
    return summary


@router.post("/prepare")
def prepare(req: Prepare):
    """The messages to send for the next reply, compacting the thread first if it won't fit.

    Returns `messages` (with the system prompt and any summary), `summary` (new or unchanged),
    `compacted`, and the token counts: `prompt`, `context` and `reserve` (room kept for the reply).
    """
    if manager.state != "ready":
        raise HTTPException(409, "No model is loaded. Load a model from the Models page first.")
    server = manager
    context = _context(server)
    reserve = min(req.max_tokens, context // 4)
    summary = req.summary if req.summary and 0 < req.summary.upto <= len(req.messages) else None
    upto = summary.upto if summary else 0

    system = with_instructions(req.system, req.project) if req.instructions else req.system

    def build(upto: int, text: str | None) -> list[dict]:
        return _with_system(system, text) + _sendable(req.messages[upto:])

    messages = build(upto, summary.text if summary else None)
    prompt = count(server, messages)
    if req.check:
        return {"fits": prompt + reserve <= context, "prompt": prompt, "context": context, "reserve": reserve}
    compacted = False
    if prompt + reserve > context and req.compact:
        # Keep the most recent turns that fit in KEEP_SHARE of the context, starting on a user turn.
        keep, kept = len(req.messages) - 1, count(server, _sendable(req.messages[-1:]))
        while keep > upto and kept + (size := count(server, _sendable(req.messages[keep - 1 : keep]))) <= context * KEEP_SHARE:
            keep, kept = keep - 1, kept + size
        while keep < len(req.messages) - 1 and req.messages[keep].get("role") != "user":
            keep += 1
        if keep > upto:
            text = summarise(server, summary.text if summary else None, _sendable(req.messages[upto:keep]))
            summary, compacted = Summary(upto=keep, text=text), True
            messages = build(keep, text)
            prompt = count(server, messages)
    if prompt + reserve > context:
        last = count(server, _sendable(req.messages[-1:]))
        what = "This message" if last + reserve > context else "This conversation"
        hint = "Attach less, or load the model with a longer context." if what == "This message" else "Turn on auto-compaction in the Generation panel, or start a new chat."
        raise HTTPException(413, f"{what} needs about {prompt:,} tokens plus {reserve:,} for the reply, more than the model's {context:,}-token context. {hint}")
    return {"messages": messages, "summary": summary.model_dump() if summary else None, "compacted": compacted, "prompt": prompt, "context": context, "reserve": reserve}
