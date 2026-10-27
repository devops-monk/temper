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
from fastapi.responses import JSONResponse
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


# Fixes worth suggesting only when they free at least this share of what's missing.
SIGNIFICANT = 0.25


def _model_max(server: RuntimeManager) -> int | None:
    """The longest context the loaded model was trained for."""
    from . import engines, model_settings

    if server.folder:
        return engines.context_max(server.folder)
    meta = model_settings._meta(server.path) if server.path else {}
    return meta.get(f"{meta.get('general.architecture')}.context_length")


def explain(server: RuntimeManager, req: Prepare, system: str, summary: Summary | None, prompt: int, reserve: int, context: int) -> dict:
    """Why a prompt doesn't fit, and what would make it: where the tokens go, and fixes with their numbers.

    `parts`: [{kind, tokens, name?}] for the instructions (system prompt, AGENTS.md, summary), the earlier
    turns, the new message and each file attached to it, and the room kept for the reply.
    `fixes`: [{kind, ...}], biggest win first; the app words them.
    """
    upto = summary.upto if summary else 0
    sendable = _sendable(req.messages[upto:])
    instructions = count(server, _with_system(system, summary.text if summary else None)) if system or summary else 0
    instruction_files = count(server, [{"role": "system", "content": t}]) if (t := with_instructions("", req.project)) else 0
    last = sendable[-1:] or [{"role": "user", "content": ""}]
    content = last[0].get("content")
    files = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "attachment":
                try:
                    name = attachments.meta(str(part.get("id", "")))["name"]
                except (OSError, ValueError, HTTPException):
                    continue
                files.append({"kind": "attachment", "name": name, "tokens": count(server, [{"role": "user", "content": [part]}])})
    message = max(count(server, last) - sum(f["tokens"] for f in files), 0)
    earlier = max(prompt - instructions - message - sum(f["tokens"] for f in files), 0)
    parts = [{"kind": "instructions", "tokens": instructions}, {"kind": "earlier", "tokens": earlier}, {"kind": "message", "tokens": message}, *files, {"kind": "reply", "tokens": reserve}]

    over = prompt + reserve - context
    fixes: list[dict] = []
    # Leaving less room for the reply fixes it outright when the prompt alone fits with a useful reply.
    if (room := context - prompt) >= min(256, reserve // 4) and room < reserve:
        fixes.append({"kind": "max_tokens", "to": room - room % 64 if room >= 128 else room})
    for f in sorted(files, key=lambda f: -f["tokens"]):
        if f["tokens"] >= over * SIGNIFICANT:
            fixes.append({"kind": "remove_attachment", "name": f["name"], "tokens": f["tokens"]})
    if earlier >= over * SIGNIFICANT and len(sendable) > 1:
        fixes.append({"kind": "compact" if not req.compact else "new_chat", "tokens": earlier})
    if instructions >= over * SIGNIFICANT:
        fixes.append({"kind": "shorten_instructions", "tokens": instructions, "files": instruction_files})
    model_max = _model_max(server)
    need = prompt + reserve
    if model_max and model_max > context:
        fixes.append({"kind": "longer_context", "to": min(model_max, 1 << max(need - 1, 1).bit_length()), "max": model_max})
    return {"context": context, "prompt": prompt, "reserve": reserve, "over": over, "parts": parts, "fixes": fixes}


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
        fit = explain(server, req, system, summary, prompt, reserve, context)
        last = next(p["tokens"] for p in fit["parts"] if p["kind"] == "message") + sum(p["tokens"] for p in fit["parts"] if p["kind"] == "attachment")
        what = "This message" if last + reserve > context else "This conversation"
        hint = "Attach less, or load the model with a longer context." if what == "This message" else "Turn on auto-compaction in the Generation panel, or start a new chat."
        detail = f"{what} needs about {prompt:,} tokens plus {reserve:,} for the reply, more than the model's {context:,}-token context. {hint}"
        return JSONResponse({"detail": detail, "fit": fit}, status_code=413)
    return {"messages": messages, "summary": summary.model_dump() if summary else None, "compacted": compacted, "prompt": prompt, "context": context, "reserve": reserve}
