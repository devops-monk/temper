"""An Anthropic-compatible API (`/v1/messages`) over the loaded model, so Claude Code and other apps built
on Anthropic's SDK can use a local model.

Requests are translated to the OpenAI shape the model servers take, and replies (whole or streamed) back
into Anthropic's: text, tool use and tool results, images, stop reasons and token counts. A model name
Temper doesn't know (such as the claude-* names Claude Code asks for) goes to chat's model; compare mode's
models answer to their own names. A thinking model's reasoning is left out of the reply.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(tags=["runtime"])

PING_SECONDS = 10
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "function_call": "tool_use"}


def _text(content) -> str:
    """The text of a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def _part(block: dict) -> dict | None:
    """A user content block as an OpenAI content part."""
    if block.get("type") == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block.get("type") == "image":
        src = block.get("source") or {}
        url = src.get("url") if src.get("type") == "url" else f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return {"type": "image_url", "image_url": {"url": url}}
    return None


def to_openai(body: dict) -> dict:
    """An Anthropic Messages request as an OpenAI chat completions one."""
    messages: list[dict] = []
    if system := _text(body.get("system")):
        messages.append({"role": "system", "content": system})
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "assistant":
            calls = [
                {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                for b in content if b.get("type") == "tool_use"
            ]
            msg: dict = {"role": "assistant", "content": _text(content)}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
            continue
        # Tool results become tool messages, straight after the assistant turn that asked for them.
        for b in content:
            if b.get("type") == "tool_result":
                result = b.get("content")
                text = _text(result) if isinstance(result, list) else str(result or "")
                messages.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": ("Error: " if b.get("is_error") else "") + text})
        parts = [p for b in content if (p := _part(b))]
        if parts:
            only_text = all(p["type"] == "text" for p in parts)
            messages.append({"role": "user", "content": "".join(p["text"] for p in parts) if only_text else parts})

    out: dict = {"messages": messages, "max_tokens": body.get("max_tokens") or 4096, "stream": bool(body.get("stream"))}
    if body.get("model"):
        out["model"] = body["model"]
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop")):
        if body.get(src) is not None:
            out[dst] = body[src]
    if tools := body.get("tools"):
        out["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema") or {"type": "object"}}}
            for t in tools if t.get("name") and "input_schema" in t  # server tools (web search and such) aren't ours to run
        ] or None
        if not out["tools"]:
            out.pop("tools")
    choice = body.get("tool_choice") or {}
    if out.get("tools") and choice.get("type"):
        out["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}.get(choice["type"]) or {"type": "function", "function": {"name": choice.get("name")}}
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out


def _arguments(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {}


def strip_thinking(text: str) -> str:
    """A reply without a leading <think>…</think> block (an unfinished one leaves nothing)."""
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return text
    end = stripped.find("</think>")
    return "" if end < 0 else stripped[end + len("</think>"):].lstrip()


def from_openai(reply: dict, model: str) -> dict:
    """An OpenAI chat completion as an Anthropic message."""
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    if text := strip_thinking(msg.get("content") or ""):
        content.append({"type": "text", "text": text})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        content.append({"type": "tool_use", "id": call.get("id") or _id("toolu"), "name": fn.get("name", ""), "input": _arguments(fn.get("arguments", ""))})
    usage = reply.get("usage") or {}
    stop = "tool_use" if msg.get("tool_calls") else STOP_REASONS.get(choice.get("finish_reason") or "stop", "end_turn")
    return {
        "id": _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class ThinkFilter:
    """Drops a leading <think>…</think> block from streamed text, holding back only what might be one."""

    def __init__(self):
        self.buffer = ""
        self.done = False

    def feed(self, text: str) -> str:
        if self.done:
            return text
        self.buffer += text
        head = self.buffer.lstrip()
        if not head or "<think>".startswith(head):
            return ""  # can't tell yet
        if not head.startswith("<think>"):
            self.done = True
            return self.buffer
        end = head.find("</think>")
        if end < 0:
            return ""
        self.done = True
        return head[end + len("</think>"):].lstrip()

    def flush(self) -> str:
        return "" if self.done or self.buffer.lstrip().startswith("<think>") else self.buffer


async def translate_stream(lines, model: str):
    """OpenAI stream lines (`data: {...}`) as Anthropic stream events."""
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": _id("msg"), "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
    }})
    index = -1
    open_kind: str | None = None  # "text", or "tool:<openai index>"
    think = ThinkFilter()
    finish, usage = None, {}

    def close():
        nonlocal open_kind
        if open_kind is None:
            return b""
        open_kind = None
        return _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    def text_events(text: str):
        nonlocal index, open_kind
        out = b""
        if not text:
            return out
        if open_kind != "text":
            out += close()
            index += 1
            open_kind = "text"
            out += _sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
        return out + _sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}})

    async for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if events := text_events(think.feed(delta.get("content") or "")):
                yield events
            for call in delta.get("tool_calls") or []:
                kind = f"tool:{call.get('index', 0)}"
                fn = call.get("function") or {}
                if open_kind != kind:
                    yield close()
                    index += 1
                    open_kind = kind
                    block = {"type": "tool_use", "id": call.get("id") or _id("toolu"), "name": fn.get("name", ""), "input": {}}
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})
                if fn.get("arguments"):
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if events := text_events(think.flush()):
        yield events
    yield close()
    stop = STOP_REASONS.get(finish or "stop", "end_turn")
    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                 "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})


async def with_pings(events, every: float = PING_SECONDS):
    """`events`, with a ping whenever none has come for `every` seconds: reading a long prompt (Claude
    Code's is tens of thousands of tokens) can take minutes, and clients take silence for a stall."""
    it = events.__aiter__()
    pending = asyncio.ensure_future(it.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=every)
            if not done:
                yield _sse("ping", {"type": "ping"})
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            yield event
            pending = asyncio.ensure_future(it.__anext__())
    finally:
        if not pending.done():
            pending.cancel()


def _error(status: int, message: str) -> JSONResponse:
    kind = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error", 409: "invalid_request_error", 429: "rate_limit_error"}.get(status, "api_error")
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


@router.post("/v1/messages")
async def v1_messages(request: Request):
    from .runtime import aroute

    from . import api_keys

    body = await request.json()
    model = body.get("model") or "temper"
    entry = getattr(request.state, "api_entry", None)
    try:
        server, url, req = await aroute(to_openai(body))
    except HTTPException as e:
        return _error(e.status_code, str(e.detail))
    api_keys.record_prompt(entry, req.get("messages"))
    if not req.get("stream"):
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=req, headers=getattr(server, "headers", None))
        if r.status_code != 200:
            return _error(r.status_code, r.text[:500])
        msg = from_openai(r.json(), model)
        api_keys.tokens(entry, server.model, msg["usage"]["input_tokens"], msg["usage"]["output_tokens"])
        api_keys.record_reply(entry, "".join(b.get("text", "") for b in msg["content"] if b.get("type") == "text"))
        return JSONResponse(msg)

    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(client.build_request("POST", url, json=req, headers=getattr(server, "headers", None)), stream=True)
    if upstream.status_code != 200:
        text = (await upstream.aread()).decode(errors="replace")
        await upstream.aclose()
        await client.aclose()
        return _error(upstream.status_code, text[:500])

    if entry is not None:
        entry["model"], entry["streaming"] = server.model, True

    async def watched(lines):
        async for line in lines:
            api_keys.observe(entry, line)
            yield line

    async def relay():
        try:
            async for event in with_pings(translate_stream(watched(upstream.aiter_lines()), model)):
                if event.startswith(b"event: message_delta"):
                    usage = json.loads(event.split(b"data: ", 1)[1])["usage"]
                    api_keys.tokens(entry, server.model, usage["input_tokens"], usage["output_tokens"])
                if event:
                    yield event
        finally:
            await upstream.aclose()
            await client.aclose()
            api_keys.done(entry)

    return StreamingResponse(relay(), media_type="text/event-stream")


@router.post("/v1/messages/count_tokens")
async def v1_count_tokens(request: Request):
    """An estimate (about four characters a token): the servers don't all tokenize on request."""
    body = to_openai(await request.json())
    chars = len(json.dumps(body["messages"])) + len(json.dumps(body.get("tools") or []))
    return {"input_tokens": max(1, chars // 4)}
