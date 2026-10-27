"""Chat replies that keep generating in the engine when the window closes or another chat is opened.

The chat page starts a generation here rather than streaming from /v1 itself. The engine streams from
the model, keeps every event, and saves the reply into the chat when it ends, so nothing is lost if
the page goes away. The page follows a generation's events, from the start when it reopens the chat.

With tools on, a generation runs rounds: when the model calls tools, the engine runs them, sends the
results back and streams the model's next round, until it answers without a call (tools.py).
Besides the model's own events, the page gets {"temper_tool": step} as a call starts and again with
its result, and {"temper_content": text} when a call written into the text was taken out of it.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import chats, tools
from .runtime import route

router = APIRouter(prefix="/api/chat/generations", tags=["chat"])

# Finished generations stay this long, so a page that reconnects late still gets the end.
KEEP_SECONDS = 120
EMPTY = "The model ended its reply without writing anything. Regenerate, or rephrase the message."


class Generation:
    def __init__(self, chat_id: str | None, upto: int, model: str | None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.chat_id, self.upto, self.model = chat_id, upto, model
        self.events: list[str] = []  # the model server's SSE data, as it sent them
        self.content = ""
        self.reasoning = ""
        self.tools: list[dict] = []  # the steps of the tools it called (tools.py)
        self.error: str | None = None
        self.cancelled = False
        self.done = False
        self.ended = 0.0
        self.task: asyncio.Task | None = None
        self.wake = asyncio.Event()

    def push(self, data: str | dict) -> None:
        data = data if isinstance(data, str) else json.dumps(data)
        self.events.append(data)
        wake, self.wake = self.wake, asyncio.Event()
        wake.set()

    def summary(self) -> dict:
        return {"id": self.id, "chat_id": self.chat_id, "upto": self.upto, "done": self.done}

    def reply(self) -> dict | None:
        """The assistant message to save; a reply stopped before it began is none."""
        if self.content or self.reasoning or self.tools:
            return {"role": "assistant", "content": self.content, "model": self.model, "error": False, "reasoning": self.reasoning or None, "attachments": None, "tools": self.tools or None}
        if self.cancelled:
            return None
        return {"role": "assistant", "content": self.error or EMPTY, "model": self.model, "error": True, "reasoning": None, "attachments": None}


live: dict[str, Generation] = {}


def for_chat(chat_id: str) -> Generation | None:
    return next((g for g in live.values() if g.chat_id == chat_id and not g.done), None)


def _forget_old() -> None:
    for gid in [gid for gid, g in live.items() if g.done and time.time() - g.ended > KEEP_SECONDS]:
        live.pop(gid)


def _save(gen: Generation) -> None:
    """Puts the reply after the first `upto` messages, as the page would have."""
    if not gen.chat_id:
        return
    reply = gen.reply()
    try:
        with chats.db() as con:
            existing = chats._with_messages(con, gen.chat_id)["messages"]
            chats.write_messages(con, gen.chat_id, existing[: gen.upto] + ([reply] if reply else []))
            con.execute("update chats set updated = ? where id = ?", (time.time(), gen.chat_id))
    except HTTPException:  # the chat was deleted meanwhile
        pass


def _absorb(gen: Generation, data: str, calls: dict[int, dict]) -> None:
    """Adds a streamed chunk to the reply; tool-call pieces are gathered into `calls` by index."""
    try:
        chunk = json.loads(data)
    except ValueError:
        return
    if isinstance(chunk.get("error"), dict | str):
        error = chunk["error"]
        gen.error = error.get("message", str(error)) if isinstance(error, dict) else error
        return
    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
    gen.content += delta.get("content") or ""
    gen.reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
    for piece in delta.get("tool_calls") or []:
        call = calls.setdefault(piece.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
        fn = piece.get("function") or {}
        call["id"] = piece.get("id") or call["id"]
        call["name"] += fn.get("name") or ""
        args = fn.get("arguments")
        call["arguments"] += args if isinstance(args, str) else json.dumps(args) if args else ""


async def _round(gen: Generation, client: httpx.AsyncClient, body: dict) -> list[tools.Call] | None:
    """Streams one round of the reply; returns the tool calls it made, or None if it failed."""
    _, url, body = route(body)
    calls: dict[int, dict] = {}
    async with client.stream("POST", url, json={**body, "stream": True}) as r:
        if r.status_code != 200:
            text = (await r.aread()).decode(errors="replace")
            try:
                detail = json.loads(text)
                gen.error = detail.get("detail") or (detail.get("error") or {}).get("message") or text
            except ValueError:
                gen.error = text or f"The model's server answered {r.status_code}."
            return None
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            _absorb(gen, data, calls)
            gen.push(data)
    return None if gen.error else [tools.Call(c["name"], c["arguments"], c["id"]) for _, c in sorted(calls.items())]


async def _run(gen: Generation, body: dict, names: list[str]) -> None:
    try:
        server, _, _ = route(dict(body))
        # llama-server takes the tools natively; other engines were offered them in the system prompt (compaction.py).
        native = server.engine == "llama.cpp"
        specs = tools.schemas(names)
        messages = list(body["messages"])
        made: dict[str, dict] = {}  # each distinct call's step, by tool and arguments
        looping = False
        async with httpx.AsyncClient(timeout=None) as client:
            for turn in range(tools.MAX_ROUNDS + 1):
                offer = bool(specs) and turn < tools.MAX_ROUNDS and not looping
                start = len(gen.content)
                calls = await _round(gen, client, {**body, "messages": messages, **({"tools": specs} if offer and native else {})})
                if calls is None or not specs:
                    return
                # Calls the model wrote into its text, in its own format, are taken out of the text and made.
                found, cleaned = tools.extract_calls(gen.content[start:], names)
                if found or cleaned != gen.content[start:]:
                    gen.content = gen.content[:start] + cleaned
                    gen.push({"temper_content": gen.content})
                calls = calls or found
                if not calls or not offer:
                    return
                steps, fresh = [], 0
                for call in calls:
                    call.id = call.id or f"call_{uuid.uuid4().hex[:10]}"
                    arguments = tools.parse_arguments(call.arguments)
                    # Small models repeat calls they've made; the repeat gets the earlier result instead of running again.
                    key = f"{tools.resolve_name(call.name, names)}:{json.dumps(arguments, sort_keys=True) if arguments is not None else call.arguments}"
                    if key in made:
                        steps.append({**made[key], "id": call.id, "result": f"{made[key]['result']}\n(The same call as before: answer from the results you have.)"})
                        continue
                    step = {"id": call.id, "name": tools.resolve_name(call.name, names) or call.name or "?", "arguments": call.arguments if arguments is None else arguments, "result": None, "error": False, "at": len(gen.content)}
                    gen.tools.append(step)
                    gen.push({"temper_tool": step})
                    step.update(await asyncio.to_thread(tools.run, call, names))
                    gen.push({"temper_tool": step})
                    steps.append(made.setdefault(key, step))
                    fresh += 1
                # A round of nothing but repeats is a loop: the next round is offered no tools, so the model answers.
                looping = not fresh
                messages += tools.round_messages(gen.content[start:], steps, native)
                if looping or turn + 1 >= tools.MAX_ROUNDS:  # the next round has no tools: say so with the last result
                    messages[-1] = {**messages[-1], "content": messages[-1]["content"] + "\n\n(No more tool calls in this reply: answer now from these results.)"}
    except asyncio.CancelledError:
        gen.cancelled = True
    except HTTPException as e:
        gen.error = str(e.detail)
    except httpx.HTTPError as e:
        gen.error = f"The model's server stopped answering: {e}"
    finally:
        await asyncio.to_thread(_save, gen)
        gen.done, gen.ended = True, time.time()
        gen.push(json.dumps({"temper_end": {"error": gen.error, "cancelled": gen.cancelled}}))


class Start(BaseModel):
    chat_id: str | None = None
    upto: int = 0  # the reply goes after the chat's first `upto` messages
    model: str | None = None  # the name saved with the reply
    body: dict  # an OpenAI chat request
    tools: list[str] = []  # the tools the reply may call (tools.py)


@router.post("")
async def start(req: Start):
    _forget_old()
    running = for_chat(req.chat_id) if req.chat_id else None
    if running and running.task:  # one reply at a time per chat
        running.task.cancel()
    gen = Generation(req.chat_id, req.upto, req.model)
    live[gen.id] = gen
    gen.task = asyncio.create_task(_run(gen, req.body, req.tools))
    return gen.summary()


@router.get("/{gen_id}/stream")
async def follow(gen_id: str, after: int = 0):
    """The generation's events from the `after`-th, then live ones until it ends (its last event is {"temper_end": …})."""
    gen = live.get(gen_id)
    if not gen:
        raise HTTPException(404, "That reply has finished and was saved to the chat.")

    async def events():
        sent = after
        while True:
            wake = gen.wake
            while sent < len(gen.events):
                yield f"data: {gen.events[sent]}\n\n"
                sent += 1
            if gen.done and sent >= len(gen.events):
                return
            try:
                await asyncio.wait_for(wake.wait(), timeout=15)
            except TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/{gen_id}/cancel")
async def cancel(gen_id: str):
    gen = live.get(gen_id)
    if gen and gen.task and not gen.done:
        gen.task.cancel()
        try:
            await gen.task
        except asyncio.CancelledError:
            pass
    return gen.summary() if gen else {"id": gen_id, "done": True}


def cancel_chat(chat_id: str) -> None:
    """Stops a deleted chat's reply; called from a worker thread, so the cancel goes through the event loop."""
    gen = for_chat(chat_id)
    if gen and gen.task:
        gen.task.get_loop().call_soon_threadsafe(gen.task.cancel)
