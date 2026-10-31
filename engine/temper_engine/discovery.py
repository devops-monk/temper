"""Engine discovery: Temper answering at the addresses other tools look for local models.

Apps that support Ollama look at localhost:11434, and llama.cpp-aware ones probe a llama-server's
/health and /props. When turned on (Connect → Other apps), Temper listens there on this computer only
and answers as they would: Ollama's API (/api/tags, /api/chat, /api/generate, ...) and llama-server's
probes, all served by the model loaded in Temper, with the OpenAI-compatible /v1 passed through.

Like Ollama, these addresses take no key, which is why they're off by default and only ever on
127.0.0.1. Ollama's own port is only taken when Ollama itself isn't running.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import prefs, usage
from .lan import Background

router = APIRouter(prefix="/api/discovery", tags=["runtime"])

PORTS = {"ollama": 11434, "llamacpp": 8080}
OLLAMA_VERSION = "0.40.0"  # what Ollama clients see; recent enough for them to use tools and images
token = ""  # the engine's own key, for passing /v1 through (set by create_app)


def _servers() -> list:
    from .compare import pool
    from .runtime import manager

    return [s for s in (manager, *pool.servers.values()) if s.state == "ready"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _details(server) -> dict:
    name = Path(server.path or server.model or "").name
    quant = next((p for p in name.replace(".gguf", "").split("-")[::-1] if p.upper().startswith(("Q", "IQ", "F16", "BF16"))), "")
    return {"format": "gguf" if name.endswith(".gguf") else "safetensors", "family": "", "families": None, "parameter_size": "", "quantization_level": quant.upper()}


def _size(server) -> int:
    try:
        return Path(server.path).stat().st_size if server.path and Path(server.path).is_file() else 0
    except OSError:
        return 0


def _tag(server) -> dict:
    # Ollama's clients expect a digest (its CLI prints the first 12 characters): one made from the path, stable per model.
    digest = hashlib.sha256((server.path or server.model or "").encode()).hexdigest()
    return {"name": server.model, "model": server.model, "modified_at": _now(), "size": _size(server), "digest": digest, "details": _details(server)}


def _options(body: dict) -> dict:
    """Ollama's request settings as OpenAI ones."""
    o = body.get("options") or {}
    out: dict = {}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("seed", "seed"), ("stop", "stop"), ("num_predict", "max_tokens"),
                     ("repeat_penalty", "repeat_penalty"), ("presence_penalty", "presence_penalty"), ("frequency_penalty", "frequency_penalty")):
        if o.get(src) is not None:
            out[dst] = o[src]
    if out.get("max_tokens", 0) < 0:
        out.pop("max_tokens")  # -1 is "until done"
    fmt = body.get("format")
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": fmt}}
    return out


def _message(m: dict) -> dict:
    """An Ollama chat message as an OpenAI one: images become image parts, tool calls get ids."""
    out: dict = {"role": m.get("role", "user"), "content": m.get("content") or ""}
    if m.get("images"):
        parts = [{"type": "text", "text": out["content"]}] if out["content"] else []
        for img in m["images"]:
            mime = "image/png" if img.startswith("iVBOR") else "image/jpeg"  # a PNG's signature, in base64
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img}"}})
        out["content"] = parts
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"].get("arguments") or {})}}
            for i, c in enumerate(m["tool_calls"])
        ]
    if m.get("role") == "tool" and m.get("tool_name"):
        out["name"] = m["tool_name"]
    return out


def to_openai(body: dict, chat: bool = True) -> dict:
    if chat:
        messages = [_message(m) for m in body.get("messages") or []]
    else:
        messages = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + [_message({"role": "user", "content": body.get("prompt", ""), "images": body.get("images")})]
    out = {"messages": messages, "stream": body.get("stream", True) is not False, **_options(body)}
    if body.get("model"):
        out["model"] = body["model"]
    if body.get("tools"):
        out["tools"] = body["tools"]
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out


def _calls(calls: list[dict] | None) -> list[dict]:
    out = []
    for c in calls or []:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def _done(model: str, started: float, counts: dict, reason: str | None, chat: bool) -> dict:
    took = int((time.time() - started) * 1e9)
    return {"model": model, "created_at": _now(), **({"message": {"role": "assistant", "content": ""}} if chat else {"response": ""}), "done": True,
            "done_reason": {"length": "length"}.get(reason or "", "stop"), "total_duration": took, "load_duration": 0,
            "prompt_eval_count": counts.get("prompt_tokens", 0), "eval_count": counts.get("completion_tokens", 0), "eval_duration": took}


async def _answer(body: dict, chat: bool):
    from .anthropic_api import ThinkFilter, strip_thinking
    from .runtime import aroute

    model = body.get("model") or ""
    try:
        _server, url, req = await aroute(to_openai(body, chat))
    except HTTPException as e:
        return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)
    started = time.time()
    if not req["stream"]:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=req, headers=getattr(_server, "headers", None))
        if r.status_code != 200:
            return JSONResponse({"error": r.text[:500]}, status_code=r.status_code)
        reply = r.json()
        choice = (reply.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = strip_thinking(msg.get("content") or "")
        out = _done(model, started, reply.get("usage") or {}, choice.get("finish_reason"), chat)
        usage.add(_server.model, out["prompt_eval_count"], out["eval_count"])
        if chat:
            out["message"] = {"role": "assistant", "content": text, **({"tool_calls": _calls(msg.get("tool_calls"))} if msg.get("tool_calls") else {})}
        else:
            out["response"] = text
        return JSONResponse(out)

    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(client.build_request("POST", url, json=req, headers=getattr(_server, "headers", None)), stream=True)

    async def lines():
        think = ThinkFilter()
        counts, reason, calls = {}, None, {}
        try:
            async for line in upstream.aiter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                counts = chunk.get("usage") or counts
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reason = choice.get("finish_reason") or reason
                    for c in delta.get("tool_calls") or []:  # gathered whole: Ollama sends each call in one piece
                        slot = calls.setdefault(c.get("index", 0), {"function": {"name": "", "arguments": ""}})
                        fn = c.get("function") or {}
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
                    if text := think.feed(delta.get("content") or ""):
                        part = {"message": {"role": "assistant", "content": text}} if chat else {"response": text}
                        yield json.dumps({"model": model, "created_at": _now(), **part, "done": False}) + "\n"
            if text := think.flush():
                part = {"message": {"role": "assistant", "content": text}} if chat else {"response": text}
                yield json.dumps({"model": model, "created_at": _now(), **part, "done": False}) + "\n"
            if calls and chat:
                yield json.dumps({"model": model, "created_at": _now(), "message": {"role": "assistant", "content": "", "tool_calls": _calls([calls[i] for i in sorted(calls)])}, "done": False}) + "\n"
            usage.add(_server.model, counts.get("prompt_tokens"), counts.get("completion_tokens"))
            yield json.dumps(_done(model, started, counts, reason, chat)) + "\n"
        finally:
            await upstream.aclose()
            await client.aclose()

    if upstream.status_code != 200:
        text = (await upstream.aread()).decode(errors="replace")
        await upstream.aclose()
        await client.aclose()
        return JSONResponse({"error": text[:500]}, status_code=upstream.status_code)
    return StreamingResponse(lines(), media_type="application/x-ndjson")


def compat_app() -> FastAPI:
    """What answers on the discovery ports: Ollama's API, llama-server's probes and /v1 passed through."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    @app.head("/")
    def root():
        return PlainTextResponse("Ollama is running")  # what Ollama clients check for

    @app.get("/api/version")
    def version():
        return {"version": OLLAMA_VERSION}

    @app.get("/api/tags")
    def tags():
        return {"models": [_tag(s) for s in _servers()]}

    @app.get("/api/ps")
    def ps():
        return {"models": [{**_tag(s), "size_vram": _size(s), "expires_at": "2099-01-01T00:00:00Z", "context_length": s.context or 0} for s in _servers()]}

    @app.post("/api/show")
    async def show(request: Request):
        body = await request.json()
        name = body.get("model") or body.get("name")
        server = next((s for s in _servers() if s.model == name), None) or next(iter(_servers()), None)
        if server is None:
            return JSONResponse({"error": f"model '{name}' not found"}, status_code=404)
        caps = ["completion", "tools"] + (["vision"] if (server.modalities or {}).get("vision") else []) + (["thinking"] if server.reasoning else [])
        return {"modelfile": f"FROM {server.path}\n", "parameters": "", "template": "", "details": _details(server),
                "model_info": {"general.architecture": "", "general.context_length": server.context or 0}, "capabilities": caps, "modified_at": _now()}

    @app.post("/api/chat")
    async def chat(request: Request):
        return await _answer(await request.json(), chat=True)

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        if not body.get("prompt") and not body.get("images"):  # Ollama's way of loading a model: say it's ready
            return {"model": body.get("model", ""), "created_at": _now(), "response": "", "done": True, "done_reason": "load"}
        return await _answer(body, chat=False)

    @app.get("/health")
    def health():
        return {"status": "ok"} if _servers() else JSONResponse({"status": "no model loaded"}, status_code=503)

    @app.get("/props")
    def props():
        server = next(iter(_servers()), None)
        if server is None:
            return JSONResponse({"error": "no model loaded"}, status_code=503)
        return {"default_generation_settings": {"n_ctx": server.context or 0}, "total_slots": 1, "model_path": server.path, "modalities": server.modalities or {}, "build_info": "temper"}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def v1(path: str, request: Request):
        engine = f"http://127.0.0.1:{os.environ.get('TEMPER_PORT', '8765')}/v1/{path}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": request.headers.get("content-type", "application/json")}
        client = httpx.AsyncClient(timeout=None)
        upstream = await client.send(client.build_request(request.method, engine, content=await request.body(), headers=headers, params=request.query_params), stream=True)

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        kind = upstream.headers.get("content-type", "application/json")
        if "event-stream" not in kind:
            body = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            return Response(body, status_code=upstream.status_code, media_type=kind)
        return StreamingResponse(relay(), status_code=upstream.status_code, media_type=kind)

    return app


listeners = {name: Background(f"{name}-discovery") for name in PORTS}
_app: FastAPI | None = None


def _ollama_running() -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{PORTS['ollama']}/api/version", timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


def apply() -> None:
    """Starts or stops each discovery listener to match the saved settings."""
    global _app
    cfg = prefs.load().get("discovery") or {}
    for name, listener in listeners.items():
        want = bool(cfg.get(name))
        if want and not listener.running:
            if name == "ollama" and _ollama_running():
                listener.error = "Ollama itself is running, so its address is taken. Quit Ollama to let Temper answer there."
                continue
            _app = _app or compat_app()
            listener.start(_app, "127.0.0.1", int(cfg.get(f"{name}_port") or PORTS[name]))
        elif not want and listener.server:
            listener.stop()
            listener.error = None


def stop_all() -> None:
    for listener in listeners.values():
        listener.stop()


def status() -> dict:
    cfg = prefs.load().get("discovery") or {}
    return {name: {"enabled": bool(cfg.get(name)), "running": listeners[name].running, "port": int(cfg.get(f"{name}_port") or PORTS[name]), "error": listeners[name].error}
            for name in PORTS}


class DiscoverySettings(BaseModel):
    ollama: bool | None = None
    llamacpp: bool | None = None
    llamacpp_port: int | None = Field(None, ge=1024, le=65535)  # llama-server's 8080 is a busy port


@router.get("")
def get_discovery():
    return status()


@router.post("")
def set_discovery(req: DiscoverySettings):
    cfg = dict(prefs.load().get("discovery") or {})
    cfg.update({k: v for k, v in req.model_dump().items() if v is not None})
    prefs.update("discovery", cfg)
    if req.llamacpp_port is not None and listeners["llamacpp"].running and listeners["llamacpp"].port != req.llamacpp_port:
        listeners["llamacpp"].stop()  # to start again on the new port
    apply()
    out = status()
    failed = [n for n in PORTS if out[n]["enabled"] and not out[n]["running"]]
    if failed:
        cfg.update({n: False for n in failed})  # leave it off rather than on and broken
        prefs.update("discovery", cfg)
        out = status()
        raise HTTPException(409, " ".join(listeners[n].error or f"Couldn't listen on port {out[n]['port']}." for n in failed))
    return out
