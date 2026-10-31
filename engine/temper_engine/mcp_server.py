"""Temper as an MCP server: other agents can use this computer's models and see its jobs.

The engine answers MCP's Streamable HTTP transport at /mcp, behind the same token as the rest of
its API (sent as `Authorization: Bearer <token>`). It answers each request with JSON and doesn't
push anything of its own, so GET has no event stream.

Tools: what's loaded, the models on this computer, loading and unloading one, asking the loaded
model (or one loaded alongside it), typed decisions (decisions.py), and listing, following and cancelling jobs (downloads,
training, exports).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import __version__

router = APIRouter(tags=["mcp"])

PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
LOG_LINES = 20


def _status(_: dict) -> Any:
    from .loaded import listing
    from .runtime import manager

    s = manager.status()
    keep = ("state", "model", "path", "engine", "context", "error")
    return {"chat": {k: s.get(k) for k in keep}, "alongside": [{k: m.get(k) for k in keep} for m in listing()["models"] if not m["chat"]]}


def _models(args: dict) -> Any:
    from .hub import local_models

    query = str(args.get("query") or "").lower()
    out = []
    for m in local_models():
        if m.get("missing"):
            continue
        files = [{"name": g["name"], "path": g["path"], "size_gb": round((g.get("size") or 0) / 1e9, 2)} for g in m.get("gguf") or [] if g.get("complete", True)]
        if not files and not m.get("chat") and m.get("kind") not in ("mlx", "safetensors"):
            continue
        entry = {"name": m["id"], "files": files} if files else {"name": m["id"], "path": m["path"], "folder": True}
        if not query or query in json.dumps(entry).lower():
            out.append(entry)
    return out or "No models found. Download one in Temper first."


def _load(args: dict) -> Any:
    from .runtime import LoadRequest, runtime_load

    path = args.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("Give the `path` of a model file or folder, from list_models.")
    # A Hugging Face snapshot folder is named by its commit; call it by its repo instead.
    repo = re.search(r"models--[^/\\]+--([^/\\]+)[/\\]snapshots[/\\]", path)
    req = {"path": path, "name": repo.group(1) if repo else None, "context": int(args["context"]) if args.get("context") else None, "alongside": bool(args.get("alongside")), "engine": args.get("engine")}
    status = runtime_load(LoadRequest(**{k: v for k, v in req.items() if v is not None}))
    return {"state": status["state"], "model": status["model"], "note": "Loading takes a few seconds to a minute; temper_status shows when it's ready."}


def _unload(_: dict) -> Any:
    from .runtime import runtime_unload

    return {"state": runtime_unload()["state"]}


def _ask(args: dict) -> Any:
    from .runtime import complete

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Give the `prompt` to send.")
    messages = ([{"role": "system", "content": args["system"]}] if args.get("system") else []) + [{"role": "user", "content": prompt}]
    temperature = float(args["temperature"]) if args.get("temperature") is not None else None
    return complete(messages, int(args.get("max_tokens") or 1024), temperature, args.get("model")) or "(The model wrote nothing.)"


def _decide(args: dict) -> Any:
    from .decisions import Decision, decide, get

    if args.get("saved"):
        d = get(str(args["saved"]))
    else:
        d = Decision(**{k: args[k] for k in ("question", "kind", "choices", "scale", "instructions", "model") if args.get(k) is not None})
    return decide(d, args.get("text"))


def _jobs(args: dict) -> Any:
    from .jobs import list_jobs

    jobs = list_jobs(args.get("kind"))[: int(args.get("limit") or 20)]
    return [{k: j[k] for k in ("id", "kind", "title", "status", "progress", "error")} for j in jobs] or "No jobs yet."


def _job(args: dict) -> Any:
    from .jobs import registry

    job = registry.get(str(args.get("id", "")))
    logs = [e["line"] for e in job.events if e["event"] == "log"][-LOG_LINES:]
    return {**job.summary(), "log": logs}


def _cancel(args: dict) -> Any:
    from .jobs import cancel_job

    return cancel_job(str(args.get("id", "")))


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, **({"required": required} if required else {})}


TOOLS = {
    "temper_status": ("What Temper has loaded: chat's model and any loaded alongside it.", _schema(), _status),
    "list_models": ("The models downloaded or imported on this computer, with the paths to load them by.", _schema({"query": {"type": "string", "description": "Only models whose name or file has this in it"}}), _models),
    "load_model": (
        "Loads a model for chat (replacing the current one), or alongside it. Use a path from list_models.",
        _schema({"path": {"type": "string"}, "context": {"type": "integer", "description": "Context length in tokens"}, "alongside": {"type": "boolean", "description": "Keep the current model loaded too"}}, ["path"]),
        _load,
    ),
    "unload_model": ("Unloads chat's model, freeing its memory.", _schema(), _unload),
    "ask_model": (
        "Sends a prompt to a model running on this computer and returns its answer. Uses chat's model unless `model` names one loaded alongside.",
        _schema(
            {
                "prompt": {"type": "string"},
                "system": {"type": "string", "description": "A system prompt"},
                "max_tokens": {"type": "integer", "description": "Longest answer, in tokens (default 1024)"},
                "temperature": {"type": "number"},
                "model": {"type": "string", "description": "The path or name of a model loaded alongside"},
            },
            ["prompt"],
        ),
        _ask,
    ),
    "decide": (
        "Asks a local model for a typed answer about some text: yes/no, one of `choices`, a score on `scale`, or the `labels` (from `choices`) that apply; "
        "with a one-sentence reason. Or runs a decision saved in Temper by its name (`saved`).",
        _schema(
            {
                "question": {"type": "string", "description": "Such as: Is this email asking for a refund?"},
                "text": {"type": "string", "description": "What the question is about"},
                "kind": {"type": "string", "enum": ["yes_no", "choice", "score", "labels"]},
                "choices": {"type": "array", "items": {"type": "string"}},
                "scale": {"type": "array", "items": {"type": "integer"}, "description": "[low, high], default [1, 10]"},
                "instructions": {"type": "string", "description": "How to decide: definitions, examples, edge cases"},
                "saved": {"type": "string", "description": "The name of a saved decision; then only `text` is needed"},
                "model": {"type": "string"},
            }
        ),
        _decide,
    ),
    "list_jobs": ("Temper's jobs (downloads, training runs, exports and installs), newest first.", _schema({"kind": {"type": "string", "description": "download, train, export, …"}, "limit": {"type": "integer"}}), _jobs),
    "get_job": ("One job's status, progress and latest log lines.", _schema({"id": {"type": "string"}}, ["id"]), _job),
    "cancel_job": ("Cancels a running job.", _schema({"id": {"type": "string"}}, ["id"]), _cancel),
}
# Tools that only read; the others change what's loaded or running.
READ_ONLY = {"temper_status", "list_models", "decide", "list_jobs", "get_job"}


def _call(name: str, args: dict) -> dict:
    if name not in TOOLS:
        raise KeyError(name)
    try:
        out = TOOLS[name][2](args)
    except HTTPException as e:
        return {"content": [{"type": "text", "text": str(e.detail)}], "isError": True}
    except (ValueError, TypeError, httpx.HTTPError) as e:
        return {"content": [{"type": "text", "text": str(e)}], "isError": True}
    text = out if isinstance(out, str) else json.dumps(out, indent=2, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], **({"structuredContent": out if isinstance(out, dict) else {"items": out}} if not isinstance(out, str) else {})}


def handle(message: dict) -> dict | None:
    """The answer to one JSON-RPC message; None for notifications and answers."""
    if not isinstance(message, dict) or "method" not in message:
        return None
    rid, method, params = message.get("id"), message["method"], message.get("params") or {}
    if rid is None:
        return None

    def ok(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def fail(code: int, text: str) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": text}}

    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok({
            "protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "temper", "title": "Temper", "version": __version__},
            "instructions": "Temper runs and fine-tunes language models on this computer. Use list_models and load_model to choose one, then ask_model.",
        })
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": d, "inputSchema": s, "annotations": {"readOnlyHint": n in READ_ONLY, "openWorldHint": False}} for n, (d, s, _) in TOOLS.items()]})
    if method == "tools/call":
        try:
            return ok(_call(str(params.get("name")), params.get("arguments") or {}))
        except KeyError:
            return fail(-32602, f"There's no tool called {params.get('name')!r}.")
    return fail(-32601, f"Temper doesn't support {method}.")


@router.post("/mcp")
async def mcp_post(request: Request):
    from starlette.concurrency import run_in_threadpool

    try:
        message = await request.json()
    except ValueError:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "That isn't JSON."}}, status_code=400)
    batch = isinstance(message, list)
    answers = [a for m in (message if batch else [message]) if (a := await run_in_threadpool(handle, m))]
    if not answers:
        return Response(status_code=202)
    headers = {"Mcp-Session-Id": uuid.uuid4().hex} if not batch and message.get("method") == "initialize" else {}
    return JSONResponse(answers if batch else answers[0], headers=headers)


@router.get("/mcp")
def mcp_get():
    return Response(status_code=405, headers={"Allow": "POST"})


@router.delete("/mcp")
def mcp_delete():
    return Response(status_code=200)
