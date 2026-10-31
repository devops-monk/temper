"""Load on demand: an API request naming a downloaded model that isn't loaded loads it first.

The model loads alongside chat's (loaded.py) and unloads after its own idle time: the request's `ttl`
in seconds, or the idle time in Settings. When memory is short, the models loaded this way that went
unused longest unload to make room; models someone loaded by hand never do.
"""

from __future__ import annotations

import struct
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from . import memory, prefs
from .model_names import pick
from .runtime import LoadRequest, manager

LOAD_TIMEOUT = 15 * 60
# GGUFs llama.cpp reads that aren't chat models: speech recognition, embeddings, vision encoders.
NOT_CHAT = {"parakeet", "whisper", "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "modern-bert", "neo-bert", "clip", "t5encoder", "wavtokenizer-dec"}
_lock = threading.Lock()  # one load at a time, so two requests for a model don't load it twice


def enabled() -> bool:
    return prefs.load().get("load_on_demand", True) is not False


def loaded(model: str | None) -> bool:
    from .compare import pool

    return not model or model in (manager.path, manager.model) and manager.state != "idle" or pool.find(model) is not None


def _chat(path: str) -> bool:
    """Whether a file can answer chat requests. Only a GGUF whose header names another kind of model is ruled
    out; anything else is left to the engine to judge when it loads."""
    if not path.lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return memory.read_gguf_metadata(f.read).get("general.architecture") not in NOT_CHAT
    except (OSError, ValueError, struct.error):
        return True


def _local() -> list[dict]:
    from .hub import local_models

    return [m for m in local_models() if not m.get("missing")]


def resolve(model: str) -> LoadRequest | None:
    """What to load for a request's `model`, or None when it isn't a whole name of a model on this computer."""
    try:
        path, name = pick(_local(), model, loose=False)
    except LookupError:
        return None
    return LoadRequest(path=path, name=name) if _chat(path) else None


def _make_room(req: LoadRequest) -> None:
    """Unloads models loaded on demand, least recently used first, until `req` fits. Holds pool.lock."""
    from .compare import _need, Model, pool
    from .loaded import MAX_LOADED

    need = _need(Model(path=req.path, name=req.name), req.context) if Path(req.path).is_file() else 0
    for path in sorted(pool.on_demand, key=lambda p: pool.servers[p].used if p in pool.servers else 0):
        if len(pool.kept) + 1 < MAX_LOADED and (not need or need <= memory.memory_budget()["free"] * 0.9):
            return
        pool.drop(path)
    if len(pool.kept) + 1 >= MAX_LOADED:
        raise HTTPException(409, f"{MAX_LOADED} models are loaded already. Unload one to load {req.name}.")
    free = memory.memory_budget()["free"]
    if need and need > free * 0.9:
        raise HTTPException(409, f"{req.name} needs about {need / memory.GB:.1f} GB and {free / memory.GB:.1f} GB is free. Unload a model to load it.")


def ensure(model: str | None, ttl: int | None = None) -> None:
    """Loads the model a request names when it's on this computer but not loaded, and waits until it's ready.

    Does nothing for a loaded model, an unknown name (the request goes to chat's model as before) or when
    load on demand is off. Raises an HTTPException when the load fails or times out.
    """
    from .compare import pool

    if loaded(model) or not enabled():
        return
    req = resolve(model)
    if req is None or req.path == manager.path and manager.state != "idle":
        return
    with _lock:
        server = pool.servers.get(req.path)
        if server is None:
            with pool.lock:
                _make_room(req)
                pool.drop(req.path)
                pool.start(req, keep=True)
                pool.on_demand.add(req.path)
                server = pool.servers[req.path]
        pool.aliases[model] = req.path
        if ttl is not None:
            pool.ttl[req.path] = max(0, int(ttl))
        deadline = time.time() + LOAD_TIMEOUT
        while server.state == "loading" and time.time() < deadline:
            time.sleep(0.25)
    if server.state != "ready":
        with pool.lock:
            pool.drop(req.path)
        raise HTTPException(503, server.error or f"{req.name} didn't finish loading.")
    server.used = time.time()


def listing() -> list[dict]:
    """The models /v1/models offers besides the loaded ones: everything on this computer that could load."""
    if not enabled():
        return []
    out = []
    for m in _local():
        try:
            path, _ = pick([m], m["id"], loose=False)
        except LookupError:
            continue
        if not _chat(path):
            continue
        out.append({"id": m["id"], "object": "model", "owned_by": "temper", "loaded": False})
    return out
