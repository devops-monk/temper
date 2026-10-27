"""Several models loaded at once: chat's model (runtime.manager) plus others loaded alongside it.

Every loaded model answers on /v1 by its name or path, so an agent can use a small model for quick
tasks and a big one for hard ones without reloading. Chat talks to runtime.manager; switching chat to
an alongside model swaps the two servers rather than reloading either. Models unload after the idle
time set here (Settings), or when asked.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import memory, prefs
from .compare import _need, Model, pool
from .runtime import LoadRequest, manager

router = APIRouter(prefix="/api/runtime/loaded", tags=["runtime"])

DEFAULT_IDLE_MINUTES = 60
MAX_LOADED = 6


def idle_seconds() -> int:
    """How long a model may go unused before it unloads; 0 keeps models loaded until they're unloaded."""
    minutes = prefs.load().get("idle_unload_minutes")
    return int((DEFAULT_IDLE_MINUTES if minutes is None else minutes) * 60)


def listing() -> dict:
    chat = [{**manager.status(), "chat": True}] if manager.state != "idle" else []
    return {"models": chat + [{**s, "chat": False} for s in pool.status(kept=True)], "idle_minutes": idle_seconds() // 60}


def load_alongside(req: LoadRequest) -> None:
    """Loads `req` next to chat's model, if they fit in memory together."""
    if req.path == manager.path or req.path in pool.kept:
        return
    with pool.lock:
        if len(pool.kept) + 1 >= MAX_LOADED:
            raise HTTPException(409, f"{MAX_LOADED} models are loaded already. Unload one first.")
        pool.keep_only(set())  # compare mode's servers would compete for memory
        pool.drop(req.path)
        need = _need(Model(path=req.path, name=req.name), req.context) if Path(req.path).is_file() else 0
        budget = memory.memory_budget()
        if need and need > budget["free"] * 0.9:
            name = req.name or Path(req.path).name
            raise HTTPException(409, f"{name} needs about {need / memory.GB:.1f} GB and {budget['free'] / memory.GB:.1f} GB is free. Unload a model, or load it in place of chat's.")
        pool.start(req, keep=True)


def switch(path: str) -> None:
    """Makes an alongside model chat's; chat's model stays loaded alongside."""
    with pool.lock:
        server = pool.servers.get(path)
        if server is None or path not in pool.kept:
            raise HTTPException(404, "That model isn't loaded alongside chat's.")
        pool.servers.pop(path)
        pool.used.pop(path, None)
        pool.kept.discard(path)
        manager.swap(server)
        if server.path and server.state != "idle":
            pool.adopt(server.path, server)


@router.get("")
def loaded():
    return listing()


class Target(BaseModel):
    path: str


@router.post("/switch")
def switch_route(req: Target):
    switch(req.path)
    return listing()


@router.post("/unload")
def unload(req: Target):
    if req.path == manager.path:
        manager.unload()
        ready = [p for p in pool.kept if pool.servers[p].state == "ready"]
        if ready:  # chat moves to a model that's still loaded
            switch(ready[0])
    else:
        with pool.lock:
            pool.drop(req.path)
    return listing()


class Idle(BaseModel):
    minutes: int = Field(ge=0, le=24 * 60)


@router.put("/idle")
def set_idle(req: Idle):
    prefs.update("idle_unload_minutes", req.minutes)
    return listing()
