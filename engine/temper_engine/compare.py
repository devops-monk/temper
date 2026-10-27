"""Compare mode: extra llama.cpp servers, so one prompt can go to several models side by side.

Chat's own model stays in runtime.manager; the others get a server each here, and /v1/chat/completions
routes a request to one by its `model` (the GGUF's path or name). When the models don't fit in memory
together, the app compares them one at a time: it loads one, asks it, then loads the next.
Servers here stop after IDLE_SECONDS without a request, when compare mode closes, or when chat loads a model.

The pool also holds models loaded alongside chat's (loaded.py). Those are "kept": compare mode reuses
them but never unloads them, and they stop only when asked or after the idle time set in Settings.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import memory
from .runtime import LoadRequest, RuntimeManager, manager

router = APIRouter(prefix="/api/compare", tags=["chat"])

IDLE_SECONDS = 10 * 60
MAX_MODELS = 4


class Pool:
    def __init__(self) -> None:
        self.servers: dict[str, RuntimeManager] = {}  # by GGUF path
        self.used: dict[str, float] = {}
        self.kept: set[str] = set()  # loaded alongside chat's model, not for compare mode
        self.lock = threading.Lock()
        self.reaper: threading.Thread | None = None

    def find(self, model: str | None) -> RuntimeManager | None:
        if not model:
            return None
        for path, server in self.servers.items():
            if model in (path, server.model):
                self.used[path] = server.used = time.time()
                return server
        return None

    def keep_only(self, paths: set[str]) -> None:
        """Unloads compare mode's servers that aren't in `paths`; kept ones stay."""
        for path in [p for p in self.servers if p not in paths and p not in self.kept]:
            self.drop(path)

    def drop(self, path: str) -> None:
        server = self.servers.pop(path, None)
        self.used.pop(path, None)
        self.kept.discard(path)
        if server:
            server.unload()

    def start(self, req: LoadRequest, keep: bool = False) -> None:
        server = RuntimeManager()
        server.load(req)
        self.servers[req.path] = server
        self.used[req.path] = time.time()
        if keep:
            self.kept.add(req.path)
        self._start_reaper()

    def adopt(self, path: str, server: RuntimeManager) -> None:
        """Takes a loaded server (chat's previous model, after a swap) as a kept one."""
        self.servers[path] = server
        self.used[path] = server.used
        self.kept.add(path)
        self._start_reaper()

    def _start_reaper(self) -> None:
        if not self.reaper:
            self.reaper = threading.Thread(target=self._reap, daemon=True)
            self.reaper.start()

    def unload_all(self) -> None:
        with self.lock:
            self.keep_only(set())

    def _reap(self) -> None:
        while True:
            time.sleep(30)
            self.reap()

    def reap(self, now: float | None = None) -> None:
        """Unloads models idle too long: compare mode's after IDLE_SECONDS, the rest after the idle time in Settings."""
        from .loaded import idle_seconds

        now = now or time.time()
        with self.lock:
            kept_idle = idle_seconds()
            for path in list(self.servers):
                limit = kept_idle if path in self.kept else IDLE_SECONDS
                if limit and now - self.servers[path].used > limit:
                    self.drop(path)
            if kept_idle and manager.state == "ready" and now - manager.used > kept_idle:
                manager.unload()

    def status(self, kept: bool | None = None) -> list[dict]:
        """Every server's status, or only the kept (True) or compare mode's (False) ones."""
        return [server.status() for path, server in self.servers.items() if kept is None or (path in self.kept) == kept]


pool = Pool()


def loaded_paths() -> list[str]:
    """Every GGUF a llama.cpp server has open: chat's model and compare mode's."""
    return [p for p in [manager.path, *pool.servers] if p]


class Model(BaseModel):
    path: str
    name: str | None = None


class CompareLoad(BaseModel):
    models: list[Model] = Field(min_length=1, max_length=MAX_MODELS)
    context: int | None = None  # unset: each model's saved settings


def _need(model: Model, context: int | None) -> int:
    """Memory the model needs; 0 when its header can't be read, so llama.cpp tries it and says what's wrong."""
    try:
        return memory.local_chat_estimate(Path(model.path), context)["total"]
    except (OSError, ValueError, LookupError, TypeError, ZeroDivisionError):
        return 0


@router.get("")
def compare_status():
    return {"servers": pool.status()}


@router.post("/load")
def compare_load(req: CompareLoad):
    """Loads the models chat hasn't, if they fit in memory together; otherwise loads nothing and says so.

    `together`: every model is (being) loaded and can answer at once. If false, the app loads them one
    at a time, calling this with a single model.
    """
    wanted = {m.path: m for m in req.models if m.path != manager.path and m.path not in pool.kept}
    for m in wanted.values():
        if not Path(m.path).is_file():
            raise HTTPException(404, f"Model file not found: {m.path}")
    with pool.lock:
        pool.keep_only(set(wanted))  # frees memory before measuring what's left
        new = [m for p, m in wanted.items() if p not in pool.servers]
        needs = [(m, _need(m, req.context)) for m in new]
        budget = memory.memory_budget()
        if len(wanted) > 1 and sum(n for _, n in needs) > budget["free"] * 0.9:
            return {"together": False, "servers": pool.status()}
        for m, need in needs:
            if memory.verdict(need, budget) == "too_large":
                name = m.name or Path(m.path).name
                hint = " Unload the chat model to make room." if manager.path else ""
                raise HTTPException(409, f"{name} needs about {need / memory.GB:.1f} GB, more than this computer can give it.{hint}")
        for m, _ in needs:
            pool.start(LoadRequest(path=m.path, name=m.name, context=req.context))
    return {"together": True, "servers": pool.status()}


@router.post("/unload")
def compare_unload():
    pool.unload_all()
    return {"servers": []}
