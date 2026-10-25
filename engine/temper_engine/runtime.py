"""Inference runtime: supervises a llama.cpp server and exposes an OpenAI-compatible /v1 API."""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .hardware import llama_server_path

router = APIRouter(tags=["runtime"])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LoadRequest(BaseModel):
    path: str
    # What to call it; Ollama's blobs are named by their hash.
    name: str | None = None
    context: int = 8192
    gpu_layers: int = 999
    flash_attention: bool = True


class RuntimeManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.model: str | None = None
        self.path: str | None = None
        self.state = "idle"  # idle | loading | ready | failed
        self.error: str | None = None
        self.log: list[str] = []
        self.lock = threading.Lock()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def status(self) -> dict:
        return {
            "state": self.state,
            "model": self.model,
            "path": self.path,
            "error": self.error,
            "engine": "llama.cpp",
            "available": llama_server_path() is not None,
        }

    def load(self, req: LoadRequest) -> None:
        binary = llama_server_path()
        if not binary:
            raise HTTPException(
                409, "llama.cpp is not installed. Install it from the Models page and try again."
            )
        if not Path(req.path).is_file():
            raise HTTPException(404, f"Model file not found: {req.path}")
        self.unload()
        with self.lock:
            self.port = _free_port()
            self.model, self.path, self.state, self.error, self.log = req.name or Path(req.path).name, req.path, "loading", None, []
            cmd = [
                binary, "-m", req.path, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(req.context), "-ngl", str(req.gpu_layers), "--jinja",
            ]
            if req.flash_attention:
                cmd += ["-fa", "on"]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()

    def _watch(self, proc: subprocess.Popen) -> None:
        assert proc.stdout
        threading.Thread(target=self._wait_ready, args=(proc,), daemon=True).start()
        for line in proc.stdout:
            self.log = (self.log + [line.rstrip()])[-200:]
        code = proc.wait()
        if proc is self.proc and self.state != "idle":
            self.state = "failed"
            tail = "\n".join(self.log[-8:])
            self.error = f"llama-server exited with code {code}\n{tail}"

    def _wait_ready(self, proc: subprocess.Popen) -> None:
        deadline = time.time() + 600
        while time.time() < deadline and proc.poll() is None and proc is self.proc:
            try:
                if httpx.get(f"{self.base}/health", timeout=1).status_code == 200:
                    self.state = "ready"
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)

    def unload(self) -> None:
        with self.lock:
            proc, self.proc = self.proc, None
            self.state, self.model, self.path, self.error = "idle", None, None, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


manager = RuntimeManager()


@router.get("/api/runtime")
def runtime_status():
    return manager.status()


@router.post("/api/runtime/load")
def runtime_load(req: LoadRequest):
    manager.load(req)
    return manager.status()


@router.post("/api/runtime/unload")
def runtime_unload():
    manager.unload()
    return manager.status()


def _require_ready() -> None:
    if manager.state != "ready":
        raise HTTPException(409, "No model is loaded. Load a model from the Models page first.")


@router.get("/v1/models")
def v1_models():
    data = [{"id": manager.model, "object": "model", "owned_by": "temper"}] if manager.state == "ready" else []
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
async def v1_chat(request: Request):
    _require_ready()
    body = await request.json()
    url = f"{manager.base}/v1/chat/completions"
    if not body.get("stream"):
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=body)
            return JSONResponse(r.json(), status_code=r.status_code)

    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(client.build_request("POST", url, json=body), stream=True)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code, media_type="text/event-stream")
