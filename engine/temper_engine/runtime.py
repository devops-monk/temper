"""Inference runtime: supervises a llama.cpp server and exposes an OpenAI-compatible /v1 API."""

from __future__ import annotations

import re
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

UNKNOWN_ARCH = re.compile(r"unknown model architecture: '([^']+)'")


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
    # A vision or audio projector; by default an mmproj*.gguf beside the model, if there is one.
    mmproj: str | None = None


PROJECTOR_PREFERENCE = ("f16", "bf16", "f32", "q8")


def find_projector(model: str) -> str | None:
    """The mmproj GGUF beside a model (as Hugging Face repos ship them), preferring 16-bit over quantized ones."""
    folder = Path(model).parent
    found = sorted((p for p in folder.glob("*.gguf") if "mmproj" in p.name.lower() and p.is_file()), key=lambda p: p.name)
    if not found:
        return None
    for tag in PROJECTOR_PREFERENCE:
        for p in found:
            if re.search(rf"(^|[-_.]){tag}([-_.]|$)", p.name.lower()):
                return str(p)
    return str(found[0])


def reasoning_support(props: dict) -> dict:
    """What the loaded model's chat template allows, from llama-server's /props.

    `thinks`: the model reasons before answering (llama-server streams that as reasoning_content).
    `toggle`: thinking can be switched off per request (the template reads `enable_thinking`).
    `effort`: the template takes a reasoning effort level (gpt-oss and others).
    """
    template = props.get("chat_template") or ""
    caps = props.get("chat_template_caps") or {}
    toggle = "enable_thinking" in template
    effort = bool(caps.get("supports_reasoning_effort")) or "reasoning_effort" in template
    thinks = toggle or effort or any(tag in template for tag in ("<think>", "reasoning_content", "<|channel|>analysis", "thinking"))
    return {"thinks": thinks, "toggle": toggle, "effort": effort}


class RuntimeManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.model: str | None = None
        self.path: str | None = None
        self.state = "idle"  # idle | loading | ready | failed
        self.error: str | None = None
        self.log: list[str] = []
        self.reasoning: dict | None = None
        self.projector: str | None = None
        # What the model takes besides text, from llama-server's /props: {"vision": bool, "audio": bool}.
        self.modalities: dict | None = None
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
            "reasoning": self.reasoning,
            "projector": Path(self.projector).name if self.projector else None,
            "modalities": self.modalities,
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
            self.reasoning, self.modalities = None, None
            self.projector = req.mmproj or find_projector(req.path)
            cmd = [
                binary, "-m", req.path, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(req.context), "-ngl", str(req.gpu_layers), "--jinja",
            ]
            if self.projector:
                cmd += ["--mmproj", self.projector]
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
            arch = next((m.group(1) for line in self.log if (m := UNKNOWN_ARCH.search(line))), None)
            reason = f"llama.cpp can't run {self.model}: it doesn't support the {arch} architecture." if arch else f"llama-server exited with code {code}"
            self.error = f"{reason}\n{tail}"

    def _wait_ready(self, proc: subprocess.Popen) -> None:
        deadline = time.time() + 600
        while time.time() < deadline and proc.poll() is None and proc is self.proc:
            try:
                if httpx.get(f"{self.base}/health", timeout=1).status_code == 200:
                    try:
                        props = httpx.get(f"{self.base}/props", timeout=5).json()
                        self.reasoning = reasoning_support(props)
                        self.modalities = {k: bool(v) for k, v in (props.get("modalities") or {}).items()}
                    except (httpx.HTTPError, ValueError):
                        self.reasoning, self.modalities = None, None
                    self.state = "ready"
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)

    def unload(self) -> None:
        with self.lock:
            proc, self.proc = self.proc, None
            self.state, self.model, self.path, self.error, self.reasoning = "idle", None, None, None, None
            self.projector, self.modalities = None, None
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
    from .compare import pool

    pool.unload_all()  # compare mode's models would compete with it for memory
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
    from .compare import pool

    body = await request.json()
    # Compare mode's models are asked for by path or name; anything else goes to chat's model.
    server = pool.find(body.get("model"))
    if server is None:
        _require_ready()
        server = manager
    elif server.state != "ready":
        raise HTTPException(409, f"{server.model} is still loading." if server.state == "loading" else server.error or f"{server.model} isn't loaded.")
    if isinstance(body.get("messages"), list):
        from . import attachments

        kinds = attachments.used(body["messages"])
        takes = server.modalities or {}
        for kind, modality, what in (("image", "vision", "see images"), ("audio", "audio", "hear audio")):
            if kind in kinds and not takes.get(modality):
                hint = " Its projector (an mmproj .gguf) wasn't found beside it." if not server.projector else ""
                raise HTTPException(400, f"{server.model} can't {what}.{hint} Load a model that can, or remove the attachment.")
        body["messages"] = attachments.expand(body["messages"])
    url = f"{server.base}/v1/chat/completions"
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
