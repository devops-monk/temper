"""Inference runtime: supervises a model server (llama.cpp, or another engine in engines.py) and exposes an OpenAI-compatible /v1 API."""

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
    # Unset fields come from the model's saved settings (model_settings.py), then the defaults.
    context: int | None = None
    gpu_layers: int | None = None
    flash_attention: bool | None = None
    cache_type: str | None = None
    speculation: str | None = None
    draft: str | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    # A vision or audio projector; by default an mmproj*.gguf beside the model, if there is one.
    mmproj: str | None = None
    # llama.cpp for a GGUF; for a model folder (or a Hugging Face id in the local cache), the best installed engine.
    engine: str | None = None
    # A LoRA adapter folder from a fine-tuning run, applied on top (MLX, vLLM, SGLang).
    adapter: str | None = None
    # Load it next to chat's model rather than in its place (loaded.py).
    alongside: bool = False


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


# What a manager knows about its loaded model, exchanged by swap().
STATE = ("proc", "port", "model", "path", "state", "error", "log", "reasoning", "projector", "modalities", "context", "settings", "engine", "folder", "adapter", "used")


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
        self.context: int | None = None  # tokens, as llama-server set it up
        self.settings: dict | None = None  # what it was loaded with (model_settings.resolve)
        self.engine = "llama.cpp"
        self.folder: str | None = None  # a model folder's local copy, for its tokenizer and chat template
        self.adapter: str | None = None
        self.used = time.time()  # the last request, for unloading idle models
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
            "engine": self.engine,
            "adapter": self.adapter,
            "available": llama_server_path() is not None,
            "reasoning": self.reasoning,
            "projector": Path(self.projector).name if self.projector else None,
            "modalities": self.modalities,
            "context": self.context,
            "idle": round(time.time() - self.used) if self.state == "ready" else None,
            "settings": {**self.settings, "draft": Path(self.settings["draft"]).name if self.settings.get("draft") else None} if self.settings else None,
        }

    def load(self, req: LoadRequest) -> None:
        from . import engines, model_settings

        folder = engines.local_folder(req.path)
        engine = req.engine or engines.default_engine(folder or req.path)
        if engine not in engines.ENGINES:
            raise HTTPException(400, f"Unknown engine {engine}.")
        if engine != "llama.cpp":
            return self._load_other(req, engine, folder)
        binary = llama_server_path()
        if not binary:
            raise HTTPException(
                409, "llama.cpp is not installed. Install it from the Models page and try again."
            )
        if not Path(req.path).is_file():
            raise HTTPException(404, f"Model file not found: {req.path}")

        settings = model_settings.resolve(req.path, req.model_dump())
        model_settings.check(settings)
        self.unload()
        with self.lock:
            self._start(req, "llama.cpp", None)
            self.projector = req.mmproj or find_projector(req.path)
            self.settings = settings
            cmd = [
                binary, "-m", req.path, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(settings["context"]), "-ngl", str(settings["gpu_layers"]), "--jinja",
                "-fa", "on" if settings["flash_attention"] else "off",
                *model_settings.flags(settings),
            ]
            if self.projector:
                cmd += ["--mmproj", self.projector]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        self._supervise(self.proc)

    def _start(self, req: LoadRequest, engine: str, folder: str | None) -> None:
        self.port = _free_port()
        self.model, self.path, self.state, self.error, self.log = req.name or Path(req.path).name, req.path, "loading", None, []
        self.reasoning, self.modalities, self.context = None, None, None
        self.engine, self.folder, self.adapter, self.projector = engine, folder, None, None
        self.used = time.time()

    def _load_other(self, req: LoadRequest, engine: str, folder: str | None) -> None:
        """Starts an engine other than llama.cpp on a model folder (engines.py)."""
        from . import engines, model_settings

        if not folder:
            raise HTTPException(404, f"Model not found on this computer: {req.path}. Download it first.")
        if engines.model_format(folder) not in engines.ENGINES[engine]["formats"]:
            raise HTTPException(400, f"{engines.ENGINES[engine]['name']} can't load this model's format.")
        if req.adapter and not (Path(req.adapter) / "adapter_config.json").is_file():
            raise HTTPException(404, "The fine-tuned adapter isn't in that run's folder any more.")
        settings = model_settings.resolve(req.path, req.model_dump())
        cmd = engines.command(engine, folder, settings, 0, req.adapter)  # checks the engine is installed
        self.unload()
        with self.lock:
            self._start(req, engine, folder)
            self.settings, self.adapter = settings, req.adapter
            cmd = engines.command(engine, folder, settings, self.port, req.adapter)
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        self._supervise(self.proc)

    def _supervise(self, proc: subprocess.Popen) -> None:
        proc.owner = self  # type: ignore[attr-defined]  # swap() hands the server to another manager
        threading.Thread(target=RuntimeManager._watch, args=(proc,), daemon=True).start()

    @staticmethod
    def _watch(proc: subprocess.Popen) -> None:
        assert proc.stdout
        threading.Thread(target=RuntimeManager._wait_ready, args=(proc,), daemon=True).start()
        for line in proc.stdout:
            owner = proc.owner  # type: ignore[attr-defined]
            owner.log = (owner.log + [line.rstrip()])[-200:]
        code = proc.wait()
        self = proc.owner  # type: ignore[attr-defined]
        if proc is self.proc and self.state != "idle":
            self.state = "failed"
            tail = "\n".join(self.log[-8:])
            arch = next((m.group(1) for line in self.log if (m := UNKNOWN_ARCH.search(line))), None)
            name = "llama-server" if self.engine == "llama.cpp" else f"The {self.engine} server"
            reason = f"{self.engine} can't run {self.model}: it doesn't support the {arch} architecture." if arch else f"{name} exited with code {code}"
            self.error = f"{reason}\n{tail}"

    @staticmethod
    def _wait_ready(proc: subprocess.Popen) -> None:
        deadline = time.time() + 600
        while time.time() < deadline and proc.poll() is None and proc is proc.owner.proc:  # type: ignore[attr-defined]
            self = proc.owner  # type: ignore[attr-defined]
            try:
                if httpx.get(f"{self.base}/health", timeout=1).status_code == 200:
                    if self.engine != "llama.cpp":
                        self._ready_other()
                        return
                    try:
                        props = httpx.get(f"{self.base}/props", timeout=5).json()
                        self.reasoning = reasoning_support(props)
                        self.modalities = {k: bool(v) for k, v in (props.get("modalities") or {}).items()}
                        self.context = (props.get("default_generation_settings") or {}).get("n_ctx")
                    except (httpx.HTTPError, ValueError):
                        self.reasoning, self.modalities, self.context = None, None, None
                    self.state = "ready"
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)

    def _ready_other(self) -> None:
        from . import engines

        assert self.folder and self.settings
        self.reasoning = reasoning_support({"chat_template": engines.chat_template(self.folder)})
        self.modalities = {"vision": False, "audio": False}
        native = engines.context_max(self.folder)
        # vLLM and SGLang are started with the context; MLX and Transformers grow theirs up to the model's maximum.
        self.context = self.settings["context"] if self.engine in ("vllm", "sglang") else native or self.settings["context"]
        self.state = "ready"

    def swap(self, other: RuntimeManager) -> None:
        """Exchanges loaded models with `other`: how an alongside model becomes chat's without reloading."""
        with self.lock, other.lock:
            for key in STATE:
                mine, theirs = getattr(self, key), getattr(other, key)
                setattr(self, key, theirs)
                setattr(other, key, mine)
            for server in (self, other):
                if server.proc:
                    server.proc.owner = server  # type: ignore[attr-defined]

    def unload(self) -> None:
        with self.lock:
            proc, self.proc = self.proc, None
            self.state, self.model, self.path, self.error, self.reasoning = "idle", None, None, None, None
            self.projector, self.modalities, self.context, self.settings = None, None, None, None
            self.engine, self.folder, self.adapter = "llama.cpp", None, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


manager = RuntimeManager()


@router.get("/api/runtime")
def runtime_status():
    from .compare import pool

    alongside = [{k: s[k] for k in ("model", "path", "state", "engine", "error", "idle")} for s in pool.status(kept=True)]
    return {**manager.status(), "alongside": alongside}


@router.post("/api/runtime/load")
def runtime_load(req: LoadRequest):
    from . import loaded
    from .compare import pool

    if req.alongside and manager.state != "idle":
        loaded.load_alongside(req)
        return runtime_status()
    if req.path in pool.kept:  # already loaded alongside: make it chat's, keeping the current one
        loaded.switch(req.path)
        return runtime_status()
    pool.unload_all()  # compare mode's models would compete with it for memory
    manager.load(req)
    pool._start_reaper()  # unloads chat's model too once it's idle
    return runtime_status()


@router.post("/api/runtime/unload")
def runtime_unload():
    manager.unload()
    return manager.status()


def _require_ready() -> None:
    if manager.state != "ready":
        raise HTTPException(409, "No model is loaded. Load a model from the Models page first.")


@router.get("/v1/models")
def v1_models():
    from .compare import pool

    servers = [manager, *pool.servers.values()]
    data = [{"id": s.model, "object": "model", "owned_by": "temper"} for s in servers if s.state == "ready"]
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
    server.used = time.time()
    if isinstance(body.get("messages"), list):
        from . import attachments

        kinds = attachments.used(body["messages"])
        takes = server.modalities or {}
        for kind, modality, what in (("image", "vision", "see images"), ("audio", "audio", "hear audio")):
            if kind in kinds and not takes.get(modality):
                hint = " Its projector (an mmproj .gguf) wasn't found beside it." if server.engine == "llama.cpp" and not server.projector else ""
                raise HTTPException(400, f"{server.model} can't {what}.{hint} Load a model that can, or remove the attachment.")
        body["messages"] = attachments.expand(body["messages"])
    from .engines import translate

    body = translate(server.engine, body)
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
