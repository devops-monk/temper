"""Inference engines besides llama.cpp: MLX on Apple Silicon, and opt-in Transformers, vLLM and SGLang.

Each runs as its own OpenAI-compatible server that runtime.RuntimeManager starts and proxies, like
llama-server. MLX comes with the engine on Apple Silicon (mlx-lm is in the "mlx" profile) and can
load a LoRA adapter straight from a fine-tuning run. Transformers, vLLM and SGLang are installed on
request, each into its own virtual environment under engines/ in Temper's home, so their PyTorch
pins never touch the training stack. vLLM and SGLang need Linux and an NVIDIA or AMD GPU, and split a
model across GPUs with tensor and pipeline parallelism.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import hardware, home, prefs
from .jobs import Job, registry

router = APIRouter(prefix="/api/runtime/engines", tags=["runtime"])

APPLE = platform.system() == "Darwin" and platform.machine() == "arm64"
LINUX = platform.system() == "Linux"

ENGINES: dict[str, dict] = {
    "llama.cpp": {"name": "llama.cpp", "formats": ["gguf"]},
    "mlx": {"name": "MLX", "formats": ["mlx", "hf"], "module": "mlx_lm"},
    "transformers": {"name": "Transformers", "formats": ["hf"], "packages": ["transformers[serving]", "torch", "accelerate"], "module": "transformers"},
    "vllm": {"name": "vLLM", "formats": ["hf"], "packages": ["vllm"], "module": "vllm", "linux_gpu": True},
    "sglang": {"name": "SGLang", "formats": ["hf"], "packages": ["sglang[all]"], "module": "sglang", "linux_gpu": True},
}
# What the proxy sends as the request's model: the name each server answers to.
SERVED = "temper"


def venv(engine: str) -> Path:
    return home() / "engines" / engine


def _venv_python(folder: Path) -> Path:
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def python_for(engine: str) -> str | None:
    """The Python that runs `engine`'s server, or None if it isn't installed."""
    if engine == "mlx":
        return sys.executable if APPLE and hardware.has_module("mlx_lm") else None
    python = _venv_python(venv(engine))
    if python.is_file() and (venv(engine) / ".installed").is_file():
        return str(python)
    return None


def supported(engine: str) -> str | None:
    """Why `engine` can't run on this computer, or None if it can."""
    if engine == "mlx" and not APPLE:
        return "MLX runs on Apple Silicon Macs."
    if ENGINES[engine].get("linux_gpu") and not (LINUX and (hardware.nvidia_gpus() or hardware.amd_gpus())):
        return f"{ENGINES[engine]['name']} needs Linux with an NVIDIA or AMD GPU."
    return None


def version(engine: str) -> str | None:
    """The installed version: llama.cpp's build, or the engine package's version from its environment."""
    if engine == "llama.cpp":
        from . import llamacpp

        if llamacpp.installed_build():
            return llamacpp.TAG
        server = hardware.llama_server_path()
        if not server:
            return None
        try:  # one found on PATH or from Homebrew: "version: 6570 (1b2c3d4)"
            out = subprocess.run([server, "--version"], capture_output=True, text=True, timeout=10)
            line = next((ln for ln in (out.stdout + out.stderr).splitlines() if ln.startswith("version:")), "")
            return "b" + line.split()[1] if line else "unknown"
        except (OSError, subprocess.SubprocessError, IndexError):
            return "unknown"
    module = ENGINES[engine]["module"]
    dist = module.replace("_", "-")
    if engine == "mlx":
        from importlib import metadata

        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None
    site = "Lib/site-packages" if os.name == "nt" else "lib/python*/site-packages"
    for found in venv(engine).glob(f"{site}/{module}-*.dist-info"):
        return found.name[len(module) + 1:-len(".dist-info")]
    return None


def preferred() -> str | None:
    """The engine picked for model folders (`temper engines use`), when it's still installed."""
    engine = prefs.load().get("engine")
    return engine if engine in ENGINES and engine != "llama.cpp" and not supported(engine) and python_for(engine) else None


def available() -> list[dict]:
    out = []
    chosen = preferred()
    for engine, info in ENGINES.items():
        if engine == "llama.cpp":
            ready, why = hardware.llama_server_path() is not None, None
        else:
            why = supported(engine)
            ready = why is None and python_for(engine) is not None
        out.append({"id": engine, "name": info["name"], "available": ready, "unsupported": why, "installable": bool(info.get("packages")) and not why,
                    "formats": info["formats"], "version": version(engine) if ready else None, "preferred": engine == chosen})
    return out


# --- Models ---------------------------------------------------------------------------------------


def local_folder(path: str) -> str | None:
    """A model folder, or a Hugging Face id's copy in the local cache; None for a GGUF file or a model that isn't here."""
    p = Path(path)
    if p.is_dir():
        return str(p)
    if p.exists() or "/" not in path or path.startswith(("/", ".", "~")):
        return None
    from . import storage

    # The cached snapshot that main points at, else any with a config (training only fetches the files it needs).
    repo = storage.models_dir() / ("models--" + path.replace("/", "--"))
    ref = repo / "refs" / "main"
    snapshots = [repo / "snapshots" / ref.read_text().strip()] if ref.is_file() else []
    snapshots += sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
    return next((str(s) for s in snapshots if (s / "config.json").is_file()), None)


def model_format(path: str) -> str:
    """"gguf" for a GGUF file, "mlx" for an MLX folder (quantized by mlx-lm), "hf" for a Hugging Face folder."""
    p = Path(path)
    if p.is_file():
        return "gguf"
    config = _read_json(p / "config.json")
    return "mlx" if "quantization" in config and not config.get("quantization_config") else "hf"


def default_engine(path: str) -> str:
    fmt = model_format(path)
    if fmt == "gguf":
        return "llama.cpp"
    if (chosen := preferred()) and fmt in ENGINES[chosen]["formats"]:
        return chosen
    for engine in ("mlx", "vllm", "sglang", "transformers"):
        if fmt in ENGINES[engine]["formats"] and not supported(engine) and python_for(engine):
            return engine
    return "mlx" if fmt == "mlx" else "transformers"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def chat_template(folder: str) -> str:
    """The folder's chat template (tokenizer_config.json or chat_template.jinja), for reasoning_support."""
    f = Path(folder)
    if (f / "chat_template.jinja").is_file():
        return (f / "chat_template.jinja").read_text(errors="replace")
    template = _read_json(f / "tokenizer_config.json").get("chat_template") or ""
    return template if isinstance(template, str) else json.dumps(template)


def chat_model(folder: str) -> bool:
    """A text-generating model with a chat template, as opposed to speech, embedding or audio models."""
    archs = _read_json(Path(folder) / "config.json").get("architectures") or []
    generates = any(str(a).endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs)
    return generates and bool(chat_template(folder).strip())


def context_max(folder: str) -> int | None:
    config = _read_json(Path(folder) / "config.json")
    return config.get("max_position_embeddings") or (config.get("text_config") or {}).get("max_position_embeddings")


def command(engine: str, path: str, settings: dict, port: int, adapter: str | None = None) -> list[str]:
    python = python_for(engine)
    if not python:
        why = supported(engine)
        raise HTTPException(409, why or f"{ENGINES[engine]['name']} isn't installed. Install it in Settings → Engines, then load the model again.")
    host = ["--host", "127.0.0.1", "--port", str(port)]
    if engine == "mlx":
        cmd = [python, "-m", "mlx_lm", "server", "--model", path, *host]
        if adapter:
            cmd += ["--adapter-path", adapter]
        if settings["cache_type"] != "f16":
            cmd += ["--kv-bits", "8" if settings["cache_type"] == "q8_0" else "4"]
        return cmd
    if engine == "transformers":
        return [python, "-m", "transformers", "serve", path, *host, "--device", "auto", "--dtype", "auto"]
    parallel = {"tp": settings.get("tensor_parallel") or 1, "pp": settings.get("pipeline_parallel") or 1}
    if engine == "vllm":
        cmd = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", path, *host, "--served-model-name", SERVED, "--max-model-len", str(settings["context"]),
               "--tensor-parallel-size", str(parallel["tp"]), "--pipeline-parallel-size", str(parallel["pp"])]
        if settings["cache_type"] != "f16":
            cmd += ["--kv-cache-dtype", "fp8"]
        if adapter:
            cmd += ["--enable-lora", "--lora-modules", f"{SERVED}={adapter}"]
        return cmd
    if engine == "sglang":
        cmd = [python, "-m", "sglang.launch_server", "--model-path", path, *host, "--served-model-name", SERVED, "--context-length", str(settings["context"]),
               "--tp-size", str(parallel["tp"]), "--pp-size", str(parallel["pp"])]
        if settings["cache_type"] != "f16":
            cmd += ["--kv-cache-dtype", "fp8_e5m2"]
        if adapter:
            cmd += ["--lora-paths", f"{SERVED}={adapter}"]
        return cmd
    raise HTTPException(400, f"Unknown engine {engine}.")


# Fields only llama-server understands, and the names other servers use for the rest.
LLAMA_ONLY = ("thinking_budget_tokens", "cache_prompt", "n_probs", "slot_id", "id_slot")
RENAMES = {"repeat_penalty": "repetition_penalty"}


def translate(engine: str, body: dict) -> dict:
    """A chat request as `engine`'s server takes it."""
    if engine == "llama.cpp":
        return body
    out = {RENAMES.get(k, k): v for k, v in body.items() if k not in LLAMA_ONLY}
    # mlx-lm loads whatever `model` names; the others answer to SERVED.
    if engine == "mlx":
        out.pop("model", None)
    else:
        out["model"] = SERVED
    if engine in ("vllm", "sglang", "transformers") and out.get("repetition_penalty") == 0:
        out.pop("repetition_penalty")
    return out


# --- Installing -----------------------------------------------------------------------------------


def _install(job: Job, engine: str) -> None:
    if install_venv(job, venv(engine), ENGINES[engine]["packages"], ENGINES[engine]["name"]):
        job.finish("done", engine=engine)


def install_venv(job: Job, folder: Path, packages: list[str], name: str) -> bool:
    """Makes a Python 3.12 venv at `folder` with `packages`, logging to `job`, and marks it with .installed.
    False if the job was cancelled (and is finished as such)."""
    from .llamacpp import uv_binary

    uv = uv_binary()
    (folder / ".installed").unlink(missing_ok=True)
    steps = [
        [uv, "venv", "--python", "3.12", "--allow-existing", str(folder)],
        [uv, "pip", "install", "--python", str(_venv_python(folder)), *packages],
    ]
    for i, cmd in enumerate(steps):
        job.log("$ " + " ".join(Path(c).name if j == 0 else c for j, c in enumerate(cmd)))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        job.process = proc
        assert proc.stdout
        for line in proc.stdout:
            if line.strip():
                job.log(line.rstrip())
        if proc.wait() != 0:
            if job.cancel_requested:
                job.finish("cancelled")
                return False
            raise RuntimeError(f"Installing {name} failed; the log says why.")
        job.progress = (i + 1) / len(steps)
        job.emit("progress", value=job.progress)
    (folder / ".installed").write_text(" ".join(packages))
    return True


@router.get("")
def list_engines():
    return {"engines": available()}


class Preferred(BaseModel):
    engine: str | None = None


@router.put("/preferred")
def set_preferred(req: Preferred):
    """The engine to load model folders with; None goes back to picking by format (MLX first on a Mac)."""
    if req.engine is not None:
        if req.engine not in ENGINES or req.engine == "llama.cpp":
            raise HTTPException(400, "Pick MLX, Transformers, vLLM or SGLang; GGUF files always load on llama.cpp.")
        if why := supported(req.engine):
            raise HTTPException(409, why)
        if not python_for(req.engine):
            raise HTTPException(409, f"{ENGINES[req.engine]['name']} isn't installed. `temper engines install {req.engine}` installs it.")
    prefs.update("engine", req.engine)
    return {"engines": available()}


@router.post("/{engine}/install")
def install(engine: str):
    if engine not in ENGINES or not ENGINES[engine].get("packages"):
        raise HTTPException(404, "There's nothing to install for that engine.")
    if why := supported(engine):
        raise HTTPException(409, why)
    job = registry.create("engine-install", f"Install {ENGINES[engine]['name']}")
    registry.run_thread(job, lambda j: _install(j, engine))
    return job.summary()


@router.delete("/{engine}")
def remove(engine: str):
    import shutil

    if engine not in ENGINES or not ENGINES[engine].get("packages"):
        raise HTTPException(404, "That engine can't be removed.")
    from .runtime import manager

    if manager.engine == engine and manager.state != "idle":
        raise HTTPException(409, f"A model is running on {ENGINES[engine]['name']}. Unload it first.")
    shutil.rmtree(venv(engine), ignore_errors=True)
    return {"engines": available()}
