"""Hardware probe: what can this machine run and train?"""

from __future__ import annotations

import functools
import importlib.util
import json
import platform
import shutil
import subprocess

import psutil
from fastapi import APIRouter

from . import home

router = APIRouter(prefix="/api/system", tags=["system"])

GB = 1024**3


def _run(cmd: list[str], timeout: float = 4) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@functools.cache
def cpu_name() -> str:
    if platform.system() == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name.strip()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def nvidia_gpus() -> list[dict]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in (out or "").strip().splitlines():
        idx, name, total, used, util, driver = [p.strip() for p in line.split(",")]
        gpus.append({
            "index": int(idx),
            "vendor": "nvidia",
            "name": name,
            "memory_total": int(float(total)) * 1024**2,
            "memory_used": int(float(used)) * 1024**2,
            "utilization": float(util) if util not in ("", "[N/A]") else None,
            "driver": driver,
            "backend": "cuda",
        })
    return gpus


def amd_gpus() -> list[dict]:
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    gpus = []
    for i, (card, info) in enumerate(sorted(data.items())):
        if not card.startswith("card"):
            continue
        gpus.append({
            "index": i,
            "vendor": "amd",
            "name": info.get("Card Series") or info.get("Card series") or card,
            "memory_total": int(info.get("VRAM Total Memory (B)", 0)),
            "memory_used": int(info.get("VRAM Total Used Memory (B)", 0)),
            "utilization": None,
            "driver": None,
            "backend": "rocm",
        })
    return gpus


def apple_gpu(ram_total: int, ram_used: int) -> list[dict]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    # Apple Silicon shares one pool of unified memory between CPU and GPU.
    return [{
        "index": 0,
        "vendor": "apple",
        "name": f"{cpu_name()} GPU",
        "memory_total": ram_total,
        "memory_used": ram_used,
        "utilization": None,
        "driver": platform.mac_ver()[0],
        "backend": "metal",
        "unified_memory": True,
    }]


def llama_server_path() -> str | None:
    from .llamacpp import binary

    return binary("llama-server")


# Python module of the GPU fine-tuning library that first-run setup installs.
GPU_TRAINER = "unsloth"


def probe_backend() -> str:
    """cuda, rocm, metal or cpu: what training and chat run on."""
    vm = psutil.virtual_memory()
    gpus = nvidia_gpus() or amd_gpus() or apple_gpu(vm.total, 0)
    return gpus[0]["backend"] if gpus else "cpu"


def probe() -> dict:
    vm = psutil.virtual_memory()
    gpus = nvidia_gpus() or amd_gpus() or apple_gpu(vm.total, vm.total - vm.available)
    backend = gpus[0]["backend"] if gpus else "cpu"
    trainer = None
    if backend in ("cuda", "rocm") and has_module(GPU_TRAINER):
        trainer = "pytorch"
    elif backend == "metal" and has_module("mlx_lm"):
        trainer = "mlx"
    disk = psutil.disk_usage(str(home()))
    return {
        "os": {"name": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu": {"name": cpu_name(), "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count()},
        "memory": {"total": vm.total, "used": vm.total - vm.available},
        "disk": {"total": disk.total, "free": disk.free},
        "gpus": gpus,
        "backend": backend,
        "capabilities": {
            "gguf_inference": llama_server_path() is not None,
            "trainer": trainer,
            "torch": has_module("torch"),
            "mlx": has_module("mlx_lm"),
        },
    }


@router.get("")
def system_info():
    return probe()
