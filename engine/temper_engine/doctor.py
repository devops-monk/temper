"""Doctor: checks drivers, GPU libraries, disk, ports and engine packages, and says how to fix problems.

Served at /api/doctor for Settings → Diagnostics and printed by `temper-engine doctor` (`temper doctor`).
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil
from fastapi import APIRouter
from packaging.requirements import Requirement

from . import __version__, hardware, home

router = APIRouter(prefix="/api/doctor", tags=["doctor"])

GB = 1024**3
DEFAULT_PORT = 8765
# CUDA 12 wheels of PyTorch need at least this NVIDIA driver.
MIN_NVIDIA_DRIVER = 525
# MLX needs macOS 13.5 or later.
MIN_MACOS_FOR_MLX = (13, 5)


def _check(id: str, title: str, status: str, detail: str, fix: str | None = None, action: str | None = None) -> dict:
    """status is ok, warn, fail or info. action names a fix the app can run for the user."""
    return {"id": id, "title": title, "status": status, "detail": detail, "fix": fix, "action": action}


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _python(code: str, timeout: float) -> tuple[bool, str]:
    """Runs code in a fresh interpreter, so importing PyTorch or MLX never weighs on the engine itself."""
    out = _run([sys.executable, "-c", code], timeout)
    if out is None:
        return False, f"timed out after {timeout:.0f} seconds"
    if out.returncode == 0:
        return True, out.stdout.strip()
    lines = out.stderr.strip().splitlines()
    return False, lines[-1] if lines else f"exited with code {out.returncode}"


# ------------------------------------------------------------------ checks


def check_engine() -> dict:
    return _check("engine", "Engine", "ok", f"Temper engine {__version__} on Python {platform.python_version()} ({sys.prefix})")


def check_packages() -> dict:
    """Every package the engine (and its installed extras) asks for is present at a matching version."""
    try:
        requires = md.requires("temper-engine") or []
    except md.PackageNotFoundError:
        return _check("packages", "Engine packages", "fail", "The engine's package metadata is missing.", "Reinstall the engine.", "reinstall_engine")
    extras = {""} | ({"mlx"} if hardware.has_module("mlx_lm") else set())
    problems = []
    for line in requires:
        req = Requirement(line)
        if req.marker and not any(req.marker.evaluate({"extra": extra}) for extra in extras):
            continue
        try:
            version = md.version(req.name)
        except md.PackageNotFoundError:
            problems.append(f"{req.name} is missing")
            continue
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            problems.append(f"{req.name} {version} doesn't match {req.specifier}")
    if problems:
        return _check("packages", "Engine packages", "fail", "; ".join(problems) + ".", "Reinstall the engine to restore them.", "reinstall_engine")
    return _check("packages", "Engine packages", "ok", f"All {len(requires)} required packages are installed.")


def _disk(path: Path, label: str, id: str) -> dict:
    try:
        free = psutil.disk_usage(str(path)).free
    except OSError:
        return _check(id, label, "info", f"Couldn't read free space for {path}.")
    detail = f"{free / GB:.1f} GB free on the disk that holds {path}."
    if free < 5 * GB:
        return _check(id, label, "fail", detail, "Free up space: downloads, training and exports will fail. Delete models you no longer use on the Models page.")
    if free < 20 * GB:
        return _check(id, label, "warn", detail, "A 7B model and its exports take 15 to 30 GB. Free up space before downloading large models.")
    return _check(id, label, "ok", detail)


def check_disk() -> list[dict]:
    from huggingface_hub import constants

    data = home()
    data.mkdir(parents=True, exist_ok=True)
    checks = [_disk(data, "Disk space", "disk")]
    cache = Path(constants.HF_HUB_CACHE)
    probe = cache if cache.exists() else cache.parent
    try:
        if probe.exists() and os.stat(probe).st_dev != os.stat(data).st_dev:
            checks.append(_disk(probe, "Model cache disk", "disk_cache"))
    except OSError:
        pass
    return checks


def check_memory() -> dict:
    total = psutil.virtual_memory().total
    detail = f"{total / GB:.0f} GB of memory."
    if total < 8 * GB:
        return _check("memory", "Memory", "warn", detail, "Models up to about 3B parameters at 4-bit fit in this much memory. Close other apps before loading one.")
    return _check("memory", "Memory", "ok", detail)


def _gpu_names() -> list[str]:
    """Display adapters the OS knows about, even when their driver isn't installed."""
    system = platform.system()
    if system == "Linux":
        out = _run(["lspci"], 5)
        return [line.split(": ", 1)[-1] for line in (out.stdout if out else "").splitlines() if re.search(r"VGA|3D|Display", line)]
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"], 15)
        return [line.strip() for line in (out.stdout if out else "").splitlines() if line.strip()]
    return []


def _rocm_version() -> str | None:
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            return Path(path).read_text().strip().split("-")[0]
        except OSError:
            continue
    return None


def check_gpu() -> dict:
    system = platform.system()
    nvidia = hardware.nvidia_gpus()
    if nvidia:
        gpu = nvidia[0]
        smi = _run(["nvidia-smi"], 10)
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi.stdout) if smi else None
        names = ", ".join(g["name"] for g in nvidia)
        detail = f"{names}; driver {gpu['driver']}" + (f", supports CUDA {cuda.group(1)}" if cuda else "") + "."
        major = int(str(gpu["driver"]).split(".")[0]) if str(gpu["driver"]).split(".")[0].isdigit() else 0
        if major and major < MIN_NVIDIA_DRIVER:
            return _check("gpu", "GPU and driver", "warn", detail, f"Update the NVIDIA driver to {MIN_NVIDIA_DRIVER} or newer; training uses CUDA 12.")
        return _check("gpu", "GPU and driver", "ok", detail)
    amd = hardware.amd_gpus()
    if amd:
        rocm = _rocm_version()
        return _check("gpu", "GPU and driver", "ok", f"{', '.join(g['name'] for g in amd)}" + (f"; ROCm {rocm}" if rocm else "") + ".")
    if system == "Darwin":
        version = platform.mac_ver()[0]
        if platform.machine() != "arm64":
            return _check("gpu", "GPU and driver", "info", f"Intel Mac, macOS {version}. Chat runs on the CPU; training needs an Apple Silicon Mac or an NVIDIA GPU.")
        parts = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
        detail = f"{hardware.cpu_name()} with Metal, macOS {version}."
        if parts and parts < MIN_MACOS_FOR_MLX:
            return _check("gpu", "GPU and driver", "warn", detail, "Update macOS to 13.5 or later to train with MLX.")
        return _check("gpu", "GPU and driver", "ok", detail)
    adapters = _gpu_names()
    if any("nvidia" in a.lower() for a in adapters):
        return _check(
            "gpu", "GPU and driver", "fail",
            f"Found {next(a for a in adapters if 'nvidia' in a.lower())}, but the NVIDIA driver isn't working (nvidia-smi didn't respond).",
            "Install the current NVIDIA driver from nvidia.com or your distribution, restart, then run the checks again.",
        )
    amd_adapter = next((a for a in adapters if re.search(r"\bAMD\b|Radeon|ATI", a)), None)
    if amd_adapter and system == "Linux":
        return _check("gpu", "GPU and driver", "warn", f"Found {amd_adapter}, but ROCm isn't installed (rocm-smi not found).", "Install ROCm 6.x from AMD to train on this GPU. Chat works without it.")
    if amd_adapter:
        return _check("gpu", "GPU and driver", "info", f"Found {amd_adapter}. Chat can use it; training on AMD GPUs needs Linux with ROCm.")
    return _check("gpu", "GPU and driver", "info", "No supported GPU found. Chat runs on the CPU; training needs an NVIDIA or AMD GPU, or an Apple Silicon Mac.")


def check_training() -> dict:
    backend = hardware.probe_backend()
    if backend == "metal":
        if not hardware.has_module("mlx_lm"):
            return _check("training", "Training", "info", "Training support isn't installed.", "Add training support (about 600 MB).", "add_training")
        ok, out = _python("import mlx.core as mx, mlx_lm; print(mx.metal.is_available(), mlx_lm.__version__)", 60)
        if ok and out.startswith("True"):
            return _check("training", "Training", "ok", f"MLX-LM {out.split()[-1]} can use the GPU.")
        return _check("training", "Training", "fail", f"MLX can't use the GPU: {out}", "Reinstall the engine.", "reinstall_engine")
    if backend in ("cuda", "rocm"):
        if not hardware.has_module("torch"):
            return _check("training", "Training", "info", "Training support isn't installed.", "Add training support (about 6 GB).", "add_training")
        ok, out = _python("import torch; print(torch.cuda.is_available(), torch.__version__, torch.version.cuda or torch.version.hip or '')", 120)
        if ok and out.startswith("True"):
            _, torch_version, toolkit = (out.split() + ["", ""])[:3]
            kind = "CUDA" if backend == "cuda" else "ROCm"
            return _check("training", "Training", "ok", f"PyTorch {torch_version} ({kind} {toolkit}) can use the GPU.")
        return _check(
            "training", "Training", "fail",
            f"PyTorch is installed but can't use the GPU ({out}).",
            "Usually the PyTorch build doesn't match the driver. Update the driver, then reinstall the engine so setup picks the matching build.",
            "reinstall_engine",
        )
    return _check("training", "Training", "info", "Training needs an NVIDIA or AMD GPU, or an Apple Silicon Mac.")


def check_llamacpp() -> dict:
    binary = hardware.llama_server_path()
    if not binary:
        return _check("llamacpp", "Chat runtime", "warn", "llama.cpp isn't installed, so GGUF models can't be loaded.", "Install llama.cpp (about 30 MB).", "install_llamacpp")
    out = _run([binary, "--version"], 20)
    text = ((out.stdout or "") + (out.stderr or "")) if out else ""
    # "version: 0.6.0-dev (build 11538, commit 79e2e74eb)"; releases are tagged b<build>.
    build = re.search(r"build (\d+)", text)
    if out and out.returncode == 0:
        return _check("llamacpp", "Chat runtime", "ok", f"llama.cpp {'b' + build.group(1) + ' ' if build else ''}at {binary}.")
    last = text.strip().splitlines()[-1] if text.strip() else "it didn't respond"
    return _check("llamacpp", "Chat runtime", "fail", f"llama.cpp is installed but won't start: {last}", "Reinstall llama.cpp.", "install_llamacpp")


def _port_owner(port: int) -> str | None:
    """None when the port is free, "Temper" when Temper's engine answers on it, else the program's
    name if the OS tells us (macOS needs root for that) or "another program"."""
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=2)
        if r.headers.get("content-type", "").startswith("application/json") and r.json().get("ok"):
            return "Temper"
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # Nothing listens there. (Windows takes seconds to refuse a closed loopback port; a program
        # that is listening accepts at once, so a connect timeout means the port is free too.)
        return None
    except (httpx.HTTPError, ValueError):
        pass
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
                return psutil.Process(conn.pid).name()
    except (psutil.Error, OSError):
        pass
    return "another program"


def check_port() -> dict:
    mine = os.environ.get("TEMPER_PORT")
    if mine and int(mine) == DEFAULT_PORT:
        return _check("port", "API port", "ok", f"Temper's OpenAI-compatible API is at http://127.0.0.1:{DEFAULT_PORT}/v1.")
    owner = _port_owner(DEFAULT_PORT)
    if not mine:  # run from the command line
        if owner is None:
            return _check("port", "API port", "info", f"Port {DEFAULT_PORT} is free; Temper's API starts there when the app is open.")
        if owner == "Temper":
            return _check("port", "API port", "ok", f"Temper's OpenAI-compatible API is at http://127.0.0.1:{DEFAULT_PORT}/v1.")
        return _check("port", "API port", "warn", f"Port {DEFAULT_PORT} is taken by {owner}.", f"Temper will use another port while it runs, and coding agents set up for {DEFAULT_PORT} won't reach Temper.")
    if owner is None:
        return _check("port", "API port", "info", f"Temper's API is on port {mine}. Port {DEFAULT_PORT} is free now, so restarting the engine moves it back.")
    return _check(
        "port", "API port", "warn",
        f"Port {DEFAULT_PORT} is taken by {owner}, so Temper is on port {mine} this time.",
        f"Coding agents set up for port {DEFAULT_PORT} won't reach Temper. Quit the other program and restart the engine, or copy the new address from the Connect page.",
    )


def check_network() -> dict:
    proxy = next((os.environ[k] for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy") if os.environ.get(k)), None)
    via = f" through the proxy {re.sub(r'//[^@/]*@', '//', proxy)}" if proxy else ""
    try:
        r = httpx.get("https://huggingface.co/api/models", params={"limit": 1}, timeout=8, follow_redirects=True)
        r.raise_for_status()
        return _check("network", "Hugging Face", "ok", f"huggingface.co is reachable{via}.")
    except httpx.HTTPError as e:
        return _check(
            "network", "Hugging Face", "warn",
            f"Couldn't reach huggingface.co{via}: {type(e).__name__}.",
            "Model search and downloads need it. Check your internet connection, or set HTTPS_PROXY if your network uses a proxy.",
        )


CHECKS = [check_engine, check_packages, check_disk, check_memory, check_gpu, check_training, check_llamacpp, check_port, check_network]


def run() -> list[dict]:
    def safe(fn):
        try:
            result = fn()
        except Exception as e:  # a broken check must not hide the others
            result = _check(fn.__name__.removeprefix("check_"), fn.__name__.removeprefix("check_").title(), "warn", f"This check couldn't run: {e}")
        return result if isinstance(result, list) else [result]

    with ThreadPoolExecutor(max_workers=len(CHECKS)) as pool:
        return [c for group in pool.map(safe, CHECKS) for c in group]


@router.get("")
def doctor():
    return {"checks": run()}


SYMBOL = {"ok": "✓", "warn": "!", "fail": "✗", "info": "·"}


def report() -> int:
    """Prints the checks for `temper doctor`; exits 1 when any check failed."""
    # Windows pipes default to a legacy code page that has no ✓ or →.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = run()
    width = max(len(c["title"]) for c in checks)
    for c in checks:
        print(f"{SYMBOL[c['status']]} {c['title']:<{width}}  {c['detail']}")
        if c["fix"] and c["status"] != "ok":
            print(f"  {'':<{width}}  → {c['fix']}")
    failed = sum(c["status"] == "fail" for c in checks)
    warned = sum(c["status"] == "warn" for c in checks)
    print()
    print("Everything looks good." if not failed and not warned else f"{failed} problem(s), {warned} warning(s).")
    return 1 if failed else 0
