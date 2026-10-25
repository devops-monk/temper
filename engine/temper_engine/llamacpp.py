"""Managed llama.cpp: pinned official builds downloaded into Temper's folder, plus the HF → GGUF converter.

Nothing is installed system-wide. Every download is checked against a pinned SHA-256. A llama.cpp
already on PATH (Homebrew, winget) is used when Temper has not installed its own.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from . import home

router = APIRouter(prefix="/api/llamacpp", tags=["llama.cpp"])

TAG = "b11538"
RELEASES = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}"

# (os, machine, gpu backend) → archives to unpack into one folder, with their published SHA-256.
_BUILDS: dict[tuple[str, str, str], list[tuple[str, str]]] = {
    ("Darwin", "arm64", "any"): [(f"llama-{TAG}-bin-macos-arm64.tar.gz", "291c46b9288975ec54771d4056911d9cb773b954afb6c5eba35dc9fa359d45f4")],
    ("Darwin", "x86_64", "any"): [(f"llama-{TAG}-bin-macos-x64.tar.gz", "610fa04364d1ba89ccfa5b18019a77c6a13039103426ab9ff907b5b49329f988")],
    ("Linux", "x86_64", "cuda"): [
        (f"llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz", "03b6555a8c615008653fbc8b872acc4be0e5858891e7964e27c07464c5cf90d0"),
        (f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz", "1ac82f3ac703af5acda734b21ee166b2aad9a032392b1fa773ce73cc82c9f020"),
    ],
    ("Linux", "x86_64", "rocm"): [(f"llama-{TAG}-bin-ubuntu-rocm-10.0-x64.tar.gz", "474dbc6463105896162b8d5e33606a15367524fa30afe5056562284964333950")],
    ("Linux", "x86_64", "cpu"): [(f"llama-{TAG}-bin-ubuntu-x64.tar.gz", "75c2d8da0ed790d7ebd0eeb3740a72eef8bce4a35745cf0306f1a9b72ff8f4d0")],
    ("Linux", "aarch64", "cuda"): [
        (f"llama-{TAG}-bin-ubuntu-cuda-13.4-arm64.tar.gz", "426c0e66eb5c53bbcd962c59b1420b3502a4ef0030f0d461313573dfeca01e76"),
        (f"cudart-llama-{TAG}-bin-ubuntu-cuda-13.4-arm64.tar.gz", "069455a731006464e2888d695892dda33c51aa433fe531faa48cb3098a7205fe"),
    ],
    ("Linux", "aarch64", "cpu"): [(f"llama-{TAG}-bin-ubuntu-arm64.tar.gz", "c6f20444e44c8d685c372849497501c9e73f85c1d5579dca20501069cc761c97")],
    ("Windows", "AMD64", "cuda"): [
        (f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", "631c1397231497dabcdd1eb21ae43ad235093c22c88cc64f09ac3be67285cd6e"),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ],
    # The Windows ROCm build needs the HIP SDK; Vulkan runs on AMD GPUs with just the driver.
    ("Windows", "AMD64", "rocm"): [(f"llama-{TAG}-bin-win-vulkan-x64.zip", "621ec0ed653ec9d40673be866558d2675eb54907ee4255ecfeb2739f2a6dccf5")],
    ("Windows", "AMD64", "cpu"): [(f"llama-{TAG}-bin-win-cpu-x64.zip", "8d27ba71dfb5c2f0733a048dfe3eea3edc0e31d7b831a25d3610c3db81dc0bec")],
    ("Windows", "ARM64", "cpu"): [(f"llama-{TAG}-bin-win-cpu-arm64.zip", "2772371aa03213e126b1e9f32818a7cd94528bc688bd0a04e8093c66b999248a")],
}

SOURCE_URL = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{TAG}"
# Hash over the converter files we extract (paths + contents); source tarballs are not byte-stable.
CONVERTER_PARTS = ("convert_hf_to_gguf.py", "conversion", "gguf-py/gguf", "requirements")
CONVERTER_SHA256 = "a8dc9268a03632042285fa85d41b0a68dff27aa4621bb940fc0879431a173255"

Report = Callable[[str, "float | None"], None]


def _quiet(_message: str, _progress: float | None = None) -> None:
    pass


def root() -> Path:
    return home() / "llama.cpp"


def _gpu_backend() -> str:
    from .hardware import probe

    return probe()["backend"]


def build_key(backend: str | None = None) -> tuple[str, str, str]:
    system, machine = platform.system(), platform.machine()
    machine = {"aarch64": "aarch64", "arm64": "arm64" if system == "Darwin" else "aarch64", "x86_64": "x86_64", "AMD64": "AMD64", "ARM64": "ARM64"}.get(machine, machine)
    if system == "Darwin":
        return (system, machine, "any")
    backend = backend or _gpu_backend()
    gpu = backend if backend in ("cuda", "rocm") else "cpu"
    for key in ((system, machine, gpu), (system, machine, "cpu")):
        if key in _BUILDS:
            return key
    raise RuntimeError(f"There is no llama.cpp build for {system} {machine}")


def build_dir(key: tuple[str, str, str]) -> Path:
    return root() / f"{TAG}-{key[1].lower()}-{key[2]}"


def installed_build() -> Path | None:
    """Temper's own llama.cpp, if installed (any variant)."""
    if not root().is_dir():
        return None
    for d in sorted(root().glob(f"{TAG}-*")):
        if (d / ".complete").is_file():
            return d
    return None


def binary(name: str) -> str | None:
    """Path to a llama.cpp tool: Temper's own build first, then PATH and the usual Homebrew spots."""
    exe = name + (".exe" if os.name == "nt" else "")
    build = installed_build()
    if build:
        found = next((p for p in build.rglob(exe) if p.is_file()), None)
        if found:
            return str(found)
    for candidate in (shutil.which(name), f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _download(url: str, sha256: str | None, report: Report, label: str) -> bytes:
    report(f"Downloading {url}", 0.0)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        for chunk in resp.iter_bytes(1 << 20):
            chunks.append(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                report(f"{label}: {done // (1 << 20)} of {total // (1 << 20)} MB", done / total)
    if sha256 and digest.hexdigest() != sha256:
        raise RuntimeError(f"{label} download is corrupt (checksum {digest.hexdigest()}). Try again.")
    return b"".join(chunks)


def _unpack(data: bytes, name: str, dest: Path) -> None:
    """Unpacks an archive into dest, dropping the single top-level folder most archives have."""
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(dest, filter="data")
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            child.rename(dest / child.name)
        inner.rmdir()


def install(report: Report = _quiet, backend: str | None = None) -> Path:
    """Downloads and unpacks the pinned llama.cpp build for this machine. Safe to call again."""
    key = build_key(backend)
    final = build_dir(key)
    if (final / ".complete").is_file():
        report(f"llama.cpp {TAG} is already installed", 1.0)
        return final
    root().mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=root()))
    try:
        for name, sha in _BUILDS[key]:
            data = _download(f"{RELEASES}/{name}", sha, report, name)
            part = staging / name.split(".")[0]
            part.mkdir()
            _unpack(data, name, part)
            for child in part.iterdir():  # merge the CUDA runtime next to the binaries
                target = staging / child.name
                if not target.exists():
                    child.rename(target)
            shutil.rmtree(part)
        for f in staging.iterdir():
            if f.is_file() and not f.suffix and os.name != "nt":
                f.chmod(0o755)
        (staging / ".complete").write_text(json.dumps({"tag": TAG, "build": key[2]}))
        shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    report(f"Installed llama.cpp {TAG}", 1.0)
    return final


def converter_dir() -> Path:
    return root() / f"{TAG}-converter"


def _tree_hash(base: Path) -> str:
    files = []
    for part in CONVERTER_PARTS:
        p = base / part
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.relative_to(base).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def install_converter(report: Report = _quiet) -> Path:
    """Fetches llama.cpp's HF → GGUF converter (the script, its model registry and gguf-py)."""
    final = converter_dir()
    if (final / ".complete").is_file():
        return final
    data = _download(SOURCE_URL, None, report, "llama.cpp converter")
    root().mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".converter-", dir=root()))
    try:
        prefix = f"llama.cpp-{TAG}/"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            wanted = [m for m in tf.getmembers() if any(m.name.startswith(prefix + p) for p in CONVERTER_PARTS)]
            for m in wanted:
                m.name = m.name[len(prefix):]
            tf.extractall(staging, members=wanted, filter="data")
        if (got := _tree_hash(staging)) != CONVERTER_SHA256:
            raise RuntimeError(f"The llama.cpp converter does not match the pinned version (hash {got}).")
        (staging / ".complete").write_text(TAG)
        shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return final


def uv_binary() -> str:
    """The uv that installed the engine (the desktop shell passes it in), else one on PATH."""
    for candidate in (os.environ.get("TEMPER_UV"), str(home() / "bin" / ("uv.exe" if os.name == "nt" else "uv")), shutil.which("uv")):
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("uv was not found, so the GGUF converter cannot run. Reinstall the engine from Settings.")


def convert_command(model_dir: Path, outfile: Path, outtype: str = "f16", mistral_format: bool = False) -> tuple[list[str], dict[str, str]]:
    """Command that converts a Hugging Face folder (or, with `mistral_format`, a Mistral-format one) to GGUF.

    The converter needs PyTorch and transformers, which the engine does not ship; uv runs it in a
    cached, isolated environment built from llama.cpp's own pinned requirements. Mistral's format
    also needs mistral-common for its tokenizer.
    """
    conv = install_converter()
    cmd = [
        uv_binary(), "run", "--no-project", "--python", "3.12",
        "--with-requirements", str(conv / "requirements" / "requirements-convert_hf_to_gguf.txt"),
        *(["--with", "mistral-common"] if mistral_format else []),
        "python", str(conv / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(outfile), "--outtype", outtype,
        *(["--mistral-format"] if mistral_format else []),
    ]
    return cmd, uv_env()


def downgraded_tokenizer_config(config: dict) -> dict | None:
    """A transformers 5 tokenizer config rewritten so the converter's pinned transformers 4 can read it;
    None if it needs no change.

    v5 saves `extra_special_tokens` as a list of strings; v4 expects a dict there and keeps the list
    under `additional_special_tokens`.
    """
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return None
    config = {k: v for k, v in config.items() if k != "extra_special_tokens"}
    config["additional_special_tokens"] = list(dict.fromkeys(config.get("additional_special_tokens", []) + extra))
    return config


def uv_env() -> dict[str, str]:
    """Environment for `uv run --no-project` tools: Temper's own Python, outside the engine's venv."""
    env = dict(os.environ)
    if (home() / "python").is_dir():  # reuse the Python first-run setup installed
        env["UV_PYTHON_INSTALL_DIR"] = str(home() / "python")
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def quantize_command(src: Path, dst: Path, quant: str) -> list[str]:
    tool = binary("llama-quantize")
    if not tool:
        raise RuntimeError("llama-quantize was not found. Install llama.cpp from the Models page.")
    return [tool, str(src), str(dst), quant.upper()]


# ------------------------------------------------------------------ API


class InstallRequest(BaseModel):
    converter: bool = False


@router.get("")
def status():
    build = installed_build()
    server = binary("llama-server")
    return {
        "tag": TAG,
        "managed": build is not None,
        "path": str(build) if build else None,
        "server": server,
        "available": server is not None,
        "converter": (converter_dir() / ".complete").is_file(),
    }


@router.post("/install")
def install_endpoint(req: InstallRequest):
    from .jobs import registry

    job = registry.create("download", f"llama.cpp {TAG}")
    job.progress = 0.0

    def run(job):
        def report(message: str, progress: float | None) -> None:
            if progress is not None:
                job.progress = progress
                job.emit("progress", value=progress)
            if not message.startswith(("llama", "cudart")) or progress in (0.0, 1.0):
                job.log(message)

        install(report)
        if req.converter:
            install_converter(report)

    registry.run_thread(job, run)
    return job.summary()


def run_streamed(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Runs a tool, echoing its output line by line (workers forward stdout to the UI log)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, errors="replace")
    assert proc.stdout
    last = ""
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        last = line.strip() or last
    if proc.wait() != 0:
        # The last line is usually the exception (e.g. "AttributeError: ..."); the full output is in the log.
        raise RuntimeError(f"{Path(cmd[0]).name} failed: {last[:300] or f'exit code {proc.returncode}'}")
