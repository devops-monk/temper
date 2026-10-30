"""Runs code a chat model writes, away from the rest of the computer.

Each chat gets a working folder (sandbox/<chat> in Temper's home) that its code can read and write,
kept between calls so a later call can use files an earlier one made. The code runs in its own
process with a stripped environment, a time limit and a cap on file sizes, and inside the operating
system's sandbox where there is one:

- macOS: Seatbelt (sandbox-exec), with no network, writes only to the working folder, and
  no reading the folders that hold keys and credentials.
- Linux: bubblewrap (bwrap) when it's installed, with the same rules: a read-only system, the
  working folder writable, the credential folders hidden, and no network.
- Otherwise (Windows, or Linux without bwrap) the limits above still apply, but nothing stops the
  code using the network or touching other files; `kind()` says so, and the app shows it.

Python runs in Temper's own interpreter (the standard library only) until data libraries are added
(numpy, pandas, matplotlib, sympy), which go in a venv of their own. JavaScript needs Node.js on PATH.
Shell commands run in bash (or sh) on macOS and Linux, and PowerShell on Windows.

`check()` tries to break out (reach the network, write outside the folder, read a credential
folder) from inside the sandbox, so the app can show what actually holds on this computer.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from . import home
from .engines import _venv_python, install_venv
from .jobs import registry

router = APIRouter(prefix="/api/sandbox", tags=["chat"])

TIMEOUT = 30  # seconds a run may take
MAX_OUTPUT = 12_000  # characters of output returned
MAX_FILE = 200 * 1024 * 1024  # bytes any file it writes may grow to
DATA_PACKAGES = ["numpy", "pandas", "matplotlib", "sympy", "scipy"]
# Folders under the home folder that hold keys and credentials; code can't read them.
SECRETS = [".ssh", ".aws", ".gnupg", ".config/gh", ".config/gcloud", ".azure", ".kube", ".docker", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "Library/Keychains"]


def root() -> Path:
    return home() / "sandbox"


def workdir(chat: str | None) -> Path:
    name = re.sub(r"[^\w-]", "", chat or "") or "scratch"
    folder = root() / "chats" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def forget(chat: str) -> None:
    """Removes a deleted chat's working folder."""
    shutil.rmtree(root() / "chats" / re.sub(r"[^\w-]", "", chat), ignore_errors=True)


def data_venv() -> Path:
    return root() / "python"


def python() -> str:
    """The Python that runs code: the data-libraries venv once it's installed, else Temper's own."""
    venv = data_venv()
    return str(_venv_python(venv)) if (venv / ".installed").is_file() else sys.executable


def node() -> str | None:
    return shutil.which("node")


def shell() -> list[str] | None:
    """The shell commands run in, with its arguments before the script's name."""
    if os.name == "nt":
        ps = shutil.which("pwsh") or shutil.which("powershell")
        return [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"] if ps else None
    sh = shutil.which("bash") or ("/bin/sh" if Path("/bin/sh").exists() else None)
    return [sh] if sh else None


@functools.cache
def _bwrap_works() -> bool:
    """bwrap is installed and allowed to make namespaces (some distributions forbid that to unprivileged users)."""
    try:
        return subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-net", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def kind() -> str | None:
    """The OS sandbox code runs in: "seatbelt", "bubblewrap", or None when there's none here."""
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap") and _bwrap_works():
        return "bubblewrap"
    return None


def _quote(path: Path | str) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _seatbelt(work: Path) -> str:
    """A Seatbelt profile: anything but the network, writes outside the working folder, and reading secrets."""
    user = Path.home()
    secrets = " ".join(f"(subpath {_quote(user / s)})" for s in SECRETS)
    writable = f"(subpath {_quote(work)})"
    return f"""(version 1)
(allow default)
(deny network*)
(allow network* (local unix-socket))
(deny file-write* (require-not (require-any {writable} (literal "/dev/null") (literal "/dev/tty") (literal "/dev/stdout") (literal "/dev/stderr"))))
(deny file-read* {secrets})
"""


def wrap(cmd: list[str], work: Path) -> list[str]:
    """`cmd` inside the OS sandbox, when there is one."""
    which = kind()
    if which == "seatbelt":
        return ["/usr/bin/sandbox-exec", "-p", _seatbelt(work), *cmd]
    if which == "bubblewrap":
        user = Path.home()
        hide = [a for s in SECRETS if (user / s).exists() for a in (("--tmpfs", str(user / s)) if (user / s).is_dir() else ("--ro-bind", "/dev/null", str(user / s)))]
        return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", str(work), str(work), *hide, "--unshare-net", "--unshare-pid", "--die-with-parent", "--chdir", str(work), *cmd]
    return cmd


# Sets the CPU-time and file-size limits, then becomes the command (a launcher, since the engine's threads make preexec_fn unsafe).
LIMITS = "import os, resource, sys; resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu})); resource.setrlimit(resource.RLIMIT_FSIZE, ({size}, {size})); os.execv(sys.argv[1], sys.argv[1:])"


def _limited(cmd: list[str]) -> list[str]:
    if os.name == "nt":
        return cmd
    # CPU time can run ahead of the clock (macOS counts it generously), so the wall-clock TIMEOUT is what normally stops a run.
    return [sys.executable, "-I", "-c", LIMITS.format(cpu=TIMEOUT * 4, size=MAX_FILE), *cmd]


def _cut(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[: MAX_OUTPUT // 2] + f"\n[… {len(text) - MAX_OUTPUT:,} characters cut …]\n" + text[-MAX_OUTPUT // 2 :]


def run(language: str, code: str, chat: str | None) -> str:
    """Runs `code` in the chat's working folder; returns what it printed, its exit code, and the files it made."""
    work = workdir(chat)
    name = f".temper-run-{uuid.uuid4().hex[:8]}"
    if language == "python":
        script, cmd = work / f"{name}.py", [python(), "-I", "-B", f"{name}.py"]
    elif language == "javascript":
        if not (runtime := node()):
            raise RuntimeError("Node.js isn't installed on this computer, so JavaScript can't run. Use Python instead.")
        script, cmd = work / f"{name}.mjs", [runtime, f"{name}.mjs"]
    elif language == "shell":
        if not (sh := shell()):
            raise RuntimeError("There's no shell on this computer for commands to run in.")
        script = work / (f"{name}.ps1" if os.name == "nt" else f"{name}.sh")
        cmd = [*sh, script.name]
    else:
        raise ValueError(f"Temper can't run {language}.")
    script.write_text(code, encoding="utf-8")
    before = {p: p.stat().st_mtime for p in work.rglob("*") if p.is_file()}
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work), "TMPDIR": str(work), "TEMP": str(work), "TMP": str(work), "LANG": "en_US.UTF-8", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg", "NO_COLOR": "1"}
    if os.name == "nt":
        env |= {k: os.environ[k] for k in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT") if k in os.environ}
    started = time.monotonic()
    proc = subprocess.Popen(
        wrap(_limited(cmd), work),
        cwd=work,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=os.name != "nt",
    )
    try:
        out, err = proc.communicate(timeout=TIMEOUT)
        stopped = False
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        out, err = proc.communicate()
        stopped = True
    finally:
        script.unlink(missing_ok=True)
    made = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file() and p != script and before.get(p) != p.stat().st_mtime)
    parts = []
    if out.strip():
        parts.append(_cut(out.rstrip()))
    if err.strip():
        parts.append("stderr:\n" + _cut(err.rstrip()))
    if stopped or (hasattr(signal, "SIGXCPU") and proc.returncode == -signal.SIGXCPU):
        parts.append(f"(Stopped: it ran longer than {TIMEOUT} seconds.)")
    elif proc.returncode:
        parts.append(f"(Exit code {proc.returncode}.)")
    if made:
        parts.append("Files written: " + ", ".join(made[:30]) + (" …" if len(made) > 30 else ""))
    return "\n\n".join(parts) or f"(It printed nothing; it ran in {time.monotonic() - started:.1f} s.)"


# Run inside the sandbox: each attempt prints whether it was stopped.
PROBE = """
import json, os, socket, sys
outside, secret = sys.argv[1], sys.argv[2]
def tried(f):
    try:
        f()
        return False
    except Exception:
        return True
def write():
    with open(outside, "w") as f:
        f.write("x")
def connect():
    socket.create_connection(("1.1.1.1", 443), timeout=4).close()
print(json.dumps({"network": tried(connect), "write": tried(write), "secrets": tried(lambda: os.listdir(secret)) if secret else None}))
"""


def check() -> dict:
    """What the sandbox stops, found by trying it: the network, writing outside the working folder, reading credentials."""
    work = workdir(None)
    outside = root() / f"outside-{uuid.uuid4().hex[:8]}"
    secret = next((str(Path.home() / s) for s in SECRETS if (Path.home() / s).is_dir()), "")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work), "TMPDIR": str(work)} | ({k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ} if os.name == "nt" else {})
    try:
        done = subprocess.run(wrap(_limited([python(), "-I", "-c", PROBE, str(outside), secret]), work), cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT)
        found = json.loads(done.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        found = {"network": None, "write": None, "secrets": None}
    finally:
        outside.unlink(missing_ok=True)
    return {
        "sandbox": kind(),
        "checks": [
            {"id": "network", "blocked": found["network"]},
            {"id": "write", "blocked": found["write"]},
            {"id": "secrets", "blocked": found["secrets"], "path": secret or None},
            {"id": "time", "blocked": True, "seconds": TIMEOUT},
            {"id": "size", "blocked": os.name != "nt", "megabytes": MAX_FILE // (1024 * 1024)},
        ],
    }


@router.get("")
def status():
    venv = data_venv()
    return {
        "sandbox": kind(),
        "python": python(),
        "data_libraries": (venv / ".installed").is_file(),
        "packages": DATA_PACKAGES,
        "node": node(),
        "shell": (shell() or [None])[0],
    }


@router.get("/check")
def check_sandbox():
    return check()


@router.post("/python")
def add_data_libraries():
    job = registry.create("sandbox-install", "Add data libraries to the code sandbox")
    registry.run_thread(job, lambda j: install_venv(j, data_venv(), DATA_PACKAGES, "the data libraries") and j.finish("done"))
    return job.summary()


@router.delete("/python")
def remove_data_libraries():
    if not data_venv().exists():
        raise HTTPException(404, "The data libraries aren't installed.")
    shutil.rmtree(data_venv(), ignore_errors=True)
    return status()
