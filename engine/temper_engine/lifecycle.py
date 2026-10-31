"""Updating and uninstalling Temper from the command line (`temper update`, `temper uninstall`).

Updates come from GitHub Releases, like the app's own updater: `latest()` reads the newest release, and
updating runs the same installer as the download page (install.sh, or install.ps1 on Windows), which
checks the download's SHA-256 before installing it.

Uninstalling removes what Temper put on this computer: the engine starting at login, its folders (Temper's
home with runs, exports, settings and its Python, and the app's caches and web storage) and the app itself
where it's a single thing to remove (Temper.app, or the AppImage). Models downloaded from Hugging Face stay,
since other apps share that cache; so does the Hugging Face token in the system keychain, which the app
removes when you sign out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from packaging.version import InvalidVersion, Version

from . import __version__, home

REPO = "devops-monk/temper"
IDENTIFIER = "app.temper.studio"  # tauri.conf.json: names the app's folders on every system
INSTALL_SH = "https://temper.devops-monk.com/install.sh"
INSTALL_PS1 = "https://temper.devops-monk.com/install.ps1"


def latest(timeout: float = 10) -> dict:
    """The newest release: its version, notes and page."""
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    r = httpx.get(f"https://api.github.com/repos/{REPO}/releases/latest", headers=headers, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    data = r.json()
    return {"version": str(data["tag_name"]).lstrip("v"), "notes": data.get("body") or "", "url": data.get("html_url") or f"https://github.com/{REPO}/releases"}


def newer(available: str, current: str = __version__) -> bool:
    try:
        return Version(available) > Version(current)
    except InvalidVersion:
        return available != current


def installer() -> list[str]:
    """The command that downloads and installs the newest release, as the download page gives it."""
    if sys.platform == "win32":
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"irm {INSTALL_PS1} | iex"]
    return ["sh", "-c", f"curl -fsSL {INSTALL_SH} | sh"]


def run_installer() -> int:
    return subprocess.call(installer())


# --- Uninstalling ---------------------------------------------------------------------------------


def app_folders() -> list[Path]:
    """The folders the app and its web view keep, on this system, that exist."""
    user = Path.home()
    if sys.platform == "darwin":
        lib = user / "Library"
        found = [lib / "Application Support" / IDENTIFIER, lib / "Caches" / IDENTIFIER, lib / "Logs" / IDENTIFIER, lib / "WebKit" / IDENTIFIER, lib / "Saved Application State" / f"{IDENTIFIER}.savedState"]
    elif sys.platform == "win32":
        found = [Path(os.environ[v]) / IDENTIFIER for v in ("APPDATA", "LOCALAPPDATA") if os.environ.get(v)]
    else:
        data = Path(os.environ.get("XDG_DATA_HOME") or user / ".local" / "share")
        cache = Path(os.environ.get("XDG_CACHE_HOME") or user / ".cache")
        config = Path(os.environ.get("XDG_CONFIG_HOME") or user / ".config")
        found = [data / IDENTIFIER, cache / IDENTIFIER, config / IDENTIFIER]
    return [p for p in found if p.exists()]


def app_bundle() -> Path | None:
    """The app itself, when removing it is deleting one thing: Temper.app on a Mac, or the AppImage."""
    if sys.platform == "darwin":
        return next((p for p in (Path("/Applications/Temper.app"), Path.home() / "Applications" / "Temper.app") if p.exists()), None)
    if os.environ.get("APPIMAGE") and Path(os.environ["APPIMAGE"]).is_file():
        return Path(os.environ["APPIMAGE"])
    return None


def how_to_remove_app() -> str | None:
    """For installs this can't remove itself: how to."""
    if sys.platform == "win32":
        return "Remove the app in Settings → Apps → Installed apps → Temper."
    if sys.platform.startswith("linux") and not app_bundle() and Path("/usr/bin/temper").exists():
        return "Remove the app with your package manager: sudo apt remove temper (or sudo dnf remove temper)."
    return None


def size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).lstat().st_size
            except OSError:
                pass
    return total


def safe(path: Path) -> bool:
    """Never the home folder, one of its parents or a top-level folder, whatever TEMPER_HOME says."""
    path = path.resolve()
    user = Path.home().resolve()
    return path != user and path not in user.parents and len(path.parts) > 3


def plan(keep_data: bool = False) -> list[Path]:
    """What uninstalling removes, the data first and the app last."""
    found: list[Path] = []
    if not keep_data:
        for p in [home(), *app_folders()]:
            if p.exists() and safe(p) and not any(p == q or q in p.parents for q in found):
                found.append(p)
    if bundle := app_bundle():
        found.append(bundle)
    return found


def remove(paths: list[Path]) -> list[tuple[Path, str]]:
    """Deletes each path; the ones that couldn't be, with why."""
    failed = []
    for p in paths:
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError as e:
            failed.append((p, e.strerror or str(e)))
    return failed


def hugging_face_cache() -> tuple[Path, int] | None:
    cache = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    return (cache, size(cache)) if cache.is_dir() else None
