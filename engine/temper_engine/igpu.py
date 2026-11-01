"""How much memory an integrated GPU may use, and how to give it more.

A GPU that shares the computer's memory gets only part of it. On a Mac, macOS lets the GPU wire about two thirds
of memory (three quarters above 36 GB), set by `iogpu.wired_limit_mb`. On an AMD APU under Linux, the BIOS carves
out a little dedicated memory, and the kernel lets the GPU map system memory up to its GTT limit (half of memory
unless the `ttm.pages_limit` boot parameter says otherwise). Both can be raised, so a bigger model fits on the GPU.
Changing them needs an administrator, so Temper shows the commands rather than running them.
"""

from __future__ import annotations

import platform
from pathlib import Path

import psutil
from fastapi import APIRouter

from . import hardware

router = APIRouter(prefix="/api/shared-memory", tags=["system"])

GB = 1024**3
MB = 1024**2
PAGE = 4096
DRM = Path("/sys/class/drm")
TTM = Path("/sys/module/ttm/parameters/pages_limit")


def keep_for_system(total: int) -> int:
    """What to leave the operating system and apps: a fifth of memory, at least 4 GB and at most 16 GB."""
    return min(max(total // 5, 4 * GB), 16 * GB)


def mac_default(total: int) -> int:
    return int(total * (0.75 if total > 36 * GB else 0.67))


def _mac(total: int) -> dict:
    out = hardware._run(["sysctl", "-n", "iogpu.wired_limit_mb"]) or ""
    set_mb = int(out.strip()) if out.strip().isdigit() else 0
    limit = set_mb * MB if set_mb else mac_default(total)
    suggested_mb = (total - keep_for_system(total)) // GB * 1024
    return {
        "kind": "apple",
        "total": total,
        "limit": limit,
        "default": mac_default(total),
        "custom": bool(set_mb),
        "suggested": suggested_mb * MB,
        "commands": {
            "raise": f"sudo sysctl iogpu.wired_limit_mb={suggested_mb}",
            "keep": f"echo 'iogpu.wired_limit_mb={suggested_mb}' | sudo tee -a /etc/sysctl.conf",
            "reset": "sudo sysctl iogpu.wired_limit_mb=0",
        },
    }


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def amd_apu(total: int) -> dict | None:
    """An AMD GPU whose own memory is a small carve-out of system memory: (card, carve-out, GTT limit)."""
    for card in sorted(DRM.glob("card[0-9]*")):
        device = card / "device"
        if (device / "vendor").is_file() and (device / "vendor").read_text().strip() == "0x1002":
            vram, gtt = _read_int(device / "mem_info_vram_total"), _read_int(device / "mem_info_gtt_total")
            if vram is not None and gtt is not None and vram <= total // 4:
                return {"card": card.name, "vram": vram, "gtt": gtt}
    return None


def _amd(total: int, apu: dict) -> dict:
    suggested = total - keep_for_system(total)
    pages = suggested // PAGE
    params = f"ttm.pages_limit={pages} ttm.page_pool_size={pages}"
    return {
        "kind": "amd_apu",
        "total": total,
        "carve_out": apu["vram"],
        "limit": apu["gtt"],
        "default": total // 2,
        "custom": _read_int(TTM) not in (None, total // 2 // PAGE),
        "suggested": suggested,
        "commands": {
            # Added to the kernel's boot parameters; GRUB is what most distributions boot with.
            "raise": f'sudo sed -i \'s/^GRUB_CMDLINE_LINUX_DEFAULT="/&{params} /\' /etc/default/grub && sudo update-grub',
            "params": params,
        },
    }


def report() -> dict:
    """What this computer's integrated GPU may use, and what to run to give it more; kind "none" without one."""
    total = psutil.virtual_memory().total
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return _mac(total)
    if platform.system() == "Linux" and (apu := amd_apu(total)):
        return _amd(total, apu)
    return {"kind": "none", "total": total}


@router.get("")
def shared_memory():
    return report()
