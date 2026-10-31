"""Linked devices: other computers running Temper whose models this one can use, such as a desktop's GPU from a laptop.

A device is linked by its address and one of its named API keys (made under Connect → API keys there, with
its network access on). Its models are listed in /v1/models as `<model>@<device>`, and a request naming one
goes to that device. A request naming a model by itself goes to this computer when it has that model, unless
the preferred device has it too; when only a linked device has it, it goes there. Forwarded requests carry
a marker so two computers linked to each other never pass a request back and forth.

The keys are kept in devices.json in Temper's home, readable only by this user.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home, prefs

router = APIRouter(prefix="/api/devices", tags=["runtime"])

# How long a device's model list is trusted before it's asked again, and how long it has to answer.
FRESH = 30.0
TIMEOUT = 4.0
# Set on requests passed to a device: that device answers them itself rather than passing them on.
HOP = "temper_hop"

_lock = threading.Lock()
_seen: dict[str, tuple[float, list[dict] | None, str | None]] = {}  # device id: (when, its models, why it didn't answer)


def _file():
    return home() / "devices.json"


def linked() -> list[dict]:
    try:
        data = json.loads(_file().read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(devices: list[dict]) -> None:
    path = _file()
    path.write_text(json.dumps(devices, indent=2))
    if os.name != "nt":
        path.chmod(0o600)


def preferred() -> str | None:
    """The device that answers when it and this computer both have the model; None for this computer."""
    found = prefs.load().get("preferred_device")
    return found if any(d["id"] == found for d in linked()) else None


def _headers(device: dict) -> dict:
    return {"Authorization": f"Bearer {device['key']}"}


def _ask(device: dict) -> tuple[list[dict] | None, str | None]:
    """The device's own models (not the ones it reaches through its linked devices), or why it didn't answer."""
    try:
        r = httpx.get(f"{device['url']}/v1/models", headers=_headers(device), timeout=TIMEOUT)
    except httpx.HTTPError:
        return None, f"Nothing answered at {device['url']}."
    if r.status_code in (401, 403):
        return None, "It refused the key; it may have been revoked."
    if r.status_code != 200:
        return None, f"It answered {r.status_code}."
    try:
        data = r.json().get("data") or []
    except (ValueError, AttributeError):
        return None, "It didn't answer like Temper."
    return [{"id": m["id"], "loaded": m.get("loaded", True)} for m in data if isinstance(m, dict) and m.get("id") and not m.get("device")], None


def models(device: dict, fresh: bool = False) -> tuple[list[dict] | None, str | None]:
    """The device's models from the last FRESH seconds, asking it again when that's older."""
    with _lock:
        seen = _seen.get(device["id"])
    if seen and not fresh and time.time() - seen[0] < FRESH:
        return seen[1], seen[2]
    found, why = _ask(device)
    with _lock:
        _seen[device["id"]] = (time.time(), found, why)
    return found, why


def _all_models(fresh: bool = False) -> list[tuple[dict, list[dict] | None, str | None]]:
    devices = linked()
    if not devices:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(devices))) as pool:
        answers = list(pool.map(lambda d: models(d, fresh), devices))
    return [(d, found, why) for d, (found, why) in zip(devices, answers, strict=True)]


def listing() -> list[dict]:
    """/v1/models entries for the linked devices' models."""
    out = []
    for device, found, _ in _all_models():
        for m in found or []:
            out.append({"id": f"{m['id']}@{device['name']}", "object": "model", "owned_by": device["name"], "device": device["name"], "loaded": m["loaded"]})
    return out


class Remote:
    """A linked device's model, standing in for a local model server in runtime.route()."""

    engine = "remote"
    state = "ready"
    modalities = None
    projector = None
    error = None
    context = None

    def __init__(self, device: dict, model: str):
        self.device, self.name = device, model
        self.model = f"{model} on {device['name']}"
        self.base = device["url"]
        self.headers = _headers(device)
        self.used = time.time()


def target(model: str | None, here) -> Remote | None:
    """The linked device that should answer a request for `model`, or None to answer it on this computer.
    `here()` says whether this computer has the model (called only when a device has it too)."""
    devices = linked()
    if not model or not devices:
        return None
    if "@" in model:
        name, _, device = model.rpartition("@")
        found = next((d for d in devices if d["name"].lower() == device.lower()), None)
        if found:
            return Remote(found, name)
    wanted = model.lower()
    having = [(d, next(m for m in found if m["id"].lower() == wanted)) for d, found, _ in _all_models() if found and any(m["id"].lower() == wanted for m in found)]
    if not having:
        return None
    chosen = next((d for d, _ in having if d["id"] == preferred()), None)
    if chosen is None:
        if here():
            return None
        # One that has it loaded already answers sooner than one that would load it first.
        chosen = sorted(having, key=lambda h: not h[1]["loaded"])[0][0]
    return Remote(chosen, model)


def forwarded(body: dict, remote: Remote) -> dict:
    return {**body, "model": remote.name, HOP: True}


# --- Linking ---------------------------------------------------------------------------------------


def address(url: str) -> str:
    """http://host:port from what was typed: a bare host:port, or an address with /v1 on the end."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    if url.endswith("/v1"):
        url = url[:-3]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "That isn't an address. Use what the other computer's Connect page shows, like http://192.168.1.20:8765.")
    if not parsed.port and parsed.scheme == "http":
        url = f"{url}:8765"
    return url


def public(device: dict, found: list[dict] | None, why: str | None) -> dict:
    return {"id": device["id"], "name": device["name"], "url": device["url"], "key_hint": f"{device['key'][:6]}…{device['key'][-4:]}",
            "online": found is not None, "error": why, "models": found or [], "preferred": device["id"] == preferred()}


class NewDevice(BaseModel):
    url: str = Field(min_length=1)
    key: str = Field(min_length=8)
    name: str | None = Field(default=None, max_length=40)


class Preferred(BaseModel):
    device: str | None = None


@router.get("")
def list_devices(fresh: bool = False):
    return {"devices": [public(*row) for row in _all_models(fresh)], "preferred": preferred()}


@router.post("")
def link(req: NewDevice):
    url = address(req.url)
    name = (req.name or "").strip() or urlparse(url).hostname or url
    if "@" in name:
        raise HTTPException(400, "A device's name can't contain @; it's what separates the model from the device in a request.")
    devices = linked()
    if any(d["name"].lower() == name.lower() for d in devices):
        raise HTTPException(409, f"A device called {name} is linked already. Give this one another name.")
    device = {"id": uuid.uuid4().hex[:12], "name": name, "url": url, "key": req.key.strip()}
    found, why = _ask(device)
    if found is None:
        raise HTTPException(400, f"Couldn't link {url}: {why}")
    devices.append(device)
    with _lock:
        _save(devices)
        _seen[device["id"]] = (time.time(), found, None)
    return public(device, found, None)


@router.delete("/{device_id}")
def unlink(device_id: str):
    devices = linked()
    if not any(d["id"] == device_id for d in devices):
        raise HTTPException(404, "No such device.")
    with _lock:
        _save([d for d in devices if d["id"] != device_id])
        _seen.pop(device_id, None)
    if prefs.load().get("preferred_device") == device_id:
        prefs.update("preferred_device", None)
    return list_devices()


@router.put("/preferred")
def set_preferred(req: Preferred):
    if req.device is not None and not any(d["id"] == req.device for d in linked()):
        raise HTTPException(404, "No such device.")
    prefs.update("preferred_device", req.device)
    return list_devices()
