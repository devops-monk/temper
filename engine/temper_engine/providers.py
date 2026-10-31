"""Cloud providers: OpenAI, Anthropic, Google, OpenRouter or any OpenAI-compatible service, used next to local models.

A provider is added with its API key (Connect → Cloud providers). Its models are named `<model>@<provider>`, like a
linked device's (devices.py): /v1/models lists the ones chosen to show, a request naming one goes to the provider,
and loading one makes it chat's model without starting anything on this computer (runtime.py). Every provider is
reached through its OpenAI-compatible endpoint, so a request only needs the parameters they all take.

The keys are kept in providers.json in Temper's home, readable only by this user.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home

router = APIRouter(prefix="/api/providers", tags=["runtime"])

KINDS = {
    "openai": {"name": "OpenAI", "url": "https://api.openai.com/v1"},
    "anthropic": {"name": "Anthropic", "url": "https://api.anthropic.com/v1"},
    "google": {"name": "Google", "url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "openrouter": {"name": "OpenRouter", "url": "https://openrouter.ai/api/v1"},
    "custom": {"name": "OpenAI-compatible", "url": ""},
}
# What every OpenAI-compatible endpoint takes; llama.cpp's own sampling settings (top_k, min_p, ...) would be refused.
PASSED = ("messages", "stream", "stream_options", "temperature", "top_p", "max_tokens", "stop", "tools", "tool_choice",
          "parallel_tool_calls", "response_format", "presence_penalty", "frequency_penalty", "seed", "reasoning_effort")
MESSAGE_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")
# OpenAI's reasoning models take only their default sampling, and count replies as max_completion_tokens.
OPENAI_REASONING = re.compile(r"^(o\d|gpt-5)")
FRESH = 300.0
TIMEOUT = 8.0
CONTEXT = 128_000  # what chat assumes a cloud model holds; providers don't say

_lock = threading.Lock()
_seen: dict[str, tuple[float, list[str] | None, str | None]] = {}  # provider id: (when, its models, why it didn't answer)


def _file():
    return home() / "providers.json"


def added() -> list[dict]:
    try:
        data = json.loads(_file().read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(providers: list[dict]) -> None:
    path = _file()
    path.write_text(json.dumps(providers, indent=2))
    if os.name != "nt":
        path.chmod(0o600)


def headers(provider: dict) -> dict:
    out = {"Authorization": f"Bearer {provider['key']}"}
    if provider["kind"] == "anthropic":  # its model list is Anthropic's own API, which wants these
        out |= {"x-api-key": provider["key"], "anthropic-version": "2023-06-01"}
    if provider["kind"] == "openrouter":
        out |= {"HTTP-Referer": "https://temper.devops-monk.com", "X-Title": "Temper"}
    return out


def _ask(provider: dict) -> tuple[list[str] | None, str | None]:
    """The provider's model ids, or why it didn't answer."""
    try:
        r = httpx.get(f"{provider['url']}/models", headers=headers(provider), timeout=TIMEOUT)
    except httpx.HTTPError:
        return None, f"Nothing answered at {provider['url']}."
    if r.status_code in (401, 403):
        return None, "It refused the key."
    if r.status_code != 200:
        return None, f"It answered {r.status_code}."
    try:
        data = r.json().get("data") or []
    except (ValueError, AttributeError):
        return None, "It didn't answer like an OpenAI-compatible API."
    ids = [str(m["id"]).removeprefix("models/") for m in data if isinstance(m, dict) and m.get("id")]
    return sorted(set(ids)), None


def models(provider: dict, fresh: bool = False) -> tuple[list[str] | None, str | None]:
    with _lock:
        seen = _seen.get(provider["id"])
    if seen and not fresh and time.time() - seen[0] < FRESH:
        return seen[1], seen[2]
    found, why = _ask(provider)
    with _lock:
        _seen[provider["id"]] = (time.time(), found, why)
    return found, why


def shown(provider: dict) -> list[str]:
    """The models listed for the provider: the ones chosen, or all it offers when none are."""
    return provider.get("models") or models(provider)[0] or []


def listing() -> list[dict]:
    """/v1/models entries for the providers' models."""
    return [{"id": f"{m}@{p['name']}", "object": "model", "owned_by": p["name"], "provider": p["name"], "loaded": True} for p in added() for m in shown(p)]


def find(name: str | None) -> tuple[dict, str] | None:
    """The provider and model a `model@provider` name means, if it names an added provider."""
    if not name or "@" not in name:
        return None
    model, _, which = name.rpartition("@")
    provider = next((p for p in added() if p["name"].lower() == which.lower()), None)
    return (provider, model) if provider and model else None


class Cloud:
    """A provider's model, standing in for a local model server in runtime.route()."""

    engine = "cloud"
    state = "ready"
    modalities = {"vision": True, "audio": False}
    projector = None
    error = None
    context = CONTEXT
    folder = None

    def __init__(self, provider: dict, model: str):
        self.provider, self.name = provider, model
        self.model = f"{model}@{provider['name']}"
        self.base = provider["url"]
        self.headers = headers(provider)
        self.used = time.time()

    @property
    def url(self) -> str:
        return f"{self.base}/chat/completions"


def target(name: str | None) -> Cloud | None:
    found = find(name)
    return Cloud(*found) if found else None


def translate(cloud: Cloud, body: dict) -> dict:
    """A chat request as an OpenAI-compatible provider takes it."""
    out = {k: v for k, v in body.items() if k in PASSED and v is not None}
    out["model"] = cloud.name
    if isinstance(out.get("messages"), list):
        out["messages"] = [{k: v for k, v in m.items() if k in MESSAGE_KEYS and v is not None} for m in out["messages"] if isinstance(m, dict)]
    if cloud.provider["kind"] == "openai":
        if "max_tokens" in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
        if OPENAI_REASONING.match(cloud.name):
            for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                out.pop(k, None)
    return out


def complete(cloud: Cloud, body: dict, timeout: float = 600) -> dict:
    """One whole reply from the provider, as its JSON."""
    try:
        r = httpx.post(cloud.url, json={**translate(cloud, body), "stream": False}, headers=cloud.headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{cloud.provider['name']} didn't answer: {e}") from e
    if r.status_code != 200:
        raise HTTPException(502, f"{cloud.provider['name']} answered {r.status_code}: {r.text[:300]}")
    return r.json()


# --- Adding ----------------------------------------------------------------------------------------


def public(provider: dict, found: list[str] | None, why: str | None) -> dict:
    key = provider["key"]
    return {"id": provider["id"], "name": provider["name"], "kind": provider["kind"], "url": provider["url"],
            "key_hint": f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "…", "online": found is not None, "error": why,
            "available": found or [], "models": provider.get("models") or []}


class NewProvider(BaseModel):
    kind: str
    key: str = Field(min_length=1)
    name: str | None = Field(default=None, max_length=40)
    url: str | None = None  # needed for custom, and to point a known kind somewhere else
    models: list[str] = Field(default_factory=list)


class Chosen(BaseModel):
    models: list[str]


def _url(kind: str, typed: str | None) -> str:
    url = (typed or KINDS[kind]["url"]).strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url.removesuffix("/chat/completions")
    if not re.match(r"^https?://[^/\s]+", url):
        raise HTTPException(400, "Give the provider's API address, like https://api.example.com/v1.")
    return url


@router.get("")
def list_providers(fresh: bool = False):
    return {"providers": [public(p, *models(p, fresh)) for p in added()], "kinds": KINDS}


@router.post("")
def add(req: NewProvider):
    if req.kind not in KINDS:
        raise HTTPException(400, f"Unknown provider {req.kind}.")
    name = (req.name or "").strip() or req.kind
    if "@" in name or not re.fullmatch(r"[\w.-]+", name):
        raise HTTPException(400, "A provider's name can use letters, digits, dots and dashes; it's what follows @ in a model's name.")
    providers = added()
    if any(p["name"].lower() == name.lower() for p in providers):
        raise HTTPException(409, f"A provider called {name} is added already. Give this one another name.")
    provider = {"id": uuid.uuid4().hex[:12], "name": name, "kind": req.kind, "url": _url(req.kind, req.url), "key": req.key.strip(),
                "models": [m.strip() for m in req.models if m.strip()]}
    found, why = _ask(provider)
    if found is None and not provider["models"]:
        raise HTTPException(400, f"Couldn't add {KINDS[req.kind]['name']}: {why} If it doesn't list its models, name the ones to use.")
    providers.append(provider)
    with _lock:
        _save(providers)
        _seen[provider["id"]] = (time.time(), found, why)
    return public(provider, found, why)


@router.put("/{provider_id}/models")
def choose(provider_id: str, req: Chosen):
    """Which of the provider's models to list (none: all of them)."""
    providers = added()
    provider = next((p for p in providers if p["id"] == provider_id), None)
    if not provider:
        raise HTTPException(404, "No such provider.")
    provider["models"] = sorted({m.strip() for m in req.models if m.strip()})
    with _lock:
        _save(providers)
    return public(provider, *models(provider))


@router.delete("/{provider_id}")
def remove(provider_id: str):
    providers = added()
    if not any(p["id"] == provider_id for p in providers):
        raise HTTPException(404, "No such provider.")
    with _lock:
        _save([p for p in providers if p["id"] != provider_id])
        _seen.pop(provider_id, None)
    return list_providers()
