"""Where models come from: Hugging Face, a Hugging Face mirror, or ModelScope.

A mirror (hf-mirror.com, or a company's own) speaks the Hugging Face API, so it only changes the
endpoint: here for the engine, and through HF_ENDPOINT for the training and export workers.

ModelScope has its own API. Search and file lists come from it, and its downloads are written into
the Hugging Face cache under the same repo id (blob named by SHA-256, snapshot link, refs/main), so
the library, chat, the memory estimate and training treat them like any other cached model.
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException
from huggingface_hub import constants
from pydantic import BaseModel

from . import prefs

router = APIRouter(prefix="/api/hub", tags=["hub"])

Source = Literal["huggingface", "mirror", "modelscope"]
HUGGING_FACE = "https://huggingface.co"
DEFAULT_MIRROR = "https://hf-mirror.com"
MODELSCOPE = "https://www.modelscope.cn"
MS_PAGE = 50  # ModelScope refuses larger pages
MS_SORT = {"downloads": "downloads", "likes": "likes", "lastModified": "last_modified", "last_modified": "last_modified"}


class HubSettings(BaseModel):
    source: Source = "huggingface"
    mirror: str = DEFAULT_MIRROR


def settings() -> HubSettings:
    try:
        return HubSettings(**prefs.load().get("hub", {}))
    except ValueError:
        return HubSettings()


def active() -> Source:
    return settings().source


def modelscope() -> bool:
    return active() == "modelscope"


def endpoint(s: HubSettings | None = None) -> str:
    s = s or settings()
    return s.mirror.rstrip("/") if s.source == "mirror" else HUGGING_FACE


def apply(s: HubSettings | None = None) -> None:
    """Points huggingface_hub (and, through the environment, every worker) at the right endpoint."""
    url = endpoint(s)
    constants.ENDPOINT = url
    constants.HUGGINGFACE_CO_URL_TEMPLATE = url + "/{repo_id}/resolve/{revision}/{filename}"
    if url == HUGGING_FACE:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = url


def _clean_mirror(raw: str) -> str:
    url = raw.strip().rstrip("/") or DEFAULT_MIRROR
    if "://" not in url:
        url = "https://" + url
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise HTTPException(400, "Enter the mirror's address, for example https://hf-mirror.com.")
    return url


@router.get("/source")
def get_source():
    return {**settings().model_dump(), "endpoint": endpoint()}


@router.put("/source")
def set_source(s: HubSettings):
    s.mirror = _clean_mirror(s.mirror)
    prefs.update("hub", s.model_dump())
    apply(s)
    return get_source()


@router.get("/source/check")
def check_source(source: Source | None = None, mirror: str | None = None):
    """Whether a source answers from this machine, and how quickly."""
    s = HubSettings(source=source or active(), mirror=_clean_mirror(mirror) if mirror else settings().mirror)
    url = f"{MODELSCOPE}/openapi/v1/models?page_size=1" if s.source == "modelscope" else f"{endpoint(s)}/api/models?limit=1"
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        ok = r.status_code < 500
        return {"ok": ok, "ms": round(r.elapsed.total_seconds() * 1000), "status": r.status_code}
    except httpx.HTTPError as e:
        return {"ok": False, "error": type(e).__name__}


# --- ModelScope ---------------------------------------------------------------------------------


class NotAvailable(LookupError):
    """The repo is missing, private or gated on ModelScope."""


def _ms_get(path: str, **params) -> dict:
    r = httpx.get(MODELSCOPE + path, params=params, timeout=httpx.Timeout(20, connect=10), follow_redirects=True)
    if r.status_code in (401, 403, 404):
        raise NotAvailable(path)
    r.raise_for_status()
    return r.json()


def ms_search(q: str, author: str | None, sort: str, limit: int, gguf: bool) -> list[dict]:
    terms = " ".join(t for t in (q.strip(), "gguf" if gguf and "gguf" not in q.lower() else "") if t)
    params: dict = {"page_size": min(limit, MS_PAGE), "page_number": 1, "sort": MS_SORT.get(sort, "downloads")}
    if terms:
        params["search"] = terms
    if author:
        params["author"] = author
    body = _ms_get("/openapi/v1/models", **params).get("data") or {}
    results = []
    for m in body.get("models") or []:
        tags = [t for t in m.get("tags") or [] if isinstance(t, str)]
        if gguf and "library:gguf" not in tags:
            continue
        results.append({
            "id": m.get("id"),
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "updated": m.get("last_modified") or m.get("created_at"),
            "pipeline": (m.get("tasks") or [None])[0],
            # "library:gguf" → "gguf", like the Hub's plain tags.
            "tags": [t.split(":", 1)[1] for t in tags if t.startswith(("library:", "custom_tag:"))][:12],
            "gated": bool(m.get("gated")),
        })
    return results


def ms_files(repo_id: str) -> list[dict]:
    data = _ms_get(f"/api/v1/models/{repo_id}/repo/files", Revision="master", Recursive="true").get("Data") or {}
    return [{"name": f["Path"], "size": f.get("Size") or 0, "sha256": f.get("Sha256")} for f in data.get("Files") or [] if f.get("Type") == "blob"]


def ms_commit(repo_id: str) -> str:
    """The commit `master` points at. The REST API doesn't name it; git's ref advertisement does."""
    try:
        r = httpx.get(f"{MODELSCOPE}/{repo_id}.git/info/refs", params={"service": "git-upload-pack"}, headers={"User-Agent": "git/2.45.0"}, timeout=20)
        for line in r.content.split(b"\n"):
            parts = line.split(b" ")
            if len(parts) >= 2 and parts[1].split(b"\0")[0] == b"refs/heads/master":
                return parts[0][-40:].decode()
    except httpx.HTTPError:
        pass
    return "master"


def ms_url(repo_id: str, filename: str) -> str:
    return f"{MODELSCOPE}/models/{repo_id}/resolve/master/{quote(filename)}"


def ms_page(repo_id: str) -> str:
    return f"{MODELSCOPE}/models/{repo_id}"


def ms_config(repo_id: str) -> dict:
    r = httpx.get(ms_url(repo_id, "config.json"), timeout=20, follow_redirects=True)
    if r.status_code in (401, 403, 404):
        raise NotAvailable(repo_id)
    r.raise_for_status()
    return r.json()
