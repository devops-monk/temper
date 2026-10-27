"""Model hub: search, inspect repos, list the local cache and download files, from the source chosen in Settings."""

from __future__ import annotations

import fnmatch
import re

import httpx

from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError
from pydantic import BaseModel

from . import downloads, engines, sources, storage
from .runtime import PROJECTOR_PREFERENCE
from .jobs import Job, registry

router = APIRouter(prefix="/api/models", tags=["models"])

SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def group_gguf(files: list[dict]) -> list[dict]:
    """One entry per model file: split GGUFs (`x-00001-of-00003.gguf`) become a single entry with the
    total size, every part's name, and the first part's name and path (llama.cpp loads the rest)."""
    groups: dict[str, dict] = {}
    for f in sorted(files, key=lambda f: f["name"]):
        name = f["name"]
        if not name.endswith(".gguf") or "mmproj" in name.lower():
            continue
        m = SPLIT.search(name)
        key = SPLIT.sub(".gguf", name) if m else name
        g = groups.setdefault(key, {**f, "size": 0, "parts": [], "split": int(m.group(2)) if m else 1})
        g["size"] += f["size"] or 0
        g["parts"].append(name)
    for g in groups.values():
        g["complete"] = len(g["parts"]) == g["split"]
    return sorted(groups.values(), key=lambda g: g["size"])


@router.get("/search")
def search(q: str = "", author: str | None = None, sort: str = "downloads", limit: int = 30, gguf: bool = False):
    if sources.modelscope():
        try:
            return sources.ms_search(q, author, sort, limit, gguf)
        except (httpx.HTTPError, sources.NotAvailable) as exc:
            raise HTTPException(502, f"ModelScope search failed: {exc}") from exc
    try:
        # A new client each time: the endpoint changes when you pick a mirror.
        models = HfApi().list_models(
            search=q or None,
            author=author or None,
            sort=sort,
            limit=min(limit, 100),
            filter="gguf" if gguf else None,
            # Without expand the Hub leaves out the date and gating.
            expand=["downloads", "likes", "lastModified", "pipeline_tag", "tags", "gated"],
        )
        return [
            {
                "id": m.id,
                "downloads": m.downloads,
                "likes": m.likes,
                "updated": m.last_modified.isoformat() if m.last_modified else None,
                "pipeline": m.pipeline_tag,
                "tags": (m.tags or [])[:12],
                "gated": bool(getattr(m, "gated", False)),
            }
            for m in models
        ]
    except HfHubHTTPError as exc:
        raise HTTPException(502, f"Hugging Face search failed: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Couldn't reach {sources.endpoint()}: {exc}") from exc


def list_files(repo_id: str) -> list[dict]:
    """Every file in a repo with its size, from the active source."""
    if sources.modelscope():
        return sources.ms_files(repo_id)
    info = HfApi().model_info(repo_id, files_metadata=True)
    return [{"name": s.rfilename, "size": s.size or 0} for s in info.siblings or []]


@router.get("/repo")
def repo_files(id: str):
    """Files in a repo with sizes, so the UI can offer a specific GGUF quant."""
    try:
        files = list_files(id)
    except sources.NotAvailable as exc:
        raise HTTPException(404, f"{id} isn't on ModelScope, or it's private. Many models are published there under another name; search for it.") from exc
    except (HfHubHTTPError, httpx.HTTPError) as exc:
        raise HTTPException(404, f"Could not open {id}: {exc}") from exc
    return {
        "id": id,
        "files": files,
        "gguf": group_gguf(files),
    }


@router.get("/local")
def local_models():
    from . import library  # imports hub

    try:
        repos = scan_cache_dir(storage.models_dir()).repos  # it binds the default folder when imported
    except Exception:  # no cache yet: a new install, or nothing downloaded
        repos = []
    models = []
    for repo in repos:
        if repo.repo_type != "model":
            continue
        revision = max(repo.revisions, key=lambda r: r.last_modified, default=None)
        files = [
            {"name": f.file_name, "path": str(f.file_path), "size": f.size_on_disk}
            for f in (revision.files if revision else [])
        ]
        safetensors = any(f["name"].endswith(".safetensors") for f in files)
        models.append({
            "id": repo.repo_id,
            "size": repo.size_on_disk,
            "last_used": repo.last_accessed,
            "path": str(revision.snapshot_path) if revision else None,
            "gguf": group_gguf(files),
            "files": len(files),
            "safetensors": safetensors,
            **(library.checkpoint(revision.snapshot_path) if revision and safetensors else {"quantization": None, "convertible": False}),
            "chat": bool(revision and safetensors and engines.chat_model(str(revision.snapshot_path))),
        })
    return library.entries() + sorted(models, key=lambda m: m["last_used"] or 0, reverse=True)


class DownloadRequest(BaseModel):
    repo_id: str
    # Glob patterns, e.g. ["*Q4_K_M*.gguf"]. Empty means the whole repo.
    patterns: list[str] = []


def _download(job: Job, req: DownloadRequest) -> None:
    listing = list_files(req.repo_id)
    files = [(f["name"], f["size"]) for f in listing if not req.patterns or any(fnmatch.fnmatch(f["name"], p) for p in req.patterns)]
    if not files:
        raise ValueError("No files in the repo match the selection")
    # A vision or audio model's GGUF needs its projector beside it; fetch one with the first quant.
    projectors = [(f["name"], f["size"]) for f in listing if f["name"].endswith(".gguf") and "mmproj" in f["name"].lower()]
    if projectors and any(n.endswith(".gguf") and "mmproj" not in n.lower() for n, _ in files) and not any("mmproj" in n.lower() for n, _ in files):
        projector = next((p for tag in PROJECTOR_PREFERENCE for p in projectors if re.search(rf"(^|[-_.]){tag}([-_.]|$)", p[0].lower())), projectors[0])
        files.append(projector)
    downloads.download(job, req.repo_id, files)


@router.post("/download")
def download(req: DownloadRequest):
    title = req.repo_id + (f" ({', '.join(req.patterns)})" if req.patterns else "")
    job = registry.create("download", title)
    job.progress = 0.0
    # Kept so the UI can resume a paused or failed download with the same selection.
    job.result["request"] = req.model_dump()
    registry.run_thread(job, lambda j: _download(j, req))
    return job.summary()
