"""Hugging Face Hub: search, inspect repos, list the local cache and download files."""

from __future__ import annotations

import fnmatch

from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError
from pydantic import BaseModel

from .jobs import Job, registry

router = APIRouter(prefix="/api/models", tags=["models"])
api = HfApi()


@router.get("/search")
def search(q: str = "", author: str | None = "unsloth", sort: str = "downloads", limit: int = 30, gguf: bool = False):
    try:
        models = api.list_models(
            search=q or None,
            author=author or None,
            sort=sort,
            limit=min(limit, 100),
            filter="gguf" if gguf else None,
            full=False,
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


@router.get("/repo")
def repo_files(id: str):
    """Files in a repo with sizes, so the UI can offer a specific GGUF quant."""
    try:
        info = api.model_info(id, files_metadata=True)
    except HfHubHTTPError as exc:
        raise HTTPException(404, f"Could not open {id}: {exc}") from exc
    files = [{"name": s.rfilename, "size": s.size or 0} for s in info.siblings or []]
    return {
        "id": id,
        "gated": bool(info.gated),
        "files": files,
        "gguf": [f for f in files if f["name"].endswith(".gguf") and "mmproj" not in f["name"].lower()],
    }


@router.get("/local")
def local_models():
    try:
        cache = scan_cache_dir()
    except Exception:
        return []
    models = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        revision = max(repo.revisions, key=lambda r: r.last_modified, default=None)
        files = [
            {"name": f.file_name, "path": str(f.file_path), "size": f.size_on_disk}
            for f in (revision.files if revision else [])
        ]
        models.append({
            "id": repo.repo_id,
            "size": repo.size_on_disk,
            "last_used": repo.last_accessed,
            "path": str(revision.snapshot_path) if revision else None,
            "gguf": [f for f in files if f["name"].endswith(".gguf") and "mmproj" not in f["name"].lower()],
            "files": len(files),
            "safetensors": any(f["name"].endswith(".safetensors") for f in files),
        })
    return sorted(models, key=lambda m: m["last_used"] or 0, reverse=True)


class DownloadRequest(BaseModel):
    repo_id: str
    # Glob patterns, e.g. ["*Q4_K_M*.gguf"]. Empty means the whole repo.
    patterns: list[str] = []


def _download(job: Job, req: DownloadRequest) -> None:
    info = api.model_info(req.repo_id, files_metadata=True)
    files = [s for s in info.siblings or [] if not req.patterns or any(fnmatch.fnmatch(s.rfilename, p) for p in req.patterns)]
    if not files:
        raise ValueError("No files in the repo match the selection")
    total = sum(s.size or 0 for s in files) or 1
    done = 0
    job.emit("plan", files=len(files), bytes=total)
    for s in files:
        if job.cancel_requested:
            return
        job.log(f"Downloading {s.rfilename}")
        path = hf_hub_download(req.repo_id, s.rfilename)
        done += s.size or 0
        job.progress = done / total
        job.emit("progress", value=job.progress, file=s.rfilename)
        job.result["path"] = path if s.rfilename.endswith(".gguf") else job.result.get("path", path)


@router.post("/download")
def download(req: DownloadRequest):
    title = req.repo_id + (f" ({', '.join(req.patterns)})" if req.patterns else "")
    job = registry.create("download", title)
    job.progress = 0.0
    registry.run_thread(job, lambda j: _download(j, req))
    return job.summary()
