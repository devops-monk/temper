"""Resumable model downloads into the Hugging Face cache, with byte-level progress.

huggingface_hub starts every attempt from zero, which hurts for 10 GB GGUFs. Here each file streams
into `blobs/<sha256>.temper-partial` with HTTP Range requests, so a cancelled, failed or interrupted
download picks up where it stopped. The finished blob is checked against its SHA-256, renamed into
place, and `hf_hub_download` then only links it into the snapshot, keeping the cache standard.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import httpx
from huggingface_hub import constants, get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from .jobs import Job

CHUNK = 1024 * 1024
RETRIES = 5


class Cancelled(Exception):
    pass


def repo_cache(repo_id: str) -> Path:
    return Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


class Progress:
    """Bytes across all files of one job, with speed averaged over the last few seconds."""

    def __init__(self, job: Job, total: int):
        self.job, self.total, self.done = job, total, 0
        self.samples: list[tuple[float, int]] = []
        self.last_emit = 0.0

    def add(self, n: int, file: str, force: bool = False) -> None:
        self.done += n
        now = time.monotonic()
        if not force and now - self.last_emit < 0.5:
            return
        self.last_emit = now
        self.samples = [(t, b) for t, b in self.samples if now - t < 5] + [(now, self.done)]
        t0, b0 = self.samples[0]
        speed = (self.done - b0) / (now - t0) if now > t0 else 0
        self.job.progress = self.done / self.total if self.total else None
        self.job.result.update(received=self.done, total=self.total, speed=speed, file=file)
        self.job.emit("progress", value=self.job.progress, received=self.done, total=self.total, speed=speed, file=file)


def _hash_existing(path: Path, digest) -> int:
    size = 0
    with path.open("rb") as f:
        while block := f.read(8 * CHUNK):
            digest.update(block)
            size += len(block)
    return size


def fetch(job: Job, repo_id: str, filename: str, progress: Progress) -> str:
    """Downloads one file (resuming a partial one) and returns its path in the snapshot."""
    url = hf_hub_url(repo_id, filename)
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        meta = get_hf_file_metadata(url, token=token)
    except GatedRepoError as e:
        hint = "then download again" if token else "then add a Hugging Face token in Settings"
        raise RuntimeError(f"{repo_id} asks you to accept its licence first. Accept it at https://huggingface.co/{repo_id}, {hint}.") from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(f"{repo_id} is private or doesn't exist." + ("" if token else " If it's yours, add a Hugging Face token in Settings.")) from e
    etag = (meta.etag or "").strip('"').removeprefix("W/").strip('"')
    sha256 = etag if re.fullmatch(r"[0-9a-f]{64}", etag) else None
    blobs = repo_cache(repo_id) / "blobs"
    blob = blobs / etag
    if blob.exists() and (meta.size is None or blob.stat().st_size == meta.size):
        progress.add(meta.size or 0, filename, force=True)
    else:
        blobs.mkdir(parents=True, exist_ok=True)
        partial = blobs / f"{etag}.temper-partial"
        digest = hashlib.sha256()
        have = _hash_existing(partial, digest) if partial.exists() else 0
        if have:
            job.log(f"Resuming {filename} at {have / 1024**3:.2f} GB")
        progress.add(have, filename, force=True)
        attempt = 0
        while meta.size is None or have < meta.size:
            try:
                with httpx.stream("GET", url, headers={**headers, "Range": f"bytes={have}-"}, follow_redirects=True, timeout=httpx.Timeout(30, read=60)) as r:
                    if r.status_code == 416:  # already complete
                        break
                    r.raise_for_status()
                    if have and r.status_code == 200:  # the server ignored Range: start over
                        have, digest = 0, hashlib.sha256()
                        partial.unlink(missing_ok=True)
                        progress.done = 0
                    with partial.open("ab") as out:
                        for chunk in r.iter_bytes(CHUNK):
                            if job.cancel_requested:
                                raise Cancelled
                            out.write(chunk)
                            digest.update(chunk)
                            have += len(chunk)
                            progress.add(len(chunk), filename)
                if meta.size is None:
                    break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403, 404):
                    raise
                attempt += 1
                if attempt > RETRIES:
                    raise RuntimeError(f"The download of {filename} kept failing ({e}). Resume it once your connection is back.") from e
                job.log(f"Connection problem ({type(e).__name__}); retrying in {2**attempt} s")
                time.sleep(2**attempt)
        if sha256 and digest.hexdigest() != sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{filename} didn't match its checksum and was deleted. Download it again.")
        os.replace(partial, blob)
    # The blob is in place, so this only creates the snapshot link and refs (one metadata request).
    return hf_hub_download(repo_id, filename)


def download(job: Job, repo_id: str, files: list[tuple[str, int]]) -> None:
    progress = Progress(job, sum(size for _, size in files))
    job.emit("plan", files=len(files), bytes=progress.total)
    for name, _size in files:
        job.log(f"Downloading {name}")
        try:
            path = fetch(job, repo_id, name, progress)
        except Cancelled:
            job.log("Paused. Resume picks up from here.")
            return
        if name.endswith(".gguf"):
            job.result["path"] = path
        job.result.setdefault("path", path)
    progress.add(0, files[-1][0], force=True)
