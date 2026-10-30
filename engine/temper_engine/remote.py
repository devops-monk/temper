"""Datasets at a web address: https:// links, and public files in S3 (s3://) or Google Cloud Storage (gs://).

A remote file is downloaded once into Temper's cache and from then on read like a local one, so
preparing, previewing and training all work the same. Previews of JSONL and CSV stream only the first
rows instead of the whole file. Private buckets would need credentials and a signing client, which
Temper doesn't have yet: their addresses fail with the storage service's own "access denied".
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import home

SUFFIXES = (".jsonl", ".json", ".csv", ".parquet")
SCHEMES = ("http://", "https://", "s3://", "gs://")


def is_remote(source: str) -> bool:
    return source.strip().lower().startswith(SCHEMES)


def url(source: str) -> str:
    """The https address of a source: s3://bucket/key and gs://bucket/key as their public URLs."""
    source = source.strip()
    if m := re.match(r"s3://([^/]+)/(.+)", source, re.I):
        return f"https://{m.group(1)}.s3.amazonaws.com/{m.group(2)}"
    if m := re.match(r"gs://([^/]+)/(.+)", source, re.I):
        return f"https://storage.googleapis.com/{m.group(1)}/{m.group(2)}"
    return source


def suffix(source: str) -> str:
    """The file type from the address's path (".jsonl", ".csv", ...), ignoring any query string."""
    ext = Path(urlparse(url(source)).path).suffix.lower()
    if ext not in SUFFIXES:
        raise ValueError(f"Temper reads .jsonl, .json, .csv and .parquet files from the web; this address ends in {ext or 'no file type'}.")
    return ext


def cached(source: str) -> Path:
    key = hashlib.sha256(url(source).encode()).hexdigest()[:24]
    folder = home() / "cache" / "remote"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}{suffix(source)}"


def _error(r: httpx.Response, address: str) -> ValueError:
    if r.status_code in (401, 403):
        return ValueError(f"{address} isn't public (the server said {r.status_code}). Download it yourself and choose the file instead.")
    return ValueError(f"Couldn't fetch {address}: the server said {r.status_code}.")


def fetch(source: str, refresh: bool = False, log=None) -> Path:
    """The local copy of a remote dataset file, downloading it first if needed."""
    path = cached(source)
    if path.is_file() and not refresh:
        return path
    address = url(source)
    tmp = path.with_suffix(path.suffix + ".part")
    with httpx.stream("GET", address, follow_redirects=True, timeout=60) as r:
        if r.status_code != 200:
            raise _error(r, address)
        total = int(r.headers.get("content-length") or 0)
        done, step = 0, 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if log and total and done * 10 // total > step:
                    step = done * 10 // total
                    log(f"Downloaded {done * 100 // total}% of {address}")
    tmp.replace(path)
    return path


def head_rows(source: str, limit: int) -> list[dict]:
    """The first rows of a remote file, streaming only as much of it as that takes (JSONL and CSV)."""
    ext = suffix(source)
    if ext in (".json", ".parquet"):
        from .datasets import _local

        return _local(fetch(source), limit) if ext == ".json" else _parquet(fetch(source), limit)
    address = url(source)
    lines: list[str] = []
    with httpx.stream("GET", address, follow_redirects=True, timeout=60) as r:
        if r.status_code != 200:
            raise _error(r, address)
        for line in r.iter_lines():
            if line.strip():
                lines.append(line)
            if len(lines) > limit:  # one spare for a CSV's header
                break
    if ext == ".csv":
        return list(islice(csv.DictReader(io.StringIO("\n".join(lines))), limit))
    return [json.loads(line) for line in lines[:limit]]


def _parquet(path: Path, limit: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ValueError("Reading Parquet needs training support. Add it in Settings.") from None
    return pq.ParquetFile(path).read_row_groups([0]).slice(0, limit).to_pylist()
