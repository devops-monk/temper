"""Datasets at a web address: https:// links, and public files in S3 (s3://) or Google Cloud Storage (gs://).

A remote file is downloaded once into Temper's cache and from then on read like a local one, so
preparing, previewing and training all work the same. Previews of JSONL and CSV stream only the first
rows instead of the whole file.

Private S3 buckets work with the credentials the AWS tools use: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
(and AWS_SESSION_TOKEN), or a profile in ~/.aws/credentials (AWS_PROFILE, else "default"), with the region from
AWS_REGION or ~/.aws/config. Requests are signed with AWS Signature Version 4. AWS_ENDPOINT_URL_S3 points s3://
at an S3-compatible service instead (Cloudflare R2, MinIO, Backblaze B2). Private gs:// files aren't supported.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import hmac
import io
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import islice
from urllib.parse import quote
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


# --- S3 credentials and signing --------------------------------------------------------------------


def s3_credentials() -> dict | None:
    """The AWS tools' credentials and region: from the environment, else ~/.aws (AWS_PROFILE, or "default")."""
    profile = os.environ.get("AWS_PROFILE") or "default"
    creds = {"key": os.environ.get("AWS_ACCESS_KEY_ID"), "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"), "token": os.environ.get("AWS_SESSION_TOKEN")}
    aws = Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or Path.home() / ".aws" / "credentials")
    if not (creds["key"] and creds["secret"]) and aws.is_file():
        files = configparser.ConfigParser()
        files.read(aws)
        if files.has_section(profile):
            section = files[profile]
            creds = {"key": section.get("aws_access_key_id"), "secret": section.get("aws_secret_access_key"), "token": section.get("aws_session_token")}
    if not (creds["key"] and creds["secret"]):
        return None
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    config = Path(os.environ.get("AWS_CONFIG_FILE") or Path.home() / ".aws" / "config")
    if not region and config.is_file():
        files = configparser.ConfigParser()
        files.read(config)
        section = "default" if profile == "default" else f"profile {profile}"
        region = files.get(section, "region", fallback=None) if files.has_section(section) else None
    return {**creds, "region": region or "us-east-1"}


def sign(method: str, address: str, creds: dict, headers: dict | None = None, now: datetime | None = None) -> dict:
    """The headers that sign a request with AWS Signature Version 4, for S3 (the payload left unsigned)."""
    now = now or datetime.now(timezone.utc)
    stamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    parts = urlparse(address)
    host = parts.netloc
    out = {**(headers or {}), "host": host, "x-amz-date": stamp, "x-amz-content-sha256": (headers or {}).get("x-amz-content-sha256", "UNSIGNED-PAYLOAD")}
    if creds.get("token"):
        out["x-amz-security-token"] = creds["token"]
    names = sorted(k.lower() for k in out)
    lower = {k.lower(): str(v).strip() for k, v in out.items()}
    canonical = "\n".join([
        method,
        quote(parts.path or "/", safe="/~"),
        "&".join(sorted(f"{quote(k, safe='~')}={quote(v, safe='~')}" for k, _, v in (p.partition("=") for p in parts.query.split("&") if p))),
        "".join(f"{n}:{lower[n]}\n" for n in names),
        ";".join(names),
        lower["x-amz-content-sha256"],
    ])
    scope = f"{day}/{creds['region']}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = f"AWS4{creds['secret']}".encode()
    for part in (day, creds["region"], "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    out.pop("host")
    out["Authorization"] = f"AWS4-HMAC-SHA256 Credential={creds['key']}/{scope}, SignedHeaders={';'.join(names)}, Signature={signature}"
    return out


def _s3_address(bucket: str, key: str, region: str) -> str:
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:  # S3-compatible services take the bucket in the path
        return f"{endpoint.rstrip('/')}/{bucket}/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


@contextmanager
def _open(source: str):
    """A streamed GET of the source: an s3:// one signed when there are credentials, following the bucket's region."""
    m = re.match(r"s3://([^/]+)/(.+)", source.strip(), re.I)
    creds = s3_credentials() if m else None
    if not creds:
        with httpx.stream("GET", url(source), follow_redirects=True, timeout=60) as r:
            yield r
        return
    bucket, key = m.group(1), m.group(2)
    region = creds["region"]
    for _ in range(2):
        address = _s3_address(bucket, key, region)
        with httpx.stream("GET", address, headers=sign("GET", address, {**creds, "region": region}), timeout=60) as r:
            moved = r.headers.get("x-amz-bucket-region")
            if r.status_code in (301, 400) and moved and moved != region:
                region = moved  # the bucket lives elsewhere: sign again for its region
                continue
            yield r
            return


def _error(r: httpx.Response, address: str) -> ValueError:
    if r.status_code in (401, 403):
        hint = " With AWS credentials (environment or ~/.aws), Temper signs s3:// requests." if address.startswith("s3://") or ".s3." in address else ""
        return ValueError(f"{address} isn't public, or the credentials can't read it (the server said {r.status_code}).{hint} Or download it yourself and choose the file instead.")
    return ValueError(f"Couldn't fetch {address}: the server said {r.status_code}.")


def fetch(source: str, refresh: bool = False, log=None) -> Path:
    """The local copy of a remote dataset file, downloading it first if needed."""
    path = cached(source)
    if path.is_file() and not refresh:
        return path
    address = source.strip() if source.strip().lower().startswith("s3://") else url(source)
    tmp = path.with_suffix(path.suffix + ".part")
    with _open(source) as r:
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
    address = source.strip() if source.strip().lower().startswith("s3://") else url(source)
    lines: list[str] = []
    with _open(source) as r:
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
