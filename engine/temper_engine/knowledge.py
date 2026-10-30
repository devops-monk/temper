"""Knowledge bases: folders and files a chat model can search, with the passages it found cited.

A knowledge base is a list of folders and files. Indexing reads each document (PDF, Word,
PowerPoint, Excel, Markdown, text, HTML and code), cuts it into passages of about a paragraph or
two, and keeps them in knowledge/<base>/index.db (SQLite) with
- an FTS5 full-text index, for exact words, names and numbers, and
- a vector for each passage, made by a small embedding model in llama.cpp, for meaning.

A search runs both and merges the rankings (reciprocal rank fusion), so a passage that matches on
either rises. Re-indexing reads only files that changed since last time.

The embedding model is a GGUF downloaded from Hugging Face (EMBEDDERS) and runs in its own
llama-server, started when needed and stopped after a few idle minutes; it never touches chat's model.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home, prefs
from .jobs import Job, registry

router = APIRouter(prefix="/api/knowledge", tags=["chat"])

CHUNK = 1_200  # characters in a passage, about 300 tokens
OVERLAP = 200
MAX_FILE = 40 * 1024 * 1024
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".org", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".tex", ".log"}
CODE_SUFFIXES = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".sql", ".lua", ".r", ".scala", ".vue", ".svelte", ".css", ".scss"}
DOC_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}
SUFFIXES = TEXT_SUFFIXES | CODE_SUFFIXES | DOC_SUFFIXES
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target", ".next", ".cache", ".idea", ".vscode"}
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")

EMBEDDERS = {
    "qwen3-embedding-0.6b": {
        "name": "Qwen3 Embedding 0.6B",
        "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "file": "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "megabytes": 640,
        "languages": "many",
        "query": "Instruct: Given a search query, retrieve passages that answer it\nQuery: ",
        "document": "",
    },
    "nomic-embed-1.5": {
        "name": "Nomic Embed Text 1.5",
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.Q8_0.gguf",
        "megabytes": 146,
        "languages": "en",
        "query": "search_query: ",
        "document": "search_document: ",
    },
    "bge-small-en-1.5": {
        "name": "BGE Small 1.5",
        "repo": "CompendiumLabs/bge-small-en-v1.5-gguf",
        "file": "bge-small-en-v1.5-q8_0.gguf",
        "megabytes": 37,
        "languages": "en",
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
}
DEFAULT_EMBEDDER = "qwen3-embedding-0.6b"


# --- Reading documents ----------------------------------------------------------------------------


def _xml_text(z: zipfile.ZipFile, name: str, tag: str) -> list[str]:
    """The text of each paragraph-like element `tag` in an Office XML part."""
    root = ElementTree.fromstring(z.read(name))
    out = []
    for p in root.iter():
        if p.tag.endswith("}" + tag):
            text = "".join(t.text or "" for t in p.iter() if t.tag.endswith("}t"))
            if text.strip():
                out.append(text)
    return out


def pptx_text(data: bytes) -> list[tuple[str, str]]:
    """Each slide's text, as (heading, text)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return [(f"Slide {i}", "\n".join(_xml_text(z, n, "p"))) for i, n in enumerate(slides, 1)]


def xlsx_text(data: bytes) -> list[tuple[str, str]]:
    """Each sheet as tab-separated rows, as (sheet name, text)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{ns}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
        names = [s.get("name") or "" for s in ElementTree.fromstring(z.read("xl/workbook.xml")).iter(f"{ns}sheet")]
        sheets = sorted((n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        out = []
        for i, n in enumerate(sheets):
            rows = []
            for row in ElementTree.fromstring(z.read(n)).iter(f"{ns}row"):
                cells = []
                for c in row.iter(f"{ns}c"):
                    v = c.find(f"{ns}v")
                    value = "".join(t.text or "" for t in c.iter(f"{ns}t")) if c.get("t") == "inlineStr" else (v.text or "") if v is not None else ""
                    if c.get("t") == "s" and value.isdigit() and int(value) < len(shared):
                        value = shared[int(value)]
                    cells.append(value)
                if any(cells):
                    rows.append("\t".join(cells))
            out.append((names[i] if i < len(names) else f"Sheet {i + 1}", "\n".join(rows)))
        return out


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Markdown split at its headings, each part under the heading it's in."""
    sections, heading, lines = [], "", []
    fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m:
            if any(x.strip() for x in lines):
                sections.append((heading, "\n".join(lines)))
            heading, lines = m.group(2), [line]
        else:
            lines.append(line)
    if any(x.strip() for x in lines):
        sections.append((heading, "\n".join(lines)))
    return sections


def read_document(path: Path) -> list[tuple[str, str]]:
    """A document's text as (heading, text) sections: pages, slides, sheets or Markdown sections."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return [(f"Page {i}", p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages, 1)]
    data = path.read_bytes()
    if suffix == ".docx":
        from .attachments import docx_text

        return _markdown_sections(docx_text(data))
    if suffix == ".pptx":
        return pptx_text(data)
    if suffix == ".xlsx":
        return xlsx_text(data)
    if suffix in (".html", ".htm"):
        from .web import readable

        title, text, _ = readable(data.decode("utf-8", errors="replace"), path.as_uri())
        return _markdown_sections(text) or [(title, text)]
    if b"\0" in data[:4096]:
        return []
    text = data.decode("utf-8", errors="replace")
    return _markdown_sections(text) if suffix in (".md", ".markdown") else [("", text)]


def chunks(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Passages of about CHUNK characters, cut at paragraph or line breaks, overlapping a little."""
    out = []
    for heading, text in sections:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        start = 0
        while start < len(text):
            end = min(start + CHUNK, len(text))
            if end < len(text):
                cut = max(text.rfind("\n\n", start + CHUNK // 2, end), text.rfind("\n", start + CHUNK // 2, end), text.rfind(". ", start + CHUNK // 2, end))
                end = cut + 1 if cut > 0 else end
            piece = text[start:end].strip()
            if piece:
                out.append((heading, piece))
            if end >= len(text):
                break
            start = max(end - OVERLAP, start + 1)
    return out


# --- The embedding model --------------------------------------------------------------------------


def embedder_id() -> str:
    chosen = prefs.load().get("embedder")
    return chosen if chosen in EMBEDDERS else DEFAULT_EMBEDDER


def embedder_path(eid: str) -> str | None:
    """The downloaded GGUF of an embedding model, if it's on this computer."""
    from huggingface_hub import try_to_load_from_cache

    e = EMBEDDERS[eid]
    found = try_to_load_from_cache(e["repo"], e["file"])
    return found if isinstance(found, str) and Path(found).is_file() else None


class Embedder:
    """A llama-server in embedding mode, started on first use and stopped when idle."""

    IDLE = 300  # seconds

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.model: str | None = None
        self.used = 0.0
        self.client: httpx.Client | None = None  # tests put a fake here

    def _start(self, path: str) -> None:
        from .llamacpp import binary
        from .runtime import _free_port

        exe = binary("llama-server")
        if not exe:
            raise ValueError("Knowledge search needs llama.cpp, which isn't installed. Install it from the Models page.")
        self.stop()
        self.port = _free_port()
        cmd = [exe, "-m", path, "--embedding", "--host", "127.0.0.1", "--port", str(self.port), "-c", "8192", "-b", "2048", "-ub", "2048", "-np", "4"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise ValueError(f"The embedding model stopped as it started (code {self.proc.returncode}).")
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1).status_code == 200:
                    self.model = path
                    threading.Thread(target=self._idle, daemon=True).start()
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        self.stop()
        raise ValueError("The embedding model took too long to start.")

    def _idle(self) -> None:
        proc = self.proc
        while proc is self.proc and proc and proc.poll() is None:
            time.sleep(15)
            if time.time() - self.used > self.IDLE:
                with self.lock:
                    if proc is self.proc:
                        self.stop()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc, self.model = None, None

    def embed(self, texts: list[str], eid: str, query: bool = False) -> np.ndarray:
        """Unit-length vectors for `texts`, one row each."""
        e = EMBEDDERS[eid]
        prefix = e["query"] if query else e["document"]
        inputs = [prefix + t[: CHUNK * 2] for t in texts]
        rows: list[list[float]] = []
        with self.lock:
            self.used = time.time()
            if self.client:
                post, url = self.client.post, "http://embedder/v1/embeddings"
            else:
                path = embedder_path(eid)
                if not path:
                    raise ValueError(f"The embedding model ({e['name']}) isn't downloaded yet: download it in Settings → Knowledge bases.")
                if self.model != path or not self.proc or self.proc.poll() is not None:
                    self._start(path)
                post, url = httpx.post, f"http://127.0.0.1:{self.port}/v1/embeddings"
            for i in range(0, len(inputs), 16):
                r = post(url, json={"input": inputs[i : i + 16], "model": eid}, timeout=300)
                if r.status_code != 200:
                    raise ValueError(f"The embedding model answered {r.status_code}: {r.text[:200]}")
                rows += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
            self.used = time.time()
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


embedder = Embedder()


# --- Bases and their index ------------------------------------------------------------------------


def root() -> Path:
    path = home() / "knowledge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dir(name: str) -> Path:
    if not NAME.match(name):
        raise HTTPException(404, f"There's no knowledge base called {name}.")
    return root() / name


def config(name: str) -> dict:
    try:
        return json.loads((_dir(name) / "base.json").read_text(encoding="utf-8"))
    except OSError:
        raise HTTPException(404, f"There's no knowledge base called {name}.") from None


def _save_config(name: str, cfg: dict) -> None:
    (_dir(name) / "base.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL, size INTEGER, passages INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS passages (id INTEGER PRIMARY KEY, file INTEGER, ord INTEGER, heading TEXT, text TEXT, vector BLOB);
CREATE INDEX IF NOT EXISTS passages_file ON passages(file);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(text, heading, content='passages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN INSERT INTO passages_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading); END;
CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN INSERT INTO passages_fts(passages_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading); END;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def db(name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_dir(name) / "index.db", timeout=30, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def files_in(paths: list[str]) -> list[Path]:
    """The documents in a base's folders and files that Temper can read."""
    out: list[Path] = []
    for p in paths:
        base = Path(p).expanduser()
        if base.is_file():
            out.append(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(filenames):
                path = Path(dirpath) / f
                if path.suffix.lower() in SUFFIXES and not f.startswith(".") and not f.startswith("~$"):
                    out.append(path)
    return out


def index(job: Job, name: str) -> None:
    cfg = config(name)
    eid = cfg["embedder"]
    conn = db(name)
    try:
        known = {row[0]: (row[1], row[2], row[3]) for row in conn.execute("SELECT path, mtime, size, id FROM files")}
        found = files_in(cfg["paths"])
        present = {str(p) for p in found}
        gone = [fid for path, (_, _, fid) in known.items() if path not in present]
        for fid in gone:
            conn.execute("DELETE FROM passages WHERE file = ?", (fid,))
            conn.execute("DELETE FROM files WHERE id = ?", (fid,))
        conn.commit()
        if gone:
            job.log(f"Removed {len(gone)} files that are gone")
        todo = []
        for p in found:
            try:
                st = p.stat()
            except OSError:
                continue
            old = known.get(str(p))
            if not old or old[0] != st.st_mtime or old[1] != st.st_size:
                todo.append((p, st))
        job.log(f"{len(found)} files, {len(todo)} to read")
        for i, (p, st) in enumerate(todo):
            if job.cancel_requested:
                return
            job.progress = i / max(len(todo), 1)
            job.emit("progress", value=job.progress, file=str(p))
            error, pieces = None, []
            if st.st_size > MAX_FILE:
                error = "too large"
            else:
                try:
                    pieces = chunks(read_document(p))
                except Exception as e:  # noqa: BLE001 — one unreadable file doesn't stop the index
                    error = f"couldn't read it: {e}"[:300]
            vectors = embedder.embed([f"{h}\n{t}" if h else t for h, t in pieces], eid) if pieces else None
            conn.execute("DELETE FROM passages WHERE file = (SELECT id FROM files WHERE path = ?)", (str(p),))
            conn.execute(
                "INSERT INTO files (path, mtime, size, passages, error) VALUES (?, ?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, passages=excluded.passages, error=excluded.error",
                (str(p), st.st_mtime, st.st_size, len(pieces), error),
            )
            fid = conn.execute("SELECT id FROM files WHERE path = ?", (str(p),)).fetchone()[0]
            conn.executemany(
                "INSERT INTO passages (file, ord, heading, text, vector) VALUES (?, ?, ?, ?, ?)",
                [(fid, j, h, t, vectors[j].tobytes()) for j, (h, t) in enumerate(pieces)],
            )
            conn.commit()
            if error:
                job.log(f"Skipped {p.name}: {error}")
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('indexed', ?), ('embedder', ?)", (str(time.time()), eid))
        conn.commit()
    finally:
        conn.close()
    job.finish("done", base=name)


def _fts_query(text: str) -> str:
    """The query's words for FTS5, each quoted, any of them matching."""
    words = re.findall(r"\w+", text.lower())
    return " OR ".join(f'"{w}"' for w in words[:32])


def search(name: str, query: str, limit: int = 6) -> list[dict]:
    """The passages that best match `query`, by words and by meaning, merged."""
    cfg = config(name)
    conn = db(name)
    try:
        by_words: list[int] = []
        if fts := _fts_query(query):
            by_words = [r[0] for r in conn.execute("SELECT rowid FROM passages_fts WHERE passages_fts MATCH ? ORDER BY bm25(passages_fts) LIMIT 40", (fts,))]
        rows = conn.execute("SELECT id, vector FROM passages").fetchall()
        by_meaning: list[int] = []
        if rows:
            q = embedder.embed([query], cfg["embedder"], query=True)[0]
            matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            if matrix.shape[1] == q.shape[0]:
                scores = matrix @ q
                top = np.argsort(-scores)[:40]
                by_meaning = [rows[i][0] for i in top]
        fused: dict[int, float] = {}
        for ranking in (by_words, by_meaning):
            for rank, pid in enumerate(ranking):
                fused[pid] = fused.get(pid, 0.0) + 1 / (60 + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:limit]
        out = []
        for pid in best:
            row = conn.execute("SELECT p.heading, p.text, f.path FROM passages p JOIN files f ON f.id = p.file WHERE p.id = ?", (pid,)).fetchone()
            if row:
                out.append({"id": pid, "heading": row[0], "text": row[1], "path": row[2], "score": round(fused[pid], 5)})
        return out
    finally:
        conn.close()


def bases() -> list[dict]:
    out = []
    for d in sorted(root().iterdir()):
        if not (d / "base.json").is_file():
            continue
        cfg = json.loads((d / "base.json").read_text(encoding="utf-8"))
        stats = {"files": 0, "passages": 0, "skipped": 0, "indexed": None}
        if (d / "index.db").is_file():
            conn = db(d.name)
            try:
                stats["files"], stats["passages"] = conn.execute("SELECT COUNT(*), COALESCE(SUM(passages), 0) FROM files WHERE error IS NULL").fetchone()
                stats["skipped"] = conn.execute("SELECT COUNT(*) FROM files WHERE error IS NOT NULL").fetchone()[0]
                indexed = conn.execute("SELECT value FROM meta WHERE key = 'indexed'").fetchone()
                stats["indexed"] = float(indexed[0]) if indexed else None
            finally:
                conn.close()
        running = next((j for j in registry.jobs.values() if j.kind == "knowledge-index" and j.status == "running" and j.result.get("base") == d.name), None)
        out.append({"name": d.name, **cfg, **stats, "job": running.id if running else None, "enabled": cfg.get("enabled", True)})
    return out


def results_text(query: str, names: list[str] | None = None, limit: int = 6) -> str:
    """The tool's result: the best passages across the bases, each with where it came from."""
    names = names or [b["name"] for b in bases() if b["enabled"]]
    if not names:
        return "There are no knowledge bases to search. Add one in Settings → Knowledge bases."
    found = []
    for n in names:
        try:
            found += [{**r, "base": n} for r in search(n, query, limit)]
        except HTTPException:
            continue
    found = sorted(found, key=lambda r: r["score"], reverse=True)[:limit]
    if not found:
        return f"Nothing in the knowledge bases matched {query!r}. Try other words."
    blocks = [f"[{i}] {r['path']}" + (f" ({r['heading']})" if r["heading"] else "") + f"\n{r['text']}" for i, r in enumerate(found, 1)]
    return (
        f"Passages from the person's documents for {query!r}. Cite them by file name when you use them.\n\n"
        + "\n\n---\n\n".join(blocks)
    )


# --- API ------------------------------------------------------------------------------------------


@router.get("")
def list_bases():
    current = embedder_id()
    return {
        "bases": bases(),
        "embedders": [{"id": k, **{x: e[x] for x in ("name", "megabytes", "languages")}, "downloaded": embedder_path(k) is not None} for k, e in EMBEDDERS.items()],
        "embedder": current,
        "llama_cpp": _llama_ready(),
    }


def _llama_ready() -> bool:
    from .llamacpp import binary

    return binary("llama-server") is not None


class NewBase(BaseModel):
    name: str
    paths: list[str] = Field(min_length=1)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:48]


def _start_index(name: str) -> str:
    running = next((j for j in registry.jobs.values() if j.kind == "knowledge-index" and j.status == "running" and j.result.get("base") == name), None)
    if running:
        return running.id
    if not embedder_path(config(name)["embedder"]) and not embedder.client:
        raise HTTPException(409, "Download the embedding model first (Settings → Knowledge bases).")
    job = registry.create("knowledge-index", f"Index {name}")
    job.result["base"] = name
    registry.run_thread(job, lambda j: index(j, name))
    return job.id


@router.post("")
def create_base(req: NewBase):
    name = _slug(req.name)
    if not NAME.match(name):
        raise HTTPException(422, "Give the knowledge base a name with some letters or digits in it.")
    if (root() / name).exists():
        raise HTTPException(409, f"There's already a knowledge base called {name}.")
    paths = [str(Path(p).expanduser().resolve()) for p in req.paths]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise HTTPException(422, f"{missing[0]} doesn't exist.")
    (root() / name).mkdir()
    _save_config(name, {"title": req.name.strip(), "paths": paths, "embedder": embedder_id(), "enabled": True, "created": time.time()})
    return {"base": name, "job": _start_index(name)}


class EditBase(BaseModel):
    paths: list[str] | None = None
    enabled: bool | None = None


@router.put("/{name}")
def edit_base(name: str, req: EditBase):
    cfg = config(name)
    if req.paths is not None:
        cfg["paths"] = [str(Path(p).expanduser().resolve()) for p in req.paths]
    if req.enabled is not None:
        cfg["enabled"] = req.enabled
    _save_config(name, cfg)
    return {"base": name, "job": _start_index(name) if req.paths is not None else None}


@router.post("/{name}/index")
def reindex(name: str):
    return {"job": _start_index(name)}


@router.delete("/{name}")
def delete_base(name: str):
    config(name)
    shutil.rmtree(_dir(name))
    return list_bases()


@router.get("/{name}/search")
def search_base(name: str, q: str, limit: int = 6):
    try:
        return {"results": search(name, q, max(1, min(limit, 20)))}
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


class ChooseEmbedder(BaseModel):
    id: str


@router.post("/embedder")
def choose_embedder(req: ChooseEmbedder):
    """Makes `id` the embedding model for new bases, downloading it if it isn't here."""
    if req.id not in EMBEDDERS:
        raise HTTPException(404, f"Temper doesn't know the embedding model {req.id}.")
    prefs.update("embedder", req.id)
    if embedder_path(req.id):
        return {"job": None}
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    from . import downloads

    e = EMBEDDERS[req.id]
    try:
        size = get_hf_file_metadata(hf_hub_url(e["repo"], e["file"])).size or e["megabytes"] * 1024 * 1024
    except Exception:  # noqa: BLE001 — the size only drives the progress bar
        size = e["megabytes"] * 1024 * 1024
    job = registry.create("download", f"Embedding model: {e['name']}")
    registry.run_thread(job, lambda j: downloads.download(j, e["repo"], [(e["file"], size)]))
    return {"job": job.id}
