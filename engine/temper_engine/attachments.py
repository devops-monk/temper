"""Chat attachments: images and audio for models that take them, and documents read into text.

Each attachment is copied into attachments/<id>/ in Temper's home with a meta.json beside it, and a
document's text in text.txt, so a saved chat can send its files again on later turns. Chat messages
refer to one with a content part {"type": "attachment", "id": ...}; /v1/chat/completions expands it
into what llama.cpp takes: an image_url or input_audio part, or the document's text.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import mimetypes
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import home

router = APIRouter(prefix="/api/attachments", tags=["chat"])

MAX_BYTES = 50 * 1024**2
# Enough for a long report; past this the text is cut, and the model's context is usually the tighter limit.
MAX_CHARS = 400_000
# What llama.cpp's image loader (stb_image) reads.
IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".bmp": "image/bmp"}
AUDIO = {".wav": "wav", ".mp3": "mp3"}
UNSUPPORTED = {
    ".webp": "llama.cpp can't read WebP images. Save it as PNG or JPEG and attach that.",
    ".heic": "llama.cpp can't read HEIC photos. Export it as JPEG and attach that.",
    ".m4a": "Models take WAV or MP3 audio. Convert it to one of those and attach that.",
    ".flac": "Models take WAV or MP3 audio. Convert it to one of those and attach that.",
    ".ogg": "Models take WAV or MP3 audio. Convert it to one of those and attach that.",
    ".doc": "Old Word files (.doc) can't be read. Save it as .docx or PDF and attach that.",
    ".xlsx": "Spreadsheets can't be read yet. Save the sheet as CSV and attach that.",
    ".pptx": "Slides can't be read yet. Save them as PDF and attach that.",
}


def root() -> Path:
    return home() / "attachments"


def _folder(attachment_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", attachment_id):
        raise HTTPException(404, "No such attachment.")
    folder = root() / attachment_id
    if not (folder / "meta.json").is_file():
        raise HTTPException(404, "That attachment isn't there any more.")
    return folder


def meta(attachment_id: str) -> dict:
    return json.loads((_folder(attachment_id) / "meta.json").read_text())


def file_path(attachment_id: str) -> Path:
    return _folder(attachment_id) / meta(attachment_id)["file"]


# --- Reading documents ----------------------------------------------------------------------------


OCR_PAGES = 20  # scanned pages read when a PDF is attached; a knowledge base reads them all


def pdf_text(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("The PDF is password-protected. Remove the password and attach it again.")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except PdfReadError as e:
        raise ValueError(f"The PDF couldn't be read: {e}") from e
    from .ocr import fill_scanned

    pages, unread = fill_scanned(data, pages, limit=OCR_PAGES)  # scanned pages, read by a vision model or Tesseract
    text = "\n\n".join(f"[Page {i}]\n{p.strip()}" for i, p in enumerate(pages, 1) if p.strip())
    if not text:
        from .ocr import available

        if available():
            raise ValueError("The PDF has no text to read, not even in its scanned pages.")
        raise ValueError("The PDF has no text to read; it may be scanned pages. Load a vision model for chat, or install Tesseract, and attach it again.")
    if unread:
        text += f"\n\n[{unread} scanned pages weren't read.]"
    return text, len(pages)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_text(data: bytes) -> str:
    """Paragraphs and table rows from word/document.xml, in order; tables become tab-separated lines."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            body = ElementTree.fromstring(z.read("word/document.xml")).find(f"{W}body")
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise ValueError("The file isn't a readable Word document.") from e

    def para(p) -> str:
        parts = []
        for node in p.iter():
            if node.tag == f"{W}t":
                parts.append(node.text or "")
            elif node.tag == f"{W}tab":
                parts.append("\t")
            elif node.tag in (f"{W}br", f"{W}cr"):
                parts.append("\n")
        return "".join(parts)

    lines = []
    for block in body if body is not None else []:
        if block.tag == f"{W}p":
            lines.append(para(block))
        elif block.tag == f"{W}tbl":
            for row in block.iter(f"{W}tr"):
                lines.append("\t".join(" ".join(para(p) for p in cell.iter(f"{W}p")).strip() for cell in row.iter(f"{W}tc")))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
            if "\x00" not in text:
                return text
        except UnicodeDecodeError:
            pass
    raise ValueError("The file isn't text Temper can read. Attach a PDF, Word document, CSV or text file.")


def table_text(data: bytes, suffix: str) -> tuple[str, int]:
    """A CSV or TSV as it is (models read CSV well), with its row count."""
    text = _decode(data)
    rows = sum(1 for _ in csv.reader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ","))
    return text.strip(), rows


def read_document(data: bytes, suffix: str) -> dict:
    """The document's text, cut at MAX_CHARS, with what's known about its size."""
    info: dict = {}
    if suffix == ".pdf":
        text, info["pages"] = pdf_text(data)
    elif suffix == ".docx":
        text = docx_text(data)
    elif suffix in (".csv", ".tsv"):
        text, info["rows"] = table_text(data, suffix)
    else:
        text = _decode(data).strip()
    if not text:
        raise ValueError("The file is empty.")
    info["truncated"] = len(text) > MAX_CHARS
    return {"text": text[:MAX_CHARS], **info}


def kind_of(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in UNSUPPORTED:
        raise HTTPException(415, UNSUPPORTED[suffix])
    return "image" if suffix in IMAGES else "audio" if suffix in AUDIO else "document"


# --- Routes ---------------------------------------------------------------------------------------


class AttachRequest(BaseModel):
    path: str | None = None  # a file on this computer (the desktop app knows paths)
    name: str | None = None  # or its name and contents, base64 (a browser only has the bytes)
    data: str | None = None


def add(name: str, data: bytes) -> dict:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{name} is {len(data) / 1024**2:.0f} MB; attachments can be up to {MAX_BYTES // 1024**2} MB.")
    kind = kind_of(name)
    doc = None
    if kind == "document":
        try:
            doc = read_document(data, Path(name).suffix.lower())
        except ValueError as e:
            raise HTTPException(422, f"{name}: {e}") from e
    attachment_id = uuid.uuid4().hex[:12]
    folder = root() / attachment_id
    folder.mkdir(parents=True)
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", Path(name).name) or "file"
    (folder / safe).write_bytes(data)
    info = {"id": attachment_id, "name": Path(name).name, "file": safe, "kind": kind, "size": len(data), "created": time.time(),
            "mime": IMAGES.get(Path(name).suffix.lower()) or mimetypes.guess_type(name)[0] or "application/octet-stream"}
    if doc:
        (folder / "text.txt").write_text(doc.pop("text"), encoding="utf-8")
        info |= doc | {"chars": (folder / "text.txt").stat().st_size}
    (folder / "meta.json").write_text(json.dumps(info, ensure_ascii=False))
    return info


@router.post("")
def attach(req: AttachRequest):
    if req.path:
        path = Path(req.path).expanduser()
        if not path.is_file():
            raise HTTPException(404, f"{path.name} isn't there any more.")
        if path.stat().st_size > MAX_BYTES:
            raise HTTPException(413, f"{path.name} is {path.stat().st_size / 1024**2:.0f} MB; attachments can be up to {MAX_BYTES // 1024**2} MB.")
        return add(path.name, path.read_bytes())
    if req.name and req.data is not None:
        try:
            data = base64.b64decode(req.data, validate=True)
        except ValueError as e:
            raise HTTPException(400, "The file's contents didn't arrive intact.") from e
        return add(req.name, data)
    raise HTTPException(400, "Give a file's path, or its name and contents.")


@router.get("/{attachment_id}")
def get_attachment(attachment_id: str):
    return meta(attachment_id)


@router.get("/{attachment_id}/file")
def get_file(attachment_id: str):
    info = meta(attachment_id)
    return FileResponse(file_path(attachment_id), media_type=info["mime"], filename=info["name"])


@router.delete("/{attachment_id}")
def delete_attachment(attachment_id: str):
    shutil.rmtree(_folder(attachment_id))
    return {"ok": True}


# --- Expanding for llama.cpp ----------------------------------------------------------------------


def expand_part(part: dict) -> dict:
    info = meta(str(part.get("id", "")))
    path = file_path(info["id"])
    if info["kind"] == "image":
        return {"type": "image_url", "image_url": {"url": f"data:{info['mime']};base64,{base64.b64encode(path.read_bytes()).decode()}"}}
    if info["kind"] == "audio":
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(path.read_bytes()).decode(), "format": AUDIO[path.suffix.lower()]}}
    text = (path.parent / "text.txt").read_text(encoding="utf-8")
    cut = "\n[The rest of the file was cut: it's longer than Temper sends.]" if info.get("truncated") else ""
    return {"type": "text", "text": f'<file name="{info["name"]}">\n{text}{cut}\n</file>'}


def expand(messages: list) -> list:
    """Messages with attachment parts replaced by image_url, input_audio or text parts."""
    out = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "attachment" for p in content):
            m = {**m, "content": [expand_part(p) if isinstance(p, dict) and p.get("type") == "attachment" else p for p in content]}
        out.append(m)
    return out


def used(messages: list) -> set[str]:
    """The kinds of attachment in `messages`: image, audio, document."""
    return {meta(str(p.get("id", "")))["kind"] for m in messages if isinstance(m, dict) and isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict) and p.get("type") == "attachment"}
