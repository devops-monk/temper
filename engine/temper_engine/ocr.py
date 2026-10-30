"""Reading scanned pages: PDF pages with no text layer, turned back into text.

A page is rendered to an image (PDFium, through pypdfium2) and transcribed:

1. by the model loaded for chat, when it can see images (a vision model with its projector), since
   it reads layouts, tables and handwriting far better; otherwise
2. by Tesseract, when it's installed (brew install tesseract, apt install tesseract-ocr, or the
   Windows installer).

With neither, scanned pages stay unread and the caller says so.
"""

from __future__ import annotations

import base64
import shutil
import struct
import subprocess
import zlib

MIN_TEXT = 25  # characters of text below which a page counts as scanned
SCALE = 2.0  # rendering scale: 144 dpi, enough for small print

PROMPT = (
    "Transcribe all the text in this page image exactly as written, in reading order. Keep headings, lists and paragraphs; "
    "write tables as Markdown tables. Write only the page's text, with no comments of your own."
)


def png(pixels, mode: str = "RGB") -> bytes:
    """A PNG of an 8-bit image array (height × width × channels)."""
    h, w = pixels.shape[:2]
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    color = {1: 0, 3: 2, 4: 6}[channels]
    raw = b"".join(b"\0" + pixels[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def render(pdf: bytes | str, index: int) -> bytes:
    """Page `index` (from 0) of a PDF, as a PNG."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf)
    try:
        bitmap = doc[index].render(scale=SCALE, rev_byteorder=True)  # RGB(A) rather than BGR(A)
        pixels = bitmap.to_numpy()
        return png(pixels[:, :, :3] if pixels.ndim == 3 and pixels.shape[2] == 4 else pixels)
    finally:
        doc.close()


def _vision() -> bool:
    from .runtime import manager

    return manager.state == "ready" and bool((manager.modalities or {}).get("vision"))


def tesseract() -> str | None:
    return shutil.which("tesseract")


def available() -> str | None:
    """What would read a scanned page now: "vision" (the chat model), "tesseract", or None."""
    return "vision" if _vision() else "tesseract" if tesseract() else None


def transcribe(image: bytes) -> str:
    """The text in a page image, by the chat model if it can see, else Tesseract."""
    if _vision():
        from .runtime import complete

        url = "data:image/png;base64," + base64.b64encode(image).decode()
        return complete([{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}], 3000, temperature=0).strip()
    exe = tesseract()
    if not exe:
        raise ValueError("Reading scanned pages needs a vision model loaded for chat, or Tesseract installed.")
    done = subprocess.run([exe, "stdin", "stdout"], input=image, capture_output=True, timeout=180)
    if done.returncode:
        raise ValueError(f"Tesseract couldn't read the page: {done.stderr.decode(errors='replace').strip()[:200]}")
    return done.stdout.decode("utf-8", errors="replace").strip()


def fill_scanned(pdf: bytes | str, pages: list[str], limit: int | None = None) -> tuple[list[str], int]:
    """`pages` (each page's text layer) with scanned pages transcribed; and how many were scanned but left unread."""
    if not available():
        return pages, sum(len(p.strip()) < MIN_TEXT for p in pages)
    out, left, done = list(pages), 0, 0
    for i, text in enumerate(pages):
        if len(text.strip()) >= MIN_TEXT:
            continue
        if limit is not None and done >= limit:
            left += 1
            continue
        try:
            out[i] = transcribe(render(pdf, i))
            done += 1
        except (ValueError, subprocess.TimeoutExpired, OSError):
            left += 1
    return out, left
