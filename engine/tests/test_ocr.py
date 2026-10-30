import zlib

import pytest

from temper_engine import attachments, knowledge, ocr, runtime


def pdf(objects: list[bytes]) -> bytes:
    """A PDF from its numbered objects (1 is the catalog)."""
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def stream(head: str, data: bytes) -> bytes:
    return f"<< {head} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def text_pdf(line: str) -> bytes:
    content = f"BT /F1 28 Tf 40 300 Td ({line}) Tj ET".encode()
    return pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 500 400] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            stream("", content),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def scanned_pdf(line: str) -> bytes:
    """A PDF whose one page is only a picture of `line`: no text layer, like a scan."""
    import pypdfium2 as pdfium

    page = pdfium.PdfDocument(text_pdf(line))[0]
    pixels = page.render(scale=2, rev_byteorder=True).to_numpy()[:, :, :3].copy()
    h, w = pixels.shape[:2]
    image = stream(f"/Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode", zlib.compress(pixels.tobytes()))
    return pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 500 400] /Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>",
            stream("", b"q 500 0 0 400 0 0 cm /Im0 Do Q"),
            image,
        ]
    )


def test_pages_render_to_png():
    image = ocr.render(text_pdf("Hello"), 0)
    assert image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) > 1000


def test_scanned_pages_are_read_by_a_vision_model_first(monkeypatch):
    monkeypatch.setattr(runtime.manager, "state", "ready")
    monkeypatch.setattr(runtime.manager, "modalities", {"vision": True})
    seen = []
    monkeypatch.setattr(runtime, "complete", lambda messages, *a, **k: seen.append(messages) or "Invoice 4821")
    pages, unread = ocr.fill_scanned(scanned_pdf("Invoice 4821"), ["", "A page that has plenty of its own text already."])
    assert pages == ["Invoice 4821", "A page that has plenty of its own text already."] and unread == 0
    assert seen[0][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_without_a_reader_scanned_pages_are_counted(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: None)
    assert ocr.fill_scanned(b"", ["", "x" * 40]) == (["", "x" * 40], 1)
    with pytest.raises(ValueError, match="scanned"):
        attachments.pdf_text(scanned_pdf("Nothing"))


@pytest.mark.skipif(not ocr.tesseract(), reason="needs Tesseract")
def test_tesseract_reads_a_scan_in_attachments_and_knowledge(tmp_path):
    data = scanned_pdf("Invoice 4821 total 1200")
    text, pages = attachments.pdf_text(data)
    assert pages == 1 and "4821" in text
    (tmp_path / "scan.pdf").write_bytes(data)
    assert "4821" in knowledge.read_document(tmp_path / "scan.pdf")[0][1]
