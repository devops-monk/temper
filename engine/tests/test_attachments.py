import base64
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from temper_engine import attachments
from temper_engine.app import create_app
from temper_engine.runtime import find_projector, manager

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def call(method, path, status=200, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == status, r.text
    return r.json()


def attach(name, data, status=200):
    return call("POST", "/api/attachments", status, json={"name": name, "data": base64.b64encode(data).decode()})


def pdf(*pages: str) -> bytes:
    """A minimal PDF with one line of text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R /Resources << /Font << /F1 {{font}} 0 R >> >> >>")
        kids.append(len(objects))
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font = len(objects)
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body.replace('{font}', str(font))}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010} 00000 n \n".encode() for o in offsets)
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def docx(xml_body: str) -> bytes:
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f"<w:document {ns}><w:body>{xml_body}</w:body></w:document>")
    return buf.getvalue()


def test_documents_are_read_into_text():
    a = attach("report.pdf", pdf("Revenue rose 12 percent", "Costs fell"))
    assert a["kind"] == "document" and a["pages"] == 2 and not a["truncated"]
    text = attachments.expand_part({"type": "attachment", "id": a["id"]})["text"]
    assert text.startswith('<file name="report.pdf">\n[Page 1]\nRevenue rose 12 percent') and "[Page 2]\nCosts fell" in text

    d = attach("notes.docx", docx("<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:tab/><w:t>world</w:t></w:r></w:p>"
                                  "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"))
    assert "Hello\tworld\na\tb" in attachments.expand_part({"type": "attachment", "id": d["id"]})["text"]

    c = attach("sales.csv", "region,total\nnorth,3\nsouth,4\n".encode())
    assert c["rows"] == 3
    assert attach("main.py", b"print('hi')\n")["kind"] == "document"


def test_unreadable_files_say_what_to_do():
    assert "Save it as PNG or JPEG" in attach("photo.webp", b"RIFF....WEBP", 415)["detail"]
    assert "Save the sheet as CSV" in attach("book.xlsx", b"PK", 415)["detail"]
    assert "isn't text" in attach("blob.bin", bytes(range(256)) * 4, 422)["detail"]
    assert "scanned pages" in attach("scan.pdf", pdf(""), 422)["detail"]


def test_images_and_audio_become_llama_cpp_parts(tmp_path):
    image = attach("cat.png", PNG)
    assert image["kind"] == "image" and image["mime"] == "image/png"
    part = attachments.expand_part({"type": "attachment", "id": image["id"]})
    assert part == {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}
    (tmp_path / "hello.wav").write_bytes(b"RIFF0000WAVE")
    sound = call("POST", "/api/attachments", json={"path": str(tmp_path / "hello.wav")})
    assert attachments.expand_part({"type": "attachment", "id": sound["id"]})["input_audio"]["format"] == "wav"
    r = client.get(f"/api/attachments/{image['id']}/file", headers=AUTH)
    assert r.content == PNG and r.headers["content-type"] == "image/png"


def test_a_model_without_vision_refuses_images(monkeypatch):
    image = attach("cat.png", PNG)
    monkeypatch.setattr(manager, "state", "ready")
    monkeypatch.setattr(manager, "model", "text-only.gguf")
    monkeypatch.setattr(manager, "modalities", {"vision": False, "audio": False})
    monkeypatch.setattr(manager, "projector", None)
    messages = [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "attachment", "id": image["id"]}]}]
    detail = call("POST", "/v1/chat/completions", 400, json={"messages": messages})["detail"]
    assert detail.startswith("text-only.gguf can't see images. Its projector (an mmproj .gguf) wasn't found beside it.")


def test_chats_keep_attachments_and_the_library_lists_them():
    image = attach("diagram.png", PNG)
    files = [{"id": image["id"], "name": "diagram.png", "kind": "image"}]
    chat = call("POST", "/api/chats", json={"messages": [{"role": "user", "content": "Explain", "attachments": files}]})
    assert call("GET", f"/api/chats/{chat['id']}")["messages"][0]["attachments"] == files
    assert "Attached: diagram.png" in call("GET", f"/api/chats/{chat['id']}/export?format=md")["text"]
    item = next(i for i in call("GET", "/api/files")["items"] if i["kind"] == "attachment" and image["id"] in i["path"])
    assert item["name"] == "diagram" and item["attached"] == "image"
    call("DELETE", f"/api/files?id={item['id']}")
    call("GET", f"/api/attachments/{image['id']}", 404)
    call("DELETE", f"/api/chats/{chat['id']}")


@pytest.mark.parametrize(
    "names, expected",
    [
        (["model-Q4_K_M.gguf"], None),
        (["model-Q4_K_M.gguf", "mmproj-model-Q8_0.gguf", "mmproj-model-F16.gguf"], "mmproj-model-F16.gguf"),
        (["model-Q4_K_M.gguf", "mmproj-Q8_0.gguf"], "mmproj-Q8_0.gguf"),
    ],
)
def test_the_projector_beside_a_model_is_found(tmp_path, names, expected):
    for name in names:
        (tmp_path / name).write_bytes(b"GGUF")
    found = find_projector(str(tmp_path / names[0]))
    assert (found and found.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]) == expected


def test_downloading_a_quant_brings_its_projector(monkeypatch):
    from temper_engine import downloads, hub

    listing = [{"name": n, "size": 1} for n in ("model-Q4_K_M.gguf", "model-Q8_0.gguf", "mmproj-model-Q8_0.gguf", "mmproj-model-f16.gguf", "README.md")]
    monkeypatch.setattr(hub, "list_files", lambda repo: listing)
    got = {}
    monkeypatch.setattr(downloads, "download", lambda job, repo, files: got.setdefault("files", [n for n, _ in files]))
    hub._download(None, hub.DownloadRequest(repo_id="org/vl-GGUF", patterns=["*Q4_K_M*.gguf"]))
    assert got["files"] == ["model-Q4_K_M.gguf", "mmproj-model-f16.gguf"]
    got.clear()
    hub._download(None, hub.DownloadRequest(repo_id="org/vl-GGUF", patterns=["README.md"]))
    assert got["files"] == ["README.md"]
