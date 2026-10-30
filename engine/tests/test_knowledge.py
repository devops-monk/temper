import io
import zipfile

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from temper_engine import knowledge, tools
from temper_engine.app import create_app
from temper_engine.tools import Call

AUTH = {"Authorization": "Bearer secret"}
WORDS = ["llama", "alpaca", "invoice", "budget", "python", "garden", "tomato", "rocket"]


def fake_embeddings(request):
    """A vector per input counting a few words, so passages about the same thing are close."""
    import json

    inputs = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [t.lower().count(w) + 0.01 for w in WORDS]} for i, t in enumerate(inputs)]
    return httpx.Response(200, json={"data": data})


@pytest.fixture
def client(monkeypatch):
    knowledge.embedder.client = httpx.Client(transport=httpx.MockTransport(fake_embeddings))
    with TestClient(create_app(token="secret")) as c:
        yield c
    knowledge.embedder.client = None


def wait(c, name):
    import time

    for _ in range(200):
        b = next(b for b in c.get("/api/knowledge", headers=AUTH).json()["bases"] if b["name"] == name)
        if not b["job"]:
            return b
        time.sleep(0.02)
    raise AssertionError(b)


def office(parts: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, xml in parts.items():
            z.writestr(name, xml)
    return buf.getvalue()


def test_documents_are_read_into_sections():
    a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    deck = office({"ppt/slides/slide2.xml": f"<s {a}><a:p><a:r><a:t>Second</a:t></a:r></a:p></s>", "ppt/slides/slide1.xml": f"<s {a}><a:p><a:r><a:t>First </a:t></a:r><a:r><a:t>slide</a:t></a:r></a:p></s>"})
    assert knowledge.pptx_text(deck) == [("Slide 1", "First slide"), ("Slide 2", "Second")]
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    book = office(
        {
            "xl/workbook.xml": f'<workbook {ns}><sheets><sheet name="Budget"/></sheets></workbook>',
            "xl/sharedStrings.xml": f"<sst {ns}><si><t>Rent</t></si></sst>",
            "xl/worksheets/sheet1.xml": f'<worksheet {ns}><sheetData><row><c t="s"><v>0</v></c><c><v>1200</v></c></row></sheetData></worksheet>',
        }
    )
    assert knowledge.xlsx_text(book) == [("Budget", "Rent\t1200")]
    md = "# Title\nintro\n```\n# not a heading\n```\n## Part\nbody"
    assert [h for h, _ in knowledge._markdown_sections(md)] == ["Title", "Part"]


def test_long_text_is_cut_into_overlapping_passages():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    pieces = knowledge.chunks([("H", text)])
    assert len(pieces) > 3 and all(len(t) <= knowledge.CHUNK for _, t in pieces) and all(h == "H" for h, _ in pieces)
    assert pieces[0][1].startswith("Paragraph 0.") and pieces[1][1][:40] in text


def test_a_base_is_indexed_searched_and_kept_up_to_date(client, tmp_path):
    docs = tmp_path / "docs"
    (docs / "node_modules").mkdir(parents=True)
    (docs / "llamas.md").write_text("# Llamas\nThe llama is a camelid. A llama lives about 20 years.")
    (docs / "money.txt").write_text("The invoice is due. The budget for March is 1200.")
    (docs / "node_modules" / "skip.md").write_text("llama llama llama")
    (docs / "image.png").write_bytes(b"\x89PNG")
    started = client.post("/api/knowledge", headers=AUTH, json={"name": "My Notes", "paths": [str(docs)]}).json()
    assert started["base"] == "my-notes"
    base = wait(client, "my-notes")
    assert (base["files"], base["skipped"]) == (2, 0)
    found = client.get("/api/knowledge/my-notes/search", headers=AUTH, params={"q": "how long does a llama live"}).json()["results"]
    assert found[0]["path"].endswith("llamas.md") and found[0]["heading"] == "Llamas"
    by_word = client.get("/api/knowledge/my-notes/search", headers=AUTH, params={"q": "1200"}).json()["results"]
    assert by_word[0]["path"].endswith("money.txt")

    (docs / "money.txt").unlink()
    (docs / "garden.md").write_text("Tomato plants in the garden need sun.")
    client.post("/api/knowledge/my-notes/index", headers=AUTH)
    base = wait(client, "my-notes")
    assert base["files"] == 2
    assert not any(r["path"].endswith("money.txt") for r in knowledge.search("my-notes", "invoice budget"))

    text = tools.run(Call("search_knowledge", {"query": "tomato garden"}), ["search_knowledge"])["result"]
    assert text.startswith("Passages from the person's documents") and "garden.md" in text.split("\n\n")[1]
    client.put("/api/knowledge/my-notes", headers=AUTH, json={"enabled": False})
    assert "search_knowledge" not in [t.name for t in tools.available()]
    assert client.delete("/api/knowledge/my-notes", headers=AUTH).json()["bases"] == []


def test_bad_bases_are_refused(client, tmp_path):
    assert client.post("/api/knowledge", headers=AUTH, json={"name": "x", "paths": [str(tmp_path / "missing")]}).status_code == 422
    assert client.post("/api/knowledge", headers=AUTH, json={"name": "!!!", "paths": [str(tmp_path)]}).status_code == 422
    assert client.get("/api/knowledge/../etc/search", headers=AUTH, params={"q": "x"}).status_code == 404
    assert client.post("/api/knowledge/embedder", headers=AUTH, json={"id": "nope"}).status_code == 404


def test_vectors_are_unit_length(client):
    v = knowledge.embedder.embed(["llama llama", "budget"], "bge-small-en-1.5")
    assert np.allclose(np.linalg.norm(v, axis=1), 1)


def test_gitignore_rules_are_honoured_in_nested_folders(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "src" / "gen").mkdir(parents=True)
    (tmp_path / ".gitignore").write_text("out/\n*.log\nsecret*.md\n")
    (tmp_path / "src" / ".gitignore").write_text("gen/\n!secret-ok.md\n")
    for f in ["a.md", "run.log", "secret.md", "out/b.md", "src/c.py", "src/gen/d.py", "src/secret-ok.md", "src/secret-no.md"]:
        (tmp_path / f).write_text("x")
    found = sorted(p.relative_to(tmp_path).as_posix() for p in knowledge.files_in([str(tmp_path)]))
    assert found == ["a.md", "src/c.py", "src/secret-ok.md"]


def test_a_synced_base_is_reindexed_when_its_folder_changes(client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("llama")
    client.post("/api/knowledge", headers=AUTH, json={"name": "sync", "paths": [str(docs)]})
    wait(client, "sync")
    assert not knowledge.changed("sync") and knowledge.watcher.check() == []
    (docs / "b.md").write_text("rocket")
    assert knowledge.changed("sync")
    client.put("/api/knowledge/sync", headers=AUTH, json={"sync": False})
    assert knowledge.watcher.check() == []
    client.put("/api/knowledge/sync", headers=AUTH, json={"sync": True})
    assert knowledge.watcher.check() == ["sync"]
    assert wait(client, "sync")["files"] == 2
    (docs / "a.md").unlink()
    assert knowledge.changed("sync")
