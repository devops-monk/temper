import re
import time

import pytest
from fastapi.testclient import TestClient

from temper_engine import research, runtime, search, web
from temper_engine.app import create_app

AUTH = {"Authorization": "Bearer secret"}

PAGES = {
    "https://a.org": "Page: Alpha\nAddress: https://a.org\n(The text below…)\n\nLlamas live 20 years.",
    "https://b.org": "Page: Beta\nAddress: https://b.org\n(The text below…)\n\nNothing about llamas.",
}


def fake_model(messages, max_tokens=1024, temperature=None, model=None):
    prompt = messages[-1]["content"]
    if "Write 3 different web searches" in prompt or "Write 4 different web searches" in prompt:
        return 'Sure: ["llama lifespan", "llama age"]'
    if "<page>" in prompt:
        return "- Llamas live about 20 years." if "20 years" in prompt else "NONE"
    if "still need" in prompt:
        return "[]"
    if "Write a clear report" in prompt:
        assert "Source [1] Alpha" in prompt and "Beta" not in prompt
        return "Llamas live about 20 years [1]."
    raise AssertionError(prompt)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(runtime, "complete", fake_model)
    monkeypatch.setattr(runtime.manager, "state", "ready")
    results = {"llama lifespan": [{"title": "A", "url": "https://a.org", "snippet": "s"}], "llama age": [{"title": "A", "url": "https://a.org", "snippet": "s"}, {"title": "B", "url": "https://b.org", "snippet": ""}]}
    monkeypatch.setattr(search, "search", lambda q, count=6, client=None: ("DuckDuckGo", results[q]))
    monkeypatch.setattr(web, "read", lambda url, part=1, client=None: PAGES[url])
    with TestClient(create_app(token="secret")) as c:
        yield c


def wait(c, rid):
    deadline = time.monotonic() + 20  # generous: a busy computer can take a while
    while time.monotonic() < deadline:
        run = c.get(f"/api/research/{rid}", headers=AUTH).json()
        if run["status"] not in ("running", "writing"):
            return run
        time.sleep(0.05)
    raise AssertionError(run)


def test_a_run_searches_reads_and_writes_a_cited_report(client):
    rid = client.post("/api/research", headers=AUTH, json={"question": "How long do llamas live?", "depth": "quick"}).json()["id"]
    run = wait(client, rid)
    assert run["status"] == "done" and run["queries"] == ["llama lifespan", "llama age"]
    assert [(s["n"], s["status"], s["title"]) for s in run["sources"]] == [(1, "used", "Alpha"), (2, "irrelevant", "Beta")]
    assert run["report"] == "Llamas live about 20 years [1]."
    md = client.get(f"/api/research/{rid}/export", headers=AUTH).json()
    assert md["filename"] == "how-long-do-llamas-live.md" and "## Sources\n\n1. [Alpha](https://a.org)" in md["text"] and "b.org" not in md["text"]
    assert client.get("/api/research", headers=AUTH).json()["runs"][0] | {"created": 0, "updated": 0} == {"id": rid, "question": "How long do llamas live?", "depth": "quick", "status": "done", "created": 0, "updated": 0, "sources": 1}


def test_review_leaves_sources_out_of_the_rewritten_report(client, monkeypatch):
    rid = client.post("/api/research", headers=AUTH, json={"question": "How long do llamas live?", "depth": "quick"}).json()["id"]
    wait(client, rid)
    r = client.post(f"/api/research/{rid}/write", headers=AUTH, json={"excluded": [1]})
    assert r.status_code == 200, r.text
    run = wait(client, rid)
    assert run["sources"][0]["included"] is False and run["report"].startswith("None of the sources"), (run["status"], run.get("error"), run["report"])
    assert client.delete(f"/api/research/{rid}", headers=AUTH).json() == {"runs": []}


def test_no_model_and_bad_input_are_refused(client, monkeypatch):
    assert client.post("/api/research", headers=AUTH, json={"question": "q?!", "depth": "forever"}).status_code == 422
    monkeypatch.setattr(runtime.manager, "state", "idle")
    assert client.post("/api/research", headers=AUTH, json={"question": "How long do llamas live?"}).status_code == 409
    assert client.get("/api/research/../../x", headers=AUTH).status_code == 404


def test_a_run_left_going_by_a_restart_shows_as_stopped():
    run = {"id": "0123456789ab", "question": "q", "depth": "quick", "created": 1, "status": "running", "job": "gone", "queries": [], "steps": [], "sources": [], "report": None}
    research.save(run)
    assert research.load("0123456789ab")["status"] == "stopped"


def test_model_lists_are_read_from_json_or_lines():
    assert research._list('Here: ["a", "b", 3]', 5) == ["a", "b"]
    assert research._list("1. first\n- second\n", 5) == ["first", "second"]
    assert re.fullmatch(r"\W*none\W*", "NONE.", re.I)
