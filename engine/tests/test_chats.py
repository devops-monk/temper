import json
import sqlite3
import time

from fastapi.testclient import TestClient

from temper_engine import chat_import
from temper_engine.app import create_app
from temper_engine.jobs import registry

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def call(method, path, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == 200, r.text
    return r.json() if "json" in r.headers.get("content-type", "") else r.text


def wait(job_id):
    job = registry.get(job_id)
    for _ in range(200):
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("job still running")


def test_save_search_rename_pin_and_export():
    chat = call("POST", "/api/chats", json={"model": "qwen3-4b.gguf", "settings": {"system": "Be brief."}})
    assert chat["title"] == "New chat" and chat["messages"] == []
    long_question = "How do I bake sourdough bread with a starter that is only three days old and very runny?"
    call("PUT", f"/api/chats/{chat['id']}/messages", json={"messages": [{"role": "user", "content": long_question}, {"role": "assistant", "content": "Feed the levain twice daily."}]})
    saved = call("GET", f"/api/chats/{chat['id']}")
    assert saved["title"].startswith("How do I bake sourdough") and saved["title"].endswith("…") and len(saved["title"]) <= 61
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"] and saved["settings"] == {"system": "Be brief."}
    other = call("POST", "/api/chats", json={"messages": [{"role": "user", "content": "Plan a trip to Kyoto"}]})

    # Search finds message text (with a prefix) and titles, and survives FTS syntax in the query.
    found = call("GET", "/api/chats", params={"q": "levai"})["chats"]
    assert [c["id"] for c in found] == [chat["id"]] and "[[levain]]" in found[0]["snippet"]
    assert [c["id"] for c in call("GET", "/api/chats", params={"q": "kyoto"})["chats"]] == [other["id"]]
    assert call("GET", "/api/chats", params={"q": 'AND "( NEAR'})["chats"] == []

    call("PATCH", f"/api/chats/{chat['id']}", json={"title": "Sourdough", "pinned": True})
    listed = call("GET", "/api/chats")["chats"]
    assert listed[0]["id"] == chat["id"] and listed[0]["pinned"] and listed[0]["messages"] == 2
    # Regenerating saves a shorter thread; the old reply leaves the index too.
    call("PUT", f"/api/chats/{chat['id']}/messages", json={"messages": [{"role": "user", "content": long_question}]})
    assert call("GET", "/api/chats", params={"q": "levain"})["chats"] == []
    assert call("GET", f"/api/chats/{chat['id']}")["title"] == "Sourdough"

    md = call("GET", f"/api/chats/{chat['id']}/export", params={"format": "md"})
    assert md["filename"] == "Sourdough.md" and md["text"].startswith("# Sourdough") and "## You" in md["text"]
    data = json.loads(call("GET", f"/api/chats/{chat['id']}/export", params={"format": "json"})["text"])
    assert data["messages"][0]["content"] == long_question
    call("DELETE", f"/api/chats/{chat['id']}")
    assert client.get(f"/api/chats/{chat['id']}", headers=AUTH).status_code == 404


def test_folders():
    chat = call("POST", "/api/chats", json={"messages": [{"role": "user", "content": "hi"}]})
    assert client.patch(f"/api/chats/{chat['id']}", json={"folder": "Work"}, headers=AUTH).status_code == 404
    call("POST", "/api/chats/folders/add", json={"name": " Work "})
    call("PATCH", f"/api/chats/{chat['id']}", json={"folder": "Work"})
    call("POST", "/api/chats/folders/rename", json={"name": "Work", "new_name": "Job"})
    assert [c["id"] for c in call("GET", "/api/chats", params={"folder": "Job"})["chats"]] == [chat["id"]]
    call("POST", "/api/chats/folders/remove", json={"name": "Job"})
    assert call("GET", f"/api/chats/{chat['id']}")["folder"] is None


def jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_import_claude_code(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    session = tmp_path / "claude" / "projects" / "-Users-me-proj" / "abc-123.jsonl"
    base = {"cwd": "/Users/me/proj", "timestamp": "2026-10-01T10:00:00Z", "isSidechain": False}
    lines = [
        {"type": "ai-title", "aiTitle": "Fix the login bug"},
        {**base, "type": "user", "message": {"role": "user", "content": "Why does login fail?<system-reminder>secret context</system-reminder>"}},
        {**base, "type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5-5", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Let me look."}]}},
        {**base, "type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Read", "input": {}}]}},
        {**base, "type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "file text"}]}},
        {**base, "type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "The token expires."}]}},
        {**base, "type": "user", "isMeta": True, "message": {"role": "user", "content": "[Image: original 10x10]"}},
        {**base, "type": "user", "message": {"role": "user", "content": "<task-notification>done</task-notification>"}},
        {**base, "type": "user", "isSidechain": True, "message": {"role": "user", "content": "subagent prompt"}},
        {**base, "type": "user", "timestamp": "2026-10-01T10:05:00Z", "message": {"role": "user", "content": "Thanks!"}},
    ]
    jsonl(session, lines)
    jsonl(tmp_path / "claude" / "projects" / "-Users-me-proj" / "only-commands.jsonl", [{**base, "type": "user", "message": {"role": "user", "content": "<command-name>/clear</command-name>"}}])
    source = next(s for s in call("GET", "/api/chats/import/sources") if s["source"] == "claude-code")
    assert source["found"] == 2 and source["imported"] == 0
    job = wait(call("POST", "/api/chats/import", json={"source": "claude-code"})["id"])
    assert job.status == "done" and job.result["added"] == 1 and job.result["skipped"] == 1
    chat = call("GET", "/api/chats", params={"source": "Claude Code"})["chats"][0]
    assert chat["title"] == "Fix the login bug" and chat["project"] == "proj" and chat["model"] == "claude-opus-5-5"
    messages = call("GET", f"/api/chats/{chat['id']}")["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [("user", "Why does login fail?"), ("assistant", "Let me look.\n\nThe token expires."), ("user", "Thanks!")]

    # Importing again: unchanged sessions are skipped; a session that grew is refreshed, keeping the person's title.
    call("PATCH", f"/api/chats/{chat['id']}", json={"title": "Login", "pinned": True})
    assert wait(call("POST", "/api/chats/import", json={"source": "claude-code"})["id"]).result["unchanged"] == 1
    jsonl(session, lines + [{**base, "type": "assistant", "timestamp": "2026-10-01T10:06:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "You're welcome."}]}}])
    assert wait(call("POST", "/api/chats/import", json={"source": "claude-code"})["id"]).result["updated"] == 1
    again = call("GET", f"/api/chats/{chat['id']}")
    assert again["title"] == "Login" and again["pinned"] and len(again["messages"]) == 4


def test_import_codex_both_formats(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    jsonl(tmp_path / "codex" / "sessions" / "2026" / "10" / "01" / "rollout-new.jsonl", [
        {"timestamp": "2026-10-01T09:00:00Z", "type": "session_meta", "payload": {"id": "s-1", "cwd": "/w/api", "instructions": None}},
        {"timestamp": "2026-10-01T09:00:00Z", "type": "turn_context", "payload": {"model": "gpt-5.5-codex"}},
        {"timestamp": "2026-10-01T09:00:01Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]}},
        {"timestamp": "2026-10-01T09:00:02Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Add a health endpoint"}]}},
        {"timestamp": "2026-10-01T09:00:03Z", "type": "response_item", "payload": {"type": "function_call", "name": "shell"}},
        {"timestamp": "2026-10-01T09:00:04Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Added /health."}]}},
    ])
    jsonl(tmp_path / "codex" / "sessions" / "2025" / "05" / "01" / "rollout-old.jsonl", [
        {"id": "s-0", "timestamp": "2025-05-01T09:00:00Z", "instructions": ""},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Explain this repo"}]},
        {"record_type": "state"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It's a CLI."}]},
    ])
    job = wait(call("POST", "/api/chats/import", json={"source": "codex"})["id"])
    assert job.result["added"] == 2
    chats = {c["title"]: c for c in call("GET", "/api/chats", params={"source": "Codex"})["chats"]}
    assert set(chats) == {"Add a health endpoint", "Explain this repo"}
    assert chats["Add a health endpoint"]["project"] == "api" and chats["Add a health endpoint"]["model"] == "gpt-5.5-codex"
    assert [m["content"] for m in call("GET", f"/api/chats/{chats['Add a health endpoint']['id']}")["messages"]] == ["Add a health endpoint", "Added /health."]


def test_import_cursor(tmp_path, monkeypatch):
    path = tmp_path / "state.vscdb"
    con = sqlite3.connect(path)
    con.execute("create table cursorDiskKV (key text unique on conflict replace, value blob)")
    rows = {
        "composerData:c1": {"composerId": "c1", "name": "Refactor service", "createdAt": 1747865852198, "lastUpdatedAt": 1747865926316,
                            "fullConversationHeadersOnly": [{"bubbleId": "b1", "type": 1}, {"bubbleId": "b2", "type": 2}, {"bubbleId": "b3", "type": 2}]},
        "bubbleId:c1:b1": {"type": 1, "text": "Split this class"},
        "bubbleId:c1:b2": {"type": 2, "text": ""},
        "bubbleId:c1:b3": {"type": 2, "text": "Here's the split."},
        "composerData:c2": {"composerId": "c2", "createdAt": 1747865852198, "conversation": [{"type": 1, "text": "Inline question"}, {"type": 2, "text": "Inline answer"}]},
        "composerData:empty": {"composerId": "empty", "fullConversationHeadersOnly": []},
    }
    con.executemany("insert into cursorDiskKV values (?, ?)", [(k, json.dumps(v)) for k, v in rows.items()])
    con.commit()
    con.close()
    monkeypatch.setattr(chat_import, "cursor_db", lambda: path)
    assert next(s for s in call("GET", "/api/chats/import/sources") if s["source"] == "cursor")["found"] == 2
    assert wait(call("POST", "/api/chats/import", json={"source": "cursor"})["id"]).result["added"] == 2
    chats = {c["title"]: c for c in call("GET", "/api/chats", params={"source": "Cursor"})["chats"]}
    assert set(chats) == {"Refactor service", "Inline question"}
    assert [m["content"] for m in call("GET", f"/api/chats/{chats['Refactor service']['id']}")["messages"]] == ["Split this class", "Here's the split."]
