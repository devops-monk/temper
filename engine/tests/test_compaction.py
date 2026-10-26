import pytest
from fastapi.testclient import TestClient

from temper_engine import compaction
from temper_engine.app import create_app
from temper_engine.runtime import manager

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def call(method, path, status=200, **kw):
    r = client.request(method, path, headers=AUTH, **kw)
    assert r.status_code == status, r.text
    return r.json()


@pytest.fixture
def model(monkeypatch):
    """A 200-token model where every word is a token; summaries say how many messages they folded in."""
    asked = []

    def post(server, path, body, timeout=60):
        if path == "/apply-template":
            return {"prompt": " ".join(f"<{m['role']}> {m['content']}" for m in body["messages"])}
        if path == "/tokenize":
            return {"tokens": body["content"].split()}
        asked.append(body["messages"][-1]["content"])
        return {"choices": [{"message": {"content": f"summary of {body['messages'][-1]['content'].count('User:')} user turns"}}]}

    monkeypatch.setattr(compaction, "_post", post)
    monkeypatch.setattr(manager, "state", "ready")
    monkeypatch.setattr(manager, "context", 200)
    yield asked


def turns(n, words=20):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": " ".join([f"w{i}"] * words)} for i in range(n)]


def test_a_short_thread_is_sent_as_it_is(model):
    out = call("POST", "/api/chat/prepare", json={"messages": turns(3), "system": "Be brief.", "max_tokens": 40})
    assert not out["compacted"] and out["summary"] is None and model == []
    assert out["messages"][0] == {"role": "system", "content": "Be brief."} and len(out["messages"]) == 4
    assert out["context"] == 200 and out["reserve"] == 40 and out["prompt"] == 3 * 21 + 3


def test_a_long_thread_has_its_oldest_turns_summarised(model):
    thread = turns(10)
    out = call("POST", "/api/chat/prepare", json={"messages": thread, "system": "Be brief.", "max_tokens": 40})
    assert out["compacted"]
    upto = out["summary"]["upto"]
    assert thread[upto]["role"] == "user" and 0 < upto < 10
    assert out["messages"][1:] == thread[upto:]
    assert out["messages"][0]["content"] == f"Be brief.\n\nSummary of the earlier conversation:\nsummary of {(upto + 1) // 2} user turns"
    assert out["prompt"] + out["reserve"] <= 200
    # The next turn reuses the summary without asking the model again.
    model.clear()
    thread += turns(1)
    again = call("POST", "/api/chat/prepare", json={"messages": thread, "summary": out["summary"], "max_tokens": 40})
    assert not again["compacted"] and again["summary"] == out["summary"] and model == []


def test_turned_off_or_too_big_a_message_says_why(model):
    detail = call("POST", "/api/chat/prepare", 413, json={"messages": turns(10), "compact": False, "max_tokens": 40})["detail"]
    assert detail.startswith("This conversation needs about") and "Turn on auto-compaction" in detail
    detail = call("POST", "/api/chat/prepare", 413, json={"messages": turns(1, words=300), "max_tokens": 40})["detail"]
    assert detail.startswith("This message needs about") and "Attach less" in detail


def test_turns_longer_than_the_context_are_summarised_a_chunk_at_a_time(model, monkeypatch):
    summary = compaction.summarise(manager, None, turns(6, words=300))  # six turns of about 900 characters: 4 a chunk
    assert len(model) == 2 and "Summary so far:" in model[1] and "Summary so far:" not in model[0]
    assert summary.startswith("summary of")


def test_chats_keep_the_summary():
    chat = call("POST", "/api/chats", json={"messages": turns(2)})
    summary = {"upto": 2, "text": "They talked about w."}
    call("PUT", f"/api/chats/{chat['id']}/messages", json={"messages": turns(4), "compaction": summary})
    assert call("GET", f"/api/chats/{chat['id']}")["compaction"] == summary
    call("PUT", f"/api/chats/{chat['id']}/messages", json={"messages": turns(1)})
    assert call("GET", f"/api/chats/{chat['id']}")["compaction"] is None


def test_a_check_only_counts(model):
    out = call("POST", "/api/chat/prepare", json={"messages": turns(10), "max_tokens": 40, "check": True})
    assert out["fits"] is False and model == []


def test_failed_replies_are_not_sent(model):
    thread = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "The model ended its reply without writing anything.", "error": True}, {"role": "user", "content": "hello?"}]
    out = call("POST", "/api/chat/prepare", json={"messages": thread, "max_tokens": 40})
    assert out["messages"] == [{"role": "user", "content": "hi"}, {"role": "user", "content": "hello?"}]
