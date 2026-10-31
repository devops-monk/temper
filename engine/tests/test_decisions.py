import json

import pytest
from fastapi.testclient import TestClient

from temper_engine import decisions, runtime
from temper_engine.app import create_app
from temper_engine.mcp_server import handle

H = {"Authorization": "Bearer t"}


@pytest.fixture
def asked(monkeypatch):
    """The model, played by a list of replies; records what each request sent."""
    calls: list[dict] = []
    replies: list = []

    def complete(messages, max_tokens=1024, temperature=None, model=None, response_format=None):
        calls.append({"messages": messages, "temperature": temperature, "model": model, "format": response_format})
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(runtime, "complete", complete)
    return calls, replies


@pytest.fixture
def client():
    with TestClient(create_app(token="t")) as c:
        yield c


def test_yes_no_is_held_to_a_schema_and_answers_a_boolean(client, asked):
    calls, replies = asked
    replies.append('{"reason": "It asks for money back.", "answer": "yes"}')
    out = client.post("/v1/decide", headers=H, json={"question": "Is this a refund request?", "text": "I want my money back"}).json()
    assert out["answer"] is True and out["reason"] == "It asks for money back." and out["kind"] == "yes_no"
    sent = calls[0]
    assert sent["temperature"] == 0 and sent["format"]["json_schema"]["schema"]["properties"]["answer"] == {"type": "string", "enum": ["yes", "no"]}
    assert "<text>\nI want my money back\n</text>" in sent["messages"][1]["content"]


def test_choices_scores_and_labels_are_read_even_from_plain_replies(client, asked):
    calls, replies = asked
    replies += ['{"reason": "r", "answer": "high"}', "Score: 7 out of 10", '{"answer": ["Billing", "bug"]}']
    choice = client.post("/v1/decide", headers=H, json={"question": "Priority?", "kind": "choice", "choices": ["Low", "High"], "text": "Site down"}).json()
    assert choice["answer"] == "High"  # matched without regard to case, given back as written
    assert client.post("/v1/decide", headers=H, json={"question": "How urgent?", "kind": "score", "text": "x"}).json()["answer"] == 7
    labels = client.post("/v1/decide", headers=H, json={"question": "Topics?", "kind": "labels", "choices": ["billing", "bug", "praise"], "text": "x"}).json()
    assert labels["answer"] == ["billing", "bug"]
    assert calls[2]["format"]["json_schema"]["schema"]["properties"]["answer"]["items"]["enum"] == ["billing", "bug", "praise"]


def test_an_unreadable_answer_is_an_error_not_a_guess(client, asked):
    _, replies = asked
    replies += ['{"answer": "maybe"}', '{"answer": "Medium"}', '{"answer": 11}']
    assert client.post("/v1/decide", headers=H, json={"question": "q", "text": "x"}).status_code == 422
    assert client.post("/v1/decide", headers=H, json={"question": "q", "kind": "choice", "choices": ["Low", "High"]}).status_code == 422
    assert client.post("/v1/decide", headers=H, json={"question": "q", "kind": "score", "scale": [1, 10]}).status_code == 422
    assert client.post("/v1/decide", headers=H, json={"question": "q", "kind": "choice", "choices": ["only"]}).status_code == 422
    assert client.post("/v1/decide", headers=H, json={"question": "q", "kind": "score", "scale": [5, 1]}).status_code == 422


def test_a_server_without_structured_output_is_asked_plainly(client, asked):
    calls, replies = asked
    replies += [ValueError("The model's server answered 400: response_format not supported"), "No."]
    assert client.post("/v1/decide", headers=H, json={"question": "q", "text": "x"}).json()["answer"] is False
    assert calls[0]["format"] is not None and calls[1]["format"] is None


def test_saved_decisions_take_only_the_text_and_run_over_many(client, asked):
    calls, replies = asked
    spam = {"question": "Is this spam?", "instructions": "Newsletters the person signed up for aren't spam.", "model": "small"}
    assert client.put("/api/decisions/is-spam", headers=H, json=spam).json()["decisions"][0]["name"] == "is-spam"
    assert client.put("/api/decisions/Bad Name", headers=H, json=spam).status_code == 400
    replies += ['{"answer": "yes"}', '{"answer": "no"}']
    out = client.post("/v1/decisions/is-spam", headers=H, json={"texts": ["WIN A PRIZE", "Lunch at 1?"]}).json()
    assert out["decision"] == "is-spam" and [r["answer"] for r in out["results"]] == [True, False]
    assert calls[0]["model"] == "small" and "Newsletters" in calls[0]["messages"][0]["content"]
    assert client.post("/v1/decisions/nope", headers=H, json={"text": "x"}).status_code == 404
    assert client.delete("/api/decisions/is-spam", headers=H).json()["decisions"] == []


def test_agents_decide_through_mcp(asked):
    _, replies = asked
    replies.append('{"reason": "Clear bug report.", "answer": "bug"}')
    args = {"question": "What is it?", "kind": "choice", "choices": ["bug", "feature"], "text": "Crash on save"}
    out = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "decide", "arguments": args}})["result"]
    assert out["structuredContent"]["answer"] == "bug" and json.loads(out["content"][0]["text"])["reason"] == "Clear bug report."
    bad = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "decide", "arguments": {"question": "q", "kind": "choice"}}})["result"]
    assert bad["isError"]


def test_the_reason_comes_before_the_answer_in_the_schema():
    d = decisions.Decision(question="q", kind="score", scale=(0, 5))
    s = decisions.schema(d)
    assert list(s["properties"]) == ["reason", "answer"] and s["properties"]["answer"] == {"type": "integer", "minimum": 0, "maximum": 5}
