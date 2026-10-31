import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import api_keys
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
CHUNKS = [
    {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
    {"choices": [{"delta": {"content": "Two: "}}]},
    {"choices": [{"delta": {"content": "apple, pear"}}]},
    {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 6}},
]


@pytest.fixture
def client(monkeypatch):
    """The engine, with a model server that streams CHUNKS (or answers whole when not streaming)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if not json.loads(request.content).get("stream"):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "apple"}}], "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in CHUNKS) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr("temper_engine.runtime.route", lambda body: (SimpleNamespace(model="qwen"), "http://model/v1/chat/completions", body))
    monkeypatch.setattr(api_keys, "_content_until", 0.0)
    with TestClient(create_app(token="t")) as c:
        yield c


def ask(c, stream=True, text="Name two fruits"):
    return c.post("/v1/chat/completions", headers=H, json={"stream": stream, "messages": [{"role": "user", "content": text}]})


def test_a_streamed_request_is_done_when_its_stream_ends_with_its_speed(client):
    ask(client)
    e = client.get("/api/keys/log", headers=H).json()[0]
    assert e["done"] and e["status"] == 200 and e["model"] == "qwen"
    assert (e["input_tokens"], e["output_tokens"]) == (9, 6)
    assert e["first_ms"] is not None and e["ms"] >= e["first_ms"]
    assert "prompt" not in e and "reply" not in e  # nobody asked to see them


def test_prompts_and_replies_are_kept_only_while_someone_reads_them(client, monkeypatch):
    ask(client, text="before anyone looked")
    assert "prompt" not in client.get("/api/keys/log?content=true", headers=H).json()[0]  # not recorded back then
    ask(client)
    streamed = client.get("/api/keys/log?content=true", headers=H).json()[0]
    assert streamed["prompt"] == "Name two fruits" and streamed["reply"] == "Two: apple, pear"
    ask(client, stream=False, text="one fruit")
    whole = client.get("/api/keys/log?content=true", headers=H).json()[0]
    assert whole["reply"] == "apple" and whole["first_ms"] is None  # no first token to time when it all comes at once
    # Readers without content=true never get them...
    assert all("prompt" not in e for e in client.get("/api/keys/log", headers=H).json())
    # ...and once nobody has asked for a while, the next request clears them from the log.
    monkeypatch.setattr(api_keys, "_content_until", 0.0)
    ask(client, text="later")
    assert all("prompt" not in e and "reply" not in e for e in api_keys.log)


def test_the_stream_sends_each_change_after_the_last_one_seen(client):
    ask(client)
    since = max(e["seq"] for e in api_keys.log) - 1
    with client.stream("GET", f"/api/keys/log/stream?since={since}", headers=H) as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                e = json.loads(line[5:])
                break
    assert e["path"] == "/v1/chat/completions" and e["done"] and e["seq"] > since
