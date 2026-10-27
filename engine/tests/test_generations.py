import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import generations
from temper_engine.app import create_app

AUTH = {"Authorization": "Bearer secret"}
REAL_CLIENT = httpx.AsyncClient


@pytest.fixture
def client():
    with TestClient(create_app(token="secret")) as c:
        yield c
    generations.live.clear()


def fake_model(monkeypatch, pieces, delay=0.0, status=200):
    """A model server that streams `pieces` (content, or ("think", text)) with `delay` seconds between them."""

    async def sse():
        for piece in pieces:
            await asyncio.sleep(delay)
            delta = {"reasoning_content": piece[1]} if isinstance(piece, tuple) else {"content": piece}
            yield f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def handler(request):
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "the model broke"}})
        return httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(generations, "route", lambda body: (None, "http://model/v1/chat/completions", body))
    monkeypatch.setattr(generations.httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))


def new_chat(client):
    r = client.post("/api/chats", headers=AUTH, json={"messages": [{"role": "user", "content": "Hi"}]})
    return r.json()["id"]


def follow(client, gen_id, after=0):
    events = []
    with client.stream("GET", f"/api/chat/generations/{gen_id}/stream", params={"after": after}, headers=AUTH) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def start(client, chat_id):
    body = {"chat_id": chat_id, "upto": 1, "model": "tiny", "body": {"messages": [{"role": "user", "content": "Hi"}]}}
    return client.post("/api/chat/generations", headers=AUTH, json=body).json()


def test_a_reply_is_saved_by_the_engine_and_can_be_followed_again(client, monkeypatch):
    fake_model(monkeypatch, [("think", "hmm"), "Hel", "lo"], delay=0.05)
    chat_id = new_chat(client)
    gen = start(client, chat_id)
    assert client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["generating"]["id"] == gen["id"]
    events = follow(client, gen["id"])
    assert events[-1] == {"temper_end": {"error": None, "cancelled": False}} and len(events) == 4
    chat = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()
    assert chat["generating"] is None
    assert chat["messages"][-1] | {"created": None} == {"role": "assistant", "content": "Hello", "model": "tiny", "error": False, "reasoning": "hmm", "attachments": None, "created": None}
    # A page that reopens the chat later replays it all, from any point.
    assert follow(client, gen["id"], after=2)[0]["choices"][0]["delta"] == {"content": "lo"}


def test_stopping_keeps_what_was_written(client, monkeypatch):
    fake_model(monkeypatch, ["One", " two", " three"], delay=0.3)
    chat_id = new_chat(client)
    gen = start(client, chat_id)
    import time

    time.sleep(0.45)
    client.post(f"/api/chat/generations/{gen['id']}/cancel", headers=AUTH)
    messages = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["messages"]
    assert [m["content"] for m in messages] == ["Hi", "One"]
    assert follow(client, gen["id"])[-1]["temper_end"]["cancelled"] is True


def test_a_failed_reply_is_saved_as_an_error(client, monkeypatch):
    fake_model(monkeypatch, [], status=500)
    chat_id = new_chat(client)
    gen = start(client, chat_id)
    assert follow(client, gen["id"])[-1]["temper_end"]["error"] == "the model broke"
    last = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["messages"][-1]
    assert last["error"] and last["content"] == "the model broke"
