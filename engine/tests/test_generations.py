import asyncio
import json
from types import SimpleNamespace

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


def fake_model(monkeypatch, pieces, delay=0.0, status=200, engine="llama.cpp"):
    """A model server that streams `pieces` (content, ("think", text), or a delta dict) with `delay` seconds between them.
    `pieces` can be a function of the request body, for replies that depend on it; the requests are kept in the returned list."""
    asked = []

    async def sse(chosen):
        for piece in chosen:
            await asyncio.sleep(delay)
            delta = piece if isinstance(piece, dict) else {"reasoning_content": piece[1]} if isinstance(piece, tuple) else {"content": piece}
            yield f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def handler(request):
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "the model broke"}})
        body = json.loads(request.content)
        asked.append(body)
        return httpx.Response(200, content=sse(pieces(body) if callable(pieces) else pieces), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(generations, "route", lambda body: (SimpleNamespace(engine=engine), "http://model/v1/chat/completions", body))
    monkeypatch.setattr(generations.httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    return asked


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


def start(client, chat_id, **extra):
    body = {"chat_id": chat_id, "upto": 1, "model": "tiny", "body": {"messages": [{"role": "user", "content": "Hi"}]}, **extra}
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
    assert chat["messages"][-1] | {"created": None} == {"role": "assistant", "content": "Hello", "model": "tiny", "error": False, "reasoning": "hmm", "attachments": None, "tools": None, "created": None}
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


def calls_then_answers(call_delta, answer="It's 42."):
    """A model that calls a tool first, then answers once it has a result."""
    return lambda body: [answer] if any(m["role"] == "tool" or "<tool_response" in str(m["content"]) for m in body["messages"]) else call_delta


def test_a_native_tool_call_is_run_and_the_model_answers_with_the_result(client, monkeypatch):
    chat_id = new_chat(client)
    call = [{"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "calculator", "arguments": '{"expression": '}}]}, {"tool_calls": [{"index": 0, "function": {"arguments": '"6*7"}'}}]}]
    asked = fake_model(monkeypatch, calls_then_answers(call))
    gen = start(client, chat_id, tools=["calculator"])
    events = follow(client, gen["id"])
    steps = [e["temper_tool"] for e in events if "temper_tool" in e]
    assert steps[0]["result"] is None and steps[-1]["result"] == "6*7 = 42" and steps[-1]["arguments"] == {"expression": "6*7"}
    assert asked[0]["tools"][0]["function"]["name"] == "calculator"
    assert asked[1]["messages"][-2]["tool_calls"][0]["id"] == "c1" and asked[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "name": "calculator", "content": "6*7 = 42"}
    reply = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["messages"][-1]
    assert reply["content"] == "It's 42." and reply["tools"][0] | {"at": 0} == {"id": "c1", "name": "calculator", "arguments": {"expression": "6*7"}, "result": "6*7 = 42", "error": False, "at": 0}


def test_a_call_written_into_the_text_is_repaired_and_taken_out(client, monkeypatch):
    chat_id = new_chat(client)
    written = ["Let me work it out.\n<tool_call>{'name': 'Calculator', 'arguments': {'expression': '2+2',}}</tool_call>"]
    asked = fake_model(monkeypatch, calls_then_answers(written, "Four."), engine="mlx")
    events = follow(client, start(client, chat_id, tools=["calculator"])["id"])
    assert {"temper_content": "Let me work it out."} in events
    assert "tools" not in asked[0]  # other engines are offered tools in the prompt
    assert asked[1]["messages"][-1]["role"] == "user" and "2+2 = 4" in asked[1]["messages"][-1]["content"]
    reply = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["messages"][-1]
    assert reply["content"] == "Let me work it out.Four." and reply["tools"][0]["at"] == len("Let me work it out.") and reply["tools"][0]["name"] == "calculator"


def test_a_model_that_keeps_calling_tools_is_made_to_answer(client, monkeypatch):
    chat_id = new_chat(client)

    def reply(body):
        n = len(body["messages"])
        return [{"tool_calls": [{"index": 0, "function": {"name": "calculator", "arguments": json.dumps({"expression": f"{n} + 1"})}}]}] if "tools" in body else ["Done."]

    asked = fake_model(monkeypatch, reply)
    follow(client, start(client, chat_id, tools=["calculator"])["id"])
    assert len(asked) == generations.tools.MAX_ROUNDS + 1 and "tools" not in asked[-1]
    reply = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["messages"][-1]
    assert reply["content"] == "Done." and len(reply["tools"]) == generations.tools.MAX_ROUNDS


def test_a_repeated_call_gets_the_earlier_result_and_ends_the_loop(client, monkeypatch):
    chat_id = new_chat(client)
    call = [{"tool_calls": [{"index": 0, "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}}]}]
    asked = fake_model(monkeypatch, lambda body: call if "tools" in body else ["Four."])
    events = follow(client, start(client, chat_id, tools=["calculator"])["id"])
    # It ran once; the repeat was answered from the first result, and the round after it had no tools.
    assert len([e for e in events if "temper_tool" in e]) == 2 and len(asked) == 3 and "tools" not in asked[2]
    assert "same call as before" in asked[2]["messages"][-1]["content"]
    reply = client.get(f"/api/chats/{chat_id}", headers=AUTH).json()["messages"][-1]
    assert reply["content"] == "Four." and len(reply["tools"]) == 1
