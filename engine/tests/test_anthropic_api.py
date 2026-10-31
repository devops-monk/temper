import asyncio
import json

from fastapi.testclient import TestClient

from temper_engine.anthropic_api import ThinkFilter, from_openai, to_openai, translate_stream
from temper_engine.app import create_app


def test_a_conversation_with_tools_and_images_is_translated_to_openai():
    body = {
        "model": "claude-sonnet-5-5",
        "system": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": " Use tools."}],
        "max_tokens": 100,
        "stop_sequences": ["END"],
        "tools": [{"name": "weather", "description": "Weather", "input_schema": {"type": "object"}}, {"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "any"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Look:"}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "sunny"}]}, {"type": "text", "text": "And?"}]},
        ],
    }
    out = to_openai(body)
    assert out["messages"][0] == {"role": "system", "content": "Be brief. Use tools."}
    assert out["messages"][1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
    assert out["messages"][2] == {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}]}
    assert out["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "sunny"}
    assert out["messages"][4] == {"role": "user", "content": "And?"}
    assert [t["function"]["name"] for t in out["tools"]] == ["weather"]  # the server-side tool is left out
    assert out["tool_choice"] == "required" and out["stop"] == ["END"] and out["max_tokens"] == 100 and out["stream"] is False


def test_a_reply_is_translated_back_without_its_thinking():
    reply = {
        "choices": [{"finish_reason": "tool_calls", "message": {"content": "<think>let me see</think>\n\nSure.", "tool_calls": [{"id": "c1", "function": {"name": "weather", "arguments": '{"city":"Oslo"}'}}]}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    msg = from_openai(reply, "claude-x")
    assert msg["content"] == [{"type": "text", "text": "Sure."}, {"type": "tool_use", "id": "c1", "name": "weather", "input": {"city": "Oslo"}}]
    assert msg["stop_reason"] == "tool_use" and msg["usage"] == {"input_tokens": 12, "output_tokens": 7} and msg["model"] == "claude-x"
    assert from_openai({"choices": [{"finish_reason": "length", "message": {"content": "abc"}}]}, "m")["stop_reason"] == "max_tokens"


def test_think_filter_holds_back_only_what_might_be_thinking():
    f = ThinkFilter()
    assert [f.feed(t) for t in ["<thi", "nk>reasoning", "</think>\n", "Hello"]] == ["", "", "", "Hello"]
    f = ThinkFilter()
    assert f.feed("<") == "" and f.feed("b>bold") == "<b>bold" and f.feed(" more") == " more"
    f = ThinkFilter()
    assert f.feed("<th") == "" and f.flush() == "<th"


def test_a_stream_with_text_and_a_tool_call_becomes_anthropic_events():
    chunks = [
        {"choices": [{"delta": {"content": "<think>x</think>"}}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "weather", "arguments": '{"ci'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty":"Rome"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 9}},
    ]

    async def lines():
        for c in chunks:
            yield f"data: {json.dumps(c)}"
        yield "data: [DONE]"

    async def collect():
        return [e async for e in translate_stream(lines(), "m") if e]

    events = []
    for raw in b"".join(asyncio.run(collect())).decode().strip().split("\n\n"):
        name, data = raw.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    assert [n for n, _ in events] == ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "Hi"}
    assert events[4][1]["content_block"] == {"type": "tool_use", "id": "c1", "name": "weather", "input": {}} and events[4][1]["index"] == 1
    assert "".join(e["delta"]["partial_json"] for n, e in events if n == "content_block_delta" and e["index"] == 1) == '{"city":"Rome"}'
    assert events[8][1]["delta"]["stop_reason"] == "tool_use" and events[8][1]["usage"]["output_tokens"] == 9


def test_without_a_model_the_error_is_in_anthropics_shape():
    with TestClient(create_app(token="t")) as c:
        r = c.post("/v1/messages", headers={"x-api-key": "t"}, json={"model": "claude-x", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 409 and r.json()["type"] == "error" and r.json()["error"]["type"] == "invalid_request_error"
        assert c.post("/v1/messages/count_tokens", headers={"x-api-key": "t"}, json={"messages": [{"role": "user", "content": "hello there"}]}).json()["input_tokens"] > 0
        assert c.post("/v1/messages", headers={"x-api-key": "wrong"}, json={}).status_code == 401


def test_a_slow_model_gets_pings_while_it_reads_the_prompt():
    from temper_engine.anthropic_api import with_pings

    async def slow():
        await asyncio.sleep(0.25)
        yield b"first"

    async def collect():
        return [e async for e in with_pings(slow(), every=0.1)]

    events = asyncio.run(collect())
    assert events[-1] == b"first" and events.count(b'event: ping\ndata: {"type": "ping"}\n\n') == 2
