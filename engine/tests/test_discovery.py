import json
import socket
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import discovery, prefs

SERVER = SimpleNamespace(model="tutor-Q4_K_M.gguf", path="/models/tutor-Q4_K_M.gguf", context=8192, modalities={"vision": True}, reasoning=None, state="ready")


@pytest.fixture
def model(monkeypatch):
    """A loaded model whose server answers with a fixed reply, recording what it was asked."""
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        if body.get("stream"):
            chunks = [{"choices": [{"delta": {"content": "<think>hm</think>Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
                      {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "time", "arguments": '{"tz":'}}]}}]},
                      {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"UTC"}'}}]}, "finish_reason": "tool_calls"}]},
                      {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "Hello"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 1}})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr("temper_engine.runtime.route", lambda body: (SERVER, "http://model/v1/chat/completions", body))
    monkeypatch.setattr(discovery, "_servers", lambda: [SERVER])
    return asked


def test_ollama_clients_find_the_loaded_model(model):
    c = TestClient(discovery.compat_app())
    assert c.get("/").text == "Ollama is running" and c.get("/api/version").json()["version"]
    tag = c.get("/api/tags").json()["models"][0]
    assert tag["name"] == "tutor-Q4_K_M.gguf" and tag["details"]["quantization_level"] == "Q4_K_M" and len(tag["digest"]) == 64
    show = c.post("/api/show", json={"model": "tutor-Q4_K_M.gguf"}).json()
    assert "vision" in show["capabilities"] and show["model_info"]["general.context_length"] == 8192
    assert c.get("/health").json() == {"status": "ok"} and c.get("/props").json()["default_generation_settings"]["n_ctx"] == 8192


def test_ollama_chat_is_answered_whole_and_streamed(model):
    c = TestClient(discovery.compat_app())
    whole = c.post("/api/chat", json={"model": "x", "stream": False, "messages": [{"role": "user", "content": "hi", "images": ["iVBORw0KGgo="]}], "options": {"temperature": 0.2, "num_predict": 50}}).json()
    assert whole["message"]["content"] == "Hello" and whole["done"] and whole["eval_count"] == 1
    assert model[0]["temperature"] == 0.2 and model[0]["max_tokens"] == 50 and model[0]["stream"] is False
    assert model[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    lines = [json.loads(line) for line in c.post("/api/chat", json={"model": "x", "messages": [{"role": "user", "content": "hi"}]}).text.splitlines()]
    assert "".join(line["message"]["content"] for line in lines if not line["done"]) == "Hello"  # thinking left out
    assert lines[-2]["message"]["tool_calls"] == [{"function": {"name": "time", "arguments": {"tz": "UTC"}}}]
    assert lines[-1]["done"] and lines[-1]["prompt_eval_count"] == 7 and lines[-1]["eval_count"] == 3

    gen = c.post("/api/generate", json={"model": "x", "prompt": "hi", "system": "be brief", "stream": False}).json()
    assert gen["response"] == "Hello" and model[-1]["messages"][0] == {"role": "system", "content": "be brief"}
    assert c.post("/api/generate", json={"model": "x"}).json()["done_reason"] == "load"


def test_a_tool_reply_goes_back_as_openai_messages():
    out = discovery.to_openai({"messages": [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "time", "arguments": {"tz": "UTC"}}}]},
        {"role": "tool", "content": "12:00", "tool_name": "time"},
    ], "format": {"type": "object"}, "stream": False})
    assert out["messages"][0]["tool_calls"][0]["function"] == {"name": "time", "arguments": '{"tz": "UTC"}'}
    assert out["messages"][1] == {"role": "tool", "content": "12:00", "name": "time"}
    assert out["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_the_listeners_follow_the_setting_and_report_a_taken_port(monkeypatch):
    from temper_engine.app import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    monkeypatch.setitem(discovery.PORTS, "llamacpp", free)
    monkeypatch.setattr(discovery, "_ollama_running", lambda: True)
    h = {"Authorization": "Bearer t"}
    with TestClient(create_app(token="t")) as c:
        on = c.post("/api/discovery", headers=h, json={"llamacpp": True}).json()
        try:
            assert on["llamacpp"]["running"] and httpx.get(f"http://127.0.0.1:{free}/").text == "Ollama is running"
        finally:
            c.post("/api/discovery", headers=h, json={"llamacpp": False})
        r = c.post("/api/discovery", headers=h, json={"ollama": True})
        assert r.status_code == 409 and "Ollama itself is running" in r.json()["detail"]
    assert prefs.load()["discovery"] == {"llamacpp": False, "ollama": False}


def test_a_port_another_app_answers_on_counts_as_taken():
    from temper_engine.lan import Background, answering

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert answering(port)
        b = Background("test")
        b.start(discovery.compat_app(), "127.0.0.1", port)
        assert not b.running and "in use" in b.error
