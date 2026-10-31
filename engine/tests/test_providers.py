import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from temper_engine import compaction, home, providers, runtime
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
KEY = "sk-test-key-0123456789"


@pytest.fixture
def cloud(monkeypatch):
    """An OpenAI-compatible provider at https://api.example.com/v1 offering two models, and this engine's app."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "api.example.com":
            raise httpx.ConnectError("no route", request=request)
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "bad key"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "big-model"}, {"id": "models/small-model"}]})
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "from the cloud"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    real_async = httpx.AsyncClient
    strip = lambda kw: {k: v for k, v in kw.items() if k != "timeout"}  # noqa: E731
    monkeypatch.setattr(providers.httpx, "get", lambda url, **kw: client.get(url, **strip(kw)))
    monkeypatch.setattr(providers.httpx, "post", lambda url, **kw: client.post(url, **strip(kw)))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw))
    providers._seen.clear()
    with TestClient(create_app(token="t")) as c:
        c.sent = sent
        yield c
    runtime.manager.unload()


def add(c, **kw):
    return c.post("/api/providers", headers=H, json={"kind": "custom", "url": "https://api.example.com/v1/", "key": KEY, "name": "example", **kw})


def test_adding_checks_the_key_and_keeps_it_private(cloud):
    assert "refused the key" in add(cloud, key="wrong").json()["detail"]
    assert add(cloud, url="ftp:nowhere").status_code == 400
    p = add(cloud).json()
    assert p["url"] == "https://api.example.com/v1" and p["online"] and p["available"] == ["big-model", "small-model"]
    assert KEY not in json.dumps(cloud.get("/api/providers", headers=H).json())
    if os.name != "nt":
        assert (home() / "providers.json").stat().st_mode & 0o777 == 0o600
    assert add(cloud).status_code == 409  # the name is taken
    assert add(cloud, name="a@b").status_code == 400
    # One that doesn't list its models can still be added, naming them.
    assert add(cloud, key="wrong", name="named", models=["only-this"]).json()["models"] == ["only-this"]


def test_its_models_are_listed_and_answered_there(cloud):
    p = add(cloud).json()
    listed = {m["id"] for m in cloud.get("/v1/models", headers=H).json()["data"]}
    assert {"big-model@example", "small-model@example"} <= listed
    r = cloud.post("/v1/chat/completions", headers=H, json={"model": "big-model@example", "messages": [{"role": "user", "content": "hi", "error": None}], "top_k": 40, "temperature": 0.5})
    assert r.json()["choices"][0]["message"]["content"] == "from the cloud"
    assert cloud.sent[-1] == {"model": "big-model", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5}  # llama.cpp's own settings left out
    # Choosing models narrows the list.
    cloud.put(f"/api/providers/{p['id']}/models", headers=H, json={"models": ["small-model"]})
    listed = {m["id"] for m in cloud.get("/v1/models", headers=H).json()["data"]}
    assert "small-model@example" in listed and "big-model@example" not in listed
    cloud.delete(f"/api/providers/{p['id']}", headers=H)
    assert not any(m["id"].endswith("@example") for m in cloud.get("/v1/models", headers=H).json()["data"])


def test_loading_one_makes_it_chats_model(cloud):
    add(cloud)
    status = cloud.post("/api/runtime/load", headers=H, json={"path": "big-model@example"}).json()
    assert status["state"] == "ready" and status["engine"] == "cloud" and status["model"] == "big-model@example"
    r = cloud.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["choices"][0]["message"]["content"] == "from the cloud" and cloud.sent[-1]["model"] == "big-model"
    # Freeing the GPU leaves it loaded: it holds nothing here.
    assert cloud.post("/api/runtime/suspend", headers=H).json()["model"] == "big-model@example"
    # Compaction estimates its tokens and summarises through the provider.
    assert compaction.count(runtime.manager, [{"role": "user", "content": "x" * 400}]) > 50
    assert compaction.summarise(runtime.manager, None, [{"role": "user", "content": "hello"}]) == "from the cloud"
    cloud.post("/api/runtime/unload", headers=H)
    assert runtime.manager.engine == "llama.cpp"


def test_requests_are_shaped_for_each_provider():
    openai = providers.Cloud({"id": "x", "name": "openai", "kind": "openai", "url": "https://api.openai.com/v1", "key": KEY}, "o3-mini")
    out = providers.translate(openai, {"messages": [], "max_tokens": 50, "temperature": 0.7, "min_p": 0.1, "stream": True})
    assert out == {"model": "o3-mini", "messages": [], "max_completion_tokens": 50, "stream": True}
    gpt = providers.Cloud({**openai.provider}, "gpt-4.1")
    assert providers.translate(gpt, {"temperature": 0.7})["temperature"] == 0.7
    anthropic = providers.Cloud({"id": "y", "name": "claude", "kind": "anthropic", "url": "https://api.anthropic.com/v1", "key": KEY}, "claude-sonnet-5-5")
    assert anthropic.headers["x-api-key"] == KEY and anthropic.url == "https://api.anthropic.com/v1/chat/completions"
    assert providers.translate(anthropic, {"max_tokens": 50})["max_tokens"] == 50
