import json
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from temper_engine import home, usage
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}


def test_statistics_count_tokens_runs_exports_and_datasets():
    usage.add("qwen", 100, 40)
    usage.add("qwen", 50, 10)
    usage.add("llama", 5, 70)
    usage.add("nothing", 0, 0)  # no tokens, not a reply
    run = home() / "runs" / "20261031-100000-tutor"
    (run / "adapter").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [{"event": "phase", "t": 1000.0}, {"event": "metrics", "step": 60, "t": 1600.0}]))
    (home() / "exports" / "20261031-100000-tutor-101500").mkdir(parents=True)
    (home() / "datasets").mkdir()
    (home() / "datasets" / "math.jsonl").write_text("{}\n{}\n{}\n")
    (home() / "datasets" / "math.eval.jsonl").write_text("{}\n")
    with TestClient(create_app(token="t")) as c:
        s = c.get("/api/stats", headers=H).json()
    assert s["tokens"]["output"] == 120 and s["tokens"]["input"] == 155 and s["tokens"]["replies"] == 3
    assert s["tokens"]["models"][0] == {"model": "llama", "output_tokens": 70} and len(s["tokens"]["days"]) == 30
    assert s["tokens"]["days"][-1]["output_tokens"] == 120
    assert s["training"] == {"runs": 1, "finished": 1, "steps": 60, "seconds": 600}
    assert s["exports"] == 1 and s["datasets"] == 1 and s["dataset_rows"] == 3 and s["chats"] == 0
    usage.flush()
    assert json.loads((home() / "usage.json").read_text())["days"]


def test_streamed_replies_are_counted_without_sending_an_unasked_usage_chunk(monkeypatch):
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content))
        chunks = [{"choices": [{"delta": {"content": "Hi"}}]}, {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}]
        return httpx.Response(200, text="".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    server = SimpleNamespace(model="qwen")
    monkeypatch.setattr("temper_engine.runtime.route", lambda body: (server, "http://model/v1/chat/completions", body))
    with TestClient(create_app(token="t")) as c:
        plain = c.post("/v1/chat/completions", headers=H, json={"stream": True, "messages": []}).text
        wanted = c.post("/v1/chat/completions", headers=H, json={"stream": True, "stream_options": {"include_usage": True}, "messages": []}).text
        s = c.get("/api/stats", headers=H).json()
    assert asked[0]["stream_options"] == {"include_usage": True}  # asked for on the client's behalf
    assert '"usage"' not in plain and '"Hi"' in plain and "[DONE]" in plain
    assert '"usage"' in wanted
    assert s["tokens"]["output"] == 4 and s["tokens"]["input"] == 18
