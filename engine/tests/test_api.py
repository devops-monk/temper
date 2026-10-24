from fastapi.testclient import TestClient

from temper_engine.app import create_app
from temper_engine.datasets import detect_format
from temper_engine.train_worker import to_messages

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def test_health_is_open():
    assert client.get("/api/health").json()["ok"] is True


def test_api_requires_token():
    assert client.get("/api/system").status_code == 401
    assert client.get("/api/system", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/models").status_code == 401


def test_system_probe_shape():
    info = client.get("/api/system", headers=AUTH).json()
    assert info["memory"]["total"] > 0
    assert info["backend"] in {"cuda", "rocm", "metal", "xpu", "cpu"}
    assert set(info["capabilities"]) >= {"gguf_inference", "trainer"}


def test_chat_without_model_explains_what_to_do():
    r = client.post("/v1/chat/completions", headers=AUTH, json={"messages": []})
    assert r.status_code == 409
    assert "Load a model" in r.json()["detail"]


def test_detect_format():
    assert detect_format(["messages"]) == "messages"
    assert detect_format(["conversations", "source"]) == "sharegpt"
    assert detect_format(["instruction", "input", "output"]) == "alpaca"
    assert detect_format(["text"]) == "text"
    assert detect_format(["question", "answer"]) is None


def test_sharegpt_rows_become_chat_messages():
    row = {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}
    assert to_messages(row, "sharegpt") == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]


def test_alpaca_input_is_appended_to_prompt():
    row = {"instruction": "Translate", "input": "bonjour", "output": "hello"}
    assert to_messages(row, "alpaca")[0]["content"] == "Translate\n\nbonjour"
