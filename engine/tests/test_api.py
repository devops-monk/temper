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


def test_parses_mlx_lm_table_output():
    from temper_engine.train_worker import parse_mlx_line

    assert parse_mlx_line("  \x1b[38;5;244m  12\x1b[0m    \x1b[1;32m1.205 ▼\x1b[0m    \x1b[1m  133\x1b[0m    \x1b[38;5;244m  9.1k\x1b[0m") == {
        "step": 12, "loss": 1.205, "tokens_per_sec": 133.0
    }
    assert parse_mlx_line("     1    val 1.623    59.44s") == {"step": 1, "val_loss": 1.623}
    assert parse_mlx_line("  iter   train_loss     tok/s     tokens") is None
    assert parse_mlx_line("Trainable parameters: 0.182% (7.340M/4022.468M)") is None


def test_parses_older_mlx_lm_output():
    from temper_engine.train_worker import parse_mlx_line

    line = "Iter 10: Train loss 2.345, Learning Rate 1.000e-05, It/sec 0.5, Tokens/sec 512.3, Trained Tokens 900"
    assert parse_mlx_line(line) == {"step": 10, "loss": 2.345, "lr": 1e-05, "tokens_per_sec": 512.3}
    assert parse_mlx_line("Iter 1: Val loss 1.900, Val took 3.2s") == {"step": 1, "val_loss": 1.9}


def test_worker_logs_are_plain_text():
    from temper_engine.jobs import ANSI

    assert ANSI.sub("", "\x1b[38;5;244m  12\x1b[0m    \x1b[1;32m0.731 ▼\x1b[0m\x1b[?25l") == "  12    0.731 ▼"
