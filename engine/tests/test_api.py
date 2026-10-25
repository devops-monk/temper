import json

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


def test_downgrade_tokenizer_config(tmp_path):
    from temper_engine.export_worker import downgrade_tokenizer_config

    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps({"extra_special_tokens": ["<|im_start|>", "<|im_end|>"], "eos_token": "<|im_end|>"}))
    downgrade_tokenizer_config(tmp_path)
    config = json.loads(path.read_text())
    assert "extra_special_tokens" not in config
    assert config["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"]
    assert config["eos_token"] == "<|im_end|>"


def test_recipe_templates_are_consistent():
    from temper_engine.recipes import TEMPLATES, Recipe, check

    for template in TEMPLATES:
        recipe = Recipe(**template)
        if recipe.seed and recipe.seed.get("type") == "documents":
            recipe.seed["sources"] = ["notes.md"]
        if recipe.seed and recipe.seed.get("type") == "file":
            recipe.seed["path"] = "rows.jsonl"
        check(recipe)  # raises HTTPException on a broken template
        names = [c["name"] for c in recipe.columns if not c.get("drop")]
        assert {"instruction", "output"} <= set(names), template["id"]  # trainable as Alpaca


def test_recipe_check_rejects_unknown_model():
    import pytest
    from fastapi import HTTPException

    from temper_engine.recipes import Recipe, check

    recipe = Recipe(name="x", models=[], columns=[{"column_type": "llm-text", "name": "a", "model_alias": "writer", "prompt": "hi"}])
    with pytest.raises(HTTPException):
        check(recipe)


def test_recipe_run_needs_loaded_model(tmp_path, monkeypatch):
    from temper_engine.recipes import TEMPLATES

    monkeypatch.setenv("TEMPER_HOME", str(tmp_path))
    client = TestClient(create_app())
    template = next(t for t in TEMPLATES if t["id"] == "topic-qa")
    r = client.post("/api/recipes/run", json={"recipe": template, "mode": "preview", "num_records": 2})
    assert r.status_code == 409


def test_chunking_overlaps_and_breaks_on_whitespace():
    from temper_engine.recipe_worker import chunk

    text = " ".join(f"word{i}" for i in range(500))
    parts = chunk(text, 300, 50)
    assert len(parts) > 5
    assert all(len(p) <= 300 for p in parts)
    assert all(not p.startswith("ord") for p in parts)  # never splits inside a word
    assert parts[0][-20:].split()[-1] in parts[1]  # neighbours share text


def test_recipe_config_from_file_seed(tmp_path):
    from temper_engine.recipe_worker import build_config

    recipe = {
        "models": [{"alias": "writer", "provider": "local", "temperature": 0.5}],
        "seed": {"type": "file", "path": str(tmp_path / "rows.jsonl"), "sampling": "shuffle"},
        "columns": [{"column_type": "llm-text", "name": "a", "model_alias": "writer", "prompt": "{{ text }}"}],
    }
    config = build_config(recipe, tmp_path)
    assert config["seed_config"] == {"source": {"seed_type": "local", "path": str(tmp_path / "rows.jsonl")}, "sampling_strategy": "shuffle"}
    assert config["model_configs"][0]["inference_parameters"] == {"temperature": 0.5}


def test_guess_mapping_for_common_column_names():
    from temper_engine.formats import guess_mapping

    assert guess_mapping(["question", "answer", "id"]) == ("alpaca", {"instruction": "question", "output": "answer"})
    assert guess_mapping(["prompt", "completion", "context"]) == ("alpaca", {"instruction": "prompt", "output": "completion", "input": "context"})
    assert guess_mapping(["conversation"]) == ("messages", {"messages": "conversation"})
    assert guess_mapping(["id", "label"]) is None


def test_mapping_system_prompt_and_csv_turns():
    from temper_engine.formats import to_messages

    row = {"q": "Hi?", "a": "Hello."}
    conv = to_messages(row, "alpaca", {"instruction": "q", "output": "a"}, "Be brief.")
    assert conv == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello."}]
    csv_row = {"chat": json.dumps([{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}])}
    assert to_messages(csv_row, "sharegpt", {"conversations": "chat"})[1] == {"role": "assistant", "content": "4"}


def test_clean_dedupes_filters_and_splits():
    from temper_engine.formats import CleanOptions, clean

    rows = [{"instruction": f"question {i}", "output": "an answer that is long enough"} for i in range(20)]
    rows += [{"instruction": "Question 0 ", "output": "an  answer that is long enough"}]  # duplicate after folding
    rows += [{"instruction": "q", "output": "ok"}]  # too short
    rows += [{"instruction": "no reply", "output": ""}]  # malformed
    result = clean(rows, "alpaca", None, None, CleanOptions(min_chars=20, eval_fraction=0.1))
    assert result.stats == {"rows_in": 23, "malformed": 1, "duplicates": 1, "too_short": 1, "too_long": 0, "train": 18, "eval": 2}
    assert len({json.dumps(c) for c in result.train + result.eval}) == 20


def test_render_previews_mapped_rows(tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text("\n".join(json.dumps({"question": f"Q{i}", "answer": f"A{i}" if i else ""}) for i in range(3)))
    r = client.post("/api/datasets/render", headers=AUTH, json={"source": str(data), "format": "alpaca", "mapping": {"instruction": "question", "output": "answer"}})
    body = r.json()
    assert r.status_code == 200 and body["usable"] == 2
    assert "error" in body["rows"][0] and body["rows"][1]["conversation"][0]["content"] == "Q1"
    preview = client.get("/api/datasets/preview", headers=AUTH, params={"source": str(data)}).json()
    assert preview["format"] is None and preview["suggested"]["mapping"] == {"instruction": "question", "output": "answer"}


class _ThinkTemplate:
    """Mimics Qwen3's template: history assistant turns get an empty think block, the generation prompt doesn't."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = ""
        for m in messages:
            body = f"<think>\n\n</think>\n\n{m['content']}" if m["role"] == "assistant" else m["content"]
            out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_training_text_matches_inference_prompt():
    from temper_engine.chat_render import render

    prompt, completion = render(_ThinkTemplate(), [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}])
    assert prompt == "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert completion == "4<|im_end|>\n"  # no think block the model would never see at inference


def test_reasoning_support_from_template():
    from temper_engine.runtime import reasoning_support

    qwen3 = {"chat_template": "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>{% endif %}", "chat_template_caps": {}}
    assert reasoning_support(qwen3) == {"thinks": True, "toggle": True, "effort": False}
    gpt_oss = {"chat_template": "Reasoning: {{ reasoning_effort }}<|channel|>analysis", "chat_template_caps": {"supports_reasoning_effort": True}}
    assert reasoning_support(gpt_oss) == {"thinks": True, "toggle": False, "effort": True}
    assert reasoning_support({"chat_template": "<|im_start|>{{ message.content }}"}) == {"thinks": False, "toggle": False, "effort": False}
