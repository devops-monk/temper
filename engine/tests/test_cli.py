import json
import os
import stat
import sys

import pytest

from temper_engine import cli, home

ENGINE = {"url": "http://127.0.0.1:8765", "token": "k" * 32, "pid": 1}


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(cli, "running", lambda: ENGINE)
    return tmp_path


def test_the_engine_leaves_its_address_for_the_cli_and_takes_it_away(monkeypatch):
    with pytest.raises(cli.CliError, match="isn't running"):
        cli.running()
    cli.write_running("http://127.0.0.1:9", "tok")
    path = home() / cli.RUNNING
    assert json.loads(path.read_text()) == {"url": "http://127.0.0.1:9", "token": "tok", "pid": os.getpid()}
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(cli.CliError, match="isn't answering"):
        cli.running()
    path.write_text(json.dumps({"url": "x", "token": "t", "pid": -5}))
    cli.clear_running()  # another engine's: left alone
    assert path.exists()
    cli.write_running("http://127.0.0.1:9", "tok")
    cli.clear_running()
    assert not path.exists()


def test_agents_start_pointed_at_the_loaded_model():
    command, env = cli.launch("codex", ENGINE, "qwen")
    assert command[0] == "codex" and command[-2:] == ["-m", "qwen"] and 'base_url = "http://127.0.0.1:8765/v1"' in command[4] and env == {"TEMPER_API_KEY": ENGINE["token"]}
    command, env = cli.launch("opencode", ENGINE, "qwen")
    assert command == ["opencode", "--model", "temper/qwen"] and json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["temper"]["models"] == {"qwen": {"name": "qwen"}}
    assert cli.launch("aider", ENGINE, "qwen") == (["aider", "--model", "openai/qwen"], {"OPENAI_API_BASE": "http://127.0.0.1:8765/v1", "OPENAI_API_KEY": ENGINE["token"]})
    command, env = cli.launch("claude", ENGINE, "qwen")
    assert command == ["claude"] and env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8765" and env["ANTHROPIC_AUTH_TOKEN"] == ENGINE["token"] and env["ANTHROPIC_MODEL"] == "qwen"
    with pytest.raises(cli.CliError, match="claude, codex, opencode, aider"):
        cli.launch("nope", ENGINE, "qwen")


def test_a_missing_agent_says_how_to_install_it(user_home, monkeypatch):
    monkeypatch.setattr(cli, "loaded_model", lambda engine: "qwen")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["start", "aider"]) == 1
    assert cli.main(["start"]) == 2


def test_mcp_add_writes_each_agents_config_and_keeps_what_was_there(user_home):
    codex = user_home / ".codex/config.toml"
    codex.parent.mkdir()
    codex.write_text('model = "o4"\n\n[mcp_servers.temper]\nurl = "http://old/mcp"\n\n[mcp_servers.other]\ncommand = "x"\n')
    cli.add("codex")
    text = codex.read_text()
    assert 'model = "o4"' in text and "http://old" not in text and text.count("[mcp_servers.temper]") == 1 and '[mcp_servers.other]\ncommand = "x"' in text
    assert 'url = "http://127.0.0.1:8765/mcp"' in text and (user_home / ".codex/config.toml.bak").exists()
    if sys.version_info >= (3, 11):
        import tomllib

        assert tomllib.loads(text)["mcp_servers"]["temper"]["bearer_token_env_var"] == "TEMPER_API_KEY"

    cursor = user_home / ".cursor/mcp.json"
    cursor.parent.mkdir()
    cursor.write_text(json.dumps({"mcpServers": {"mine": {"command": "y"}}}))
    cli.add("cursor")
    assert json.loads(cursor.read_text())["mcpServers"] == {"mine": {"command": "y"}, "temper": {"url": "http://127.0.0.1:8765/mcp", "headers": {"Authorization": f"Bearer {ENGINE['token']}"}}}
    cli.add("vscode")
    path, key = cli.json_config("vscode")
    assert key == "servers" and json.loads(path.read_text())["servers"]["temper"]["type"] == "http"

    found = {label: (where != "not added", current) for label, where, current in cli.registered(ENGINE)}
    assert found == {"Claude Code": (False, False), "Codex": (True, True), "Cursor": (True, True), "VS Code": (True, True)}
    assert not any(current for *_, current in cli.registered({**ENGINE, "url": "http://127.0.0.1:9999"}))

    cursor.write_text("{not json")
    with pytest.raises(cli.CliError, match="isn't valid JSON"):
        cli.add("cursor")


@pytest.fixture
def engine(monkeypatch):
    """The commands talk to a real engine app, in-process."""
    from fastapi.testclient import TestClient

    from temper_engine.app import create_app

    client = TestClient(create_app(token="t"), headers={"Authorization": "Bearer t"})
    monkeypatch.setattr(cli, "running", lambda: {"url": "http://testserver", "token": "t", "pid": os.getpid()})
    monkeypatch.setattr(cli, "_client", lambda engine: client)
    with client:
        yield client


MODELS = [
    {"id": "Qwen/Qwen3-4B-GGUF", "path": "/m/qwen", "safetensors": False, "gguf": [{"name": "Qwen3-4B-Q8_0.gguf", "path": "/m/q8.gguf"}, {"name": "Qwen3-4B-Q4_K_M.gguf", "path": "/m/q4.gguf"}]},
    {"id": "mlx-community/Qwen3-0.6B-4bit", "path": "/m/mlx", "safetensors": True, "gguf": []},
]


def test_a_model_is_found_by_name_file_or_quant():
    assert cli.resolve_model(MODELS, "qwen/qwen3-4b-gguf") == ("/m/q4.gguf", "Qwen3-4B-Q4_K_M.gguf")  # Q4_K_M when there's a choice
    assert cli.resolve_model(MODELS, "Qwen3-4B-GGUF", "q8_0")[0] == "/m/q8.gguf"
    assert cli.resolve_model(MODELS, "Qwen3-4B-Q8_0.gguf")[0] == "/m/q8.gguf"
    assert cli.resolve_model(MODELS, "0.6B") == ("/m/mlx", "mlx-community/Qwen3-0.6B-4bit")
    with pytest.raises(cli.CliError, match="temper models"):
        cli.resolve_model(MODELS, "llama")
    with pytest.raises(cli.CliError, match="no IQ2_M"):
        cli.resolve_model(MODELS, "Qwen3-4B-GGUF", "IQ2_M")


def test_a_yaml_config_is_checked_and_printed_in_full_on_a_dry_run(tmp_path, capsys):
    cfg = tmp_path / "tutor.yaml"
    cfg.write_text("base_model: mlx-community/Qwen3-0.6B-4bit\ndataset: ./math.jsonl\nmethod: lora\nmax_steps: 30\n")
    (tmp_path / "math.jsonl").write_text("{}\n")
    assert cli.train_config(str(cfg), [])["dataset"] == str((tmp_path / "math.jsonl").resolve())  # beside the config
    assert cli.train_config(str(cfg), [])["base_model"] == "mlx-community/Qwen3-0.6B-4bit"  # a Hub name stays one
    assert cli.main(["train", str(cfg), "--set", "learning_rate=1e-4", "--set", "name=tutor", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "base_model: mlx-community/Qwen3-0.6B-4bit" in out and "learning_rate: 0.0001" in out and "max_steps: 30" in out and "lora_r: 16" in out

    assert cli.main(["train", str(cfg), "--set", "learnig_rate=1e-4", "--dry-run"]) == 1
    assert "Unknown settings: learnig_rate" in capsys.readouterr().err
    assert cli.main(["train", str(cfg), "--set", "max_steps=-3", "--dry-run"]) == 1
    assert "max_steps" in capsys.readouterr().err
    cfg.write_text("- just\n- a list\n")
    assert cli.main(["train", str(cfg), "--dry-run"]) == 1
    assert "should be settings" in capsys.readouterr().err


def test_runs_and_exports_go_through_the_engine(engine, capsys):
    run = home() / "runs" / "20261031-090000-tutor"
    (run / "adapter").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"name": "tutor", "base_model": "Qwen/Qwen3-0.6B", "backend": "gpu", "method": "lora"}))
    assert cli.main(["runs"]) == 0
    assert "20261031-090000-tutor" in capsys.readouterr().out
    # The engine's own checks come back as plain messages.
    assert cli.main(["export", "tutor", "--format", "mlx"]) == 1
    assert "Apple Silicon" in capsys.readouterr().err
    assert cli.main(["export", "nothing-like-it"]) == 1
    assert "No run matches" in capsys.readouterr().err


def test_chat_needs_a_loaded_model_and_load_says_what_it_couldnt_find(engine, capsys):
    assert cli.main(["chat", "hello"]) == 1
    assert "No model is loaded" in capsys.readouterr().err
    assert cli.main(["load", "no-such-model"]) == 1
    assert "No model called 'no-such-model'" in capsys.readouterr().err


def test_serve_uses_the_apps_key_and_refuses_a_second_engine(monkeypatch):
    key = cli.stable_key()
    assert len(key) == 32 and cli.stable_key() == key and (home() / "api-key").read_text() == key
    monkeypatch.setattr(cli, "running", lambda: ENGINE)
    assert cli.main(["serve"]) == 1
