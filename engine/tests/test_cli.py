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
