import contextlib
import json
import os
import stat
import sys

import httpx
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
    # Each command's `with` would otherwise shut the app down (and unload every model) as it ends.
    monkeypatch.setattr(cli, "_client", lambda engine: contextlib.nullcontext(client))
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


@pytest.fixture
def servers(tmp_path, monkeypatch):
    """Two downloaded GGUFs; their servers are ready the moment they start."""
    import time

    from temper_engine import compare, hub, library
    from temper_engine.runtime import RuntimeManager, manager

    rows = []
    for name in ("small", "mid"):
        path = tmp_path / f"{name}-Q4_K_M.gguf"
        path.write_bytes(b"GGUF")
        rows.append({"id": f"acme/{name}-GGUF", "gguf": [{"name": path.name, "path": str(path), "size": 1}]})
    monkeypatch.setattr(library, "entries", lambda: rows)  # /api/models/local lists these and nothing from the cache
    monkeypatch.setattr(hub, "scan_cache_dir", lambda folder: (_ for _ in ()).throw(OSError("no cache")))

    def fake_load(self, req):
        self.model, self.path, self.state, self.port, self.used, self.engine, self.context = req.name, req.path, "ready", 1, time.time(), "llama.cpp", 8192

    def fake_unload(self):
        self.model, self.path, self.state = None, None, "idle"

    monkeypatch.setattr(RuntimeManager, "load", fake_load)
    monkeypatch.setattr(RuntimeManager, "unload", fake_unload)
    monkeypatch.setattr(compare.Pool, "_start_reaper", lambda self: None)
    yield rows
    for d in (compare.pool.servers, compare.pool.used, compare.pool.kept, compare.pool.on_demand, compare.pool.ttl, compare.pool.aliases):
        d.clear()
    fake_unload(manager)


def test_ps_load_and_unload_manage_several_models(engine, servers, capsys):
    assert cli.main(["ps"]) == 0
    assert "No models are loaded" in capsys.readouterr().out
    assert cli.main(["load", "acme/small-GGUF", "--ttl", "120", "--alias", "small"]) == 0
    assert "Loaded small with 8192 tokens" in capsys.readouterr().out
    assert cli.main(["load", "mid", "--alongside"]) == 0
    assert "alongside chat's model" in capsys.readouterr().out
    assert cli.main(["ps"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["MODEL", "ROLE", "ENGINE", "CONTEXT", "STATE", "IDLE", "UNLOADS", "IN"]
    assert lines[1].split()[:2] == ["small", "chat"] and lines[1].split()[-1] in ("2m", "1m")  # its own ttl
    assert lines[2].split()[1] == "alongside" and lines[2].split()[-1] in ("1h00m", "59m")  # Settings' hour
    assert cli.main(["ps", "--json"]) == 0
    assert [m["ttl"] for m in json.loads(capsys.readouterr().out)["models"]] == [120, None]

    assert cli.main(["unload", "nope"]) == 1
    assert "No loaded model called 'nope'" in capsys.readouterr().err
    assert cli.main(["unload", "--all"]) == 0
    assert capsys.readouterr().out.splitlines() == ["Unloaded mid-Q4_K_M.gguf", "Unloaded small"]


def test_load_checks_the_fit_first(engine, servers, monkeypatch, capsys):
    est = {"total": 9e9, "weights": 5e9, "kv_cache": 3.7e9, "overhead": 3e8, "context": 8192, "verdict": "too_large", "budget": {"free": 4e9}}
    monkeypatch.setattr(cli, "_estimate", lambda c, path, context: est)
    assert cli.main(["load", "small", "--estimate-only"]) == 0
    assert "needs about 9.0 GB" in capsys.readouterr().out
    assert cli.main(["load", "small"]) == 1
    assert "pass -y to try anyway" in capsys.readouterr().err
    assert cli.main(["load", "small", "-y"]) == 0


def test_status_says_what_temper_is_doing(engine, servers, capsys):
    assert cli.main(["load", "small"]) == 0
    capsys.readouterr()
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "running at http://testserver/v1, reachable from this computer" in out
    assert "small-Q4_K_M.gguf (chat)" in out and "none running" in out
    assert cli.main(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["running"] and data["loaded"]["models"][0]["model"] == "small-Q4_K_M.gguf" and data["jobs"] == []


def test_status_and_version_work_with_temper_closed(monkeypatch, capsys):
    from temper_engine import __version__

    def closed():
        raise cli.CliError("Temper isn't running.")

    monkeypatch.setattr(cli, "running", closed)
    assert cli.main(["status", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == {"running": False, "detail": "Temper isn't running."}
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out == f"temper {__version__}\n"


def test_agents_get_a_loaded_model_not_one_that_would_load(engine, servers):
    with pytest.raises(cli.CliError, match="No model is loaded"):
        cli.loaded_model(cli.running())  # /v1/models lists the downloaded ones too
    assert cli.main(["load", "mid"]) == 0
    assert cli.loaded_model(cli.running()) == "mid-Q4_K_M.gguf"


def test_get_takes_the_best_quant_that_fits(monkeypatch):
    ggufs = [{"name": f"m-{q}.gguf", "parts": [f"m-{q}.gguf"], "size": n} for q, n in (("Q2_K", 1), ("Q4_K_M", 2), ("Q8_0", 4), ("F16", 8))]
    fits = {"m-Q4_K_M.gguf", "m-Q2_K.gguf"}
    asked = []

    class Client:
        def post(self, path, json):
            asked.append(json["filename"])
            return httpx.Response(200, json={"verdict": "fits" if json["filename"] in fits else "too_large", "total": 3e9})

    chosen, est = cli._pick_quant(Client(), "acme/m-GGUF", ggufs)
    assert chosen["name"] == "m-Q4_K_M.gguf" and est["verdict"] == "fits"
    assert asked == ["m-Q8_0.gguf", "m-Q4_K_M.gguf"]  # best first; F16 isn't a quant it offers
    fits.clear()
    assert cli._pick_quant(Client(), "acme/m-GGUF", ggufs) == (ggufs[0], None)  # nothing fits: the smallest


def test_import_points_at_a_link_not_its_blob(engine, tmp_path, capsys):
    from tests.test_library import gguf

    blob = tmp_path / "blobs" / "2e80ce"
    blob.parent.mkdir()
    gguf(blob, "Tiny")
    link = tmp_path / "snap" / "tiny-Q4_K_M.gguf"
    link.parent.mkdir()
    link.symlink_to(blob)
    assert cli.main(["import", str(link)]) == 0
    assert "`temper load tiny-Q4_K_M.gguf` loads it" in capsys.readouterr().out


def _stream_client(chunks, seen):
    """A client whose /v1/chat/completions streams `chunks` and records each request body."""

    def handler(request):
        seen.append(json.loads(request.content))
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    return httpx.Client(base_url="http://model", transport=httpx.MockTransport(handler))


def test_a_reply_counts_its_tokens_and_hides_or_shows_reasoning(capsys):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "12 and 3..."}}]},
        {"choices": [{"delta": {"content": "36"}}]},
        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9}},
    ]
    seen = []
    text, reasoning, stats = cli._reply(_stream_client(chunks, seen), [{"role": "user", "content": "12*3?"}], model="tiny")
    assert (text, reasoning) == ("36", "12 and 3...") and capsys.readouterr().out == "36\n"
    assert stats["prompt_tokens"] == 20 and stats["completion_tokens"] == 9 and stats["first_token_seconds"] is not None
    assert seen[0]["model"] == "tiny" and seen[0]["stream_options"] == {"include_usage": True} and "chat_template_kwargs" not in seen[0]
    cli._reply(_stream_client(chunks, seen), [], think="on")
    assert "12 and 3..." in capsys.readouterr().out
    cli._reply(_stream_client(chunks, seen), [], think="off")
    assert seen[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_chat_repl_reads_blocks_and_runs_commands(monkeypatch, capsys):
    typed = iter(['"""', "first line", 'second line"""', "/system Be brief.", "/think sideways", "/think off", "//etc is a path", "/stats", "/nope", "/bye"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
    seen = []
    chunks = [{"choices": [{"delta": {"content": "ok"}}]}, {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}]
    repl = cli.Repl(_stream_client(chunks, seen), None)
    assert repl.run() == 0
    out = capsys.readouterr().out
    assert seen[0]["messages"] == [{"role": "user", "content": "first line\nsecond line"}]
    assert seen[1]["messages"][0] == {"role": "system", "content": "Be brief."} and seen[1]["messages"][-1]["content"] == "/etc is a path"
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "/think on, off or auto (now auto)" in out and "1 tokens" in out and "There's no /nope" in out


def test_save_keeps_the_conversation_in_tempers_chats(engine):
    repl = cli.Repl(engine, "Be brief.")
    repl.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    repl.save("from the terminal")
    chat = engine.get(f"/api/chats/{repl.chat_id}").json()
    assert chat["title"] == "from the terminal" and [m["content"] for m in chat["messages"]] == ["Be brief.", "hi", "hello"]
    repl.history += [{"role": "user", "content": "again"}, {"role": "assistant", "content": "sure"}]
    repl.save("")
    assert len(engine.get(f"/api/chats/{repl.chat_id}").json()["messages"]) == 5  # the same chat, brought up to date
