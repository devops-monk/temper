import sys
from pathlib import Path

import pytest

from temper_engine import sandbox, tools
from temper_engine.tools import Call

seatbelt = pytest.mark.skipif(sandbox.kind() != "seatbelt", reason="needs macOS's sandbox-exec")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPER_HOME", str(tmp_path / "home"))
    return tmp_path


def test_code_runs_in_the_chats_own_folder_which_it_keeps():
    out = sandbox.run("python", "open('notes.txt', 'w').write('kept'); print(6 * 7)", "chat1")
    assert out.startswith("42") and "Files written: notes.txt" in out
    assert sandbox.run("python", "print(open('notes.txt').read())", "chat1").startswith("kept")
    assert "FileNotFoundError" in sandbox.run("python", "print(open('notes.txt').read())", "chat2")
    sandbox.forget("chat1")
    assert not (sandbox.root() / "chats" / "chat1").exists()


def test_errors_and_exit_codes_come_back():
    out = sandbox.run("python", "import sys; print('partial'); sys.exit(3)", None)
    assert out.startswith("partial") and "(Exit code 3.)" in out
    assert "ZeroDivisionError" in sandbox.run("python", "1/0", None)
    assert sandbox.run("python", "x = 1", None).startswith("(It printed nothing")


def test_a_run_that_takes_too_long_is_stopped(monkeypatch):
    monkeypatch.setattr(sandbox, "TIMEOUT", 1)
    assert "Stopped: it ran longer than 1 seconds" in sandbox.run("python", "import time\nwhile True: time.sleep(0.1)", None)


def test_the_environment_is_stripped(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    out = sandbox.run("python", "import os; print(os.environ.get('SECRET_TOKEN'), os.environ['HOME'] == os.getcwd())", None)
    assert out.startswith("None True")


@seatbelt
def test_seatbelt_blocks_the_network_writes_elsewhere_and_secrets(tmp_path, monkeypatch):
    user = tmp_path / "user"
    (user / ".ssh").mkdir(parents=True)
    (user / ".ssh" / "id_ed25519").write_text("key")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
    code = f"""
import socket
for what, attempt in [
    ("network", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3)),
    ("write", lambda: open({str(tmp_path / "outside.txt")!r}, "w")),
    ("secret", lambda: open({str(user / ".ssh" / "id_ed25519")!r}).read()),
]:
    try:
        attempt()
        print(what, "allowed")
    except OSError:
        print(what, "blocked")
"""
    assert sandbox.run("python", code, None).splitlines()[:3] == ["network blocked", "write blocked", "secret blocked"]


def test_the_python_tool_runs_code_for_the_chat():
    step = tools.run(Call("run_python", {"code": "print(sum(range(10)))"}), ["run_python"], {"chat": "c9"})
    assert step["result"].startswith("45") and not step["error"]
    assert "Give the code" in tools.run(Call("run_python", {"code": ""}), ["run_python"])["result"]


def test_code_tools_ask_first_and_javascript_needs_node(monkeypatch):
    assert tools.TOOLS["run_python"].mode == "ask" and tools.TOOLS["calculator"].mode == "auto"
    monkeypatch.setattr(sandbox, "node", lambda: None)
    assert "run_javascript" not in [t.name for t in tools.available()]
    assert "Node.js isn't installed" in tools.run(Call("run_javascript", {"code": "1"}), ["run_javascript"])["result"]


@pytest.mark.skipif(sys.platform == "win32", reason="limits use POSIX rlimits")
def test_files_are_capped_in_size(monkeypatch):
    monkeypatch.setattr(sandbox, "MAX_FILE", 1000)
    out = sandbox.run("python", "open('big', 'wb').write(b'x' * 50000)", None)
    assert "File too large" in out or "Exit code" in out
