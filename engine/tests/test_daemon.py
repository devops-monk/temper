import json
import plistlib
import sys

import pytest

from temper_engine import daemon, home


@pytest.fixture
def managers(tmp_path, monkeypatch):
    """A fake home folder and service managers that record what they were asked."""
    monkeypatch.setattr(daemon.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    calls = []

    def run(*cmd):
        calls.append(cmd)
        return daemon.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(daemon, "_run", run)
    return calls


def test_a_launchd_agent_runs_the_engine_at_login(managers, monkeypatch):
    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    where = daemon.install()
    plist = plistlib.loads(daemon._plist().read_bytes())
    assert where == str(daemon._plist()) and plist["Label"] == "app.temper.engine" and plist["RunAtLoad"]
    assert plist["ProgramArguments"][1:] == ["-m", "temper_engine", "serve", "--host", "127.0.0.1", "--port", "8765"]
    assert plist["EnvironmentVariables"] == {"TEMPER_HOME": str(home())}
    assert managers[-1][:2] == ("launchctl", "bootstrap") and daemon.login_item()
    assert daemon.uninstall() and not daemon._plist().exists() and managers[-1][:2] == ("launchctl", "bootout")
    assert not daemon.uninstall()  # nothing left to undo


def test_a_systemd_user_unit_on_linux(managers, monkeypatch):
    monkeypatch.setattr(daemon.sys, "platform", "linux")
    daemon.install()
    unit = daemon._unit().read_text()
    assert "ExecStart=" in unit and "temper_engine serve" in unit and f'TEMPER_HOME={home()}' in unit
    assert "Restart=on-abnormal" in unit  # a clean exit (the app's engine was running) isn't retried
    assert ("systemctl", "--user", "enable", "--now", "temper-engine.service") in managers
    assert daemon.uninstall() and not daemon._unit().exists()


def test_a_failing_service_manager_says_why(managers, monkeypatch):
    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    monkeypatch.setattr(daemon, "_run", lambda *cmd: daemon.subprocess.CompletedProcess(cmd, 5, "", "Bootstrap failed: 5: Input/output error"))
    with pytest.raises(RuntimeError, match="Input/output error"):
        daemon.install()


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in engine is a POSIX sleep")
def test_up_and_down_start_and_stop_a_detached_process(monkeypatch):
    # A stand-in for the engine: a process whose command line names temper_engine, as the real one's does.
    monkeypatch.setattr(daemon, "command", lambda host, port: [sys.executable, "-c", "import time; time.sleep(60)", "temper_engine"])
    started = daemon.start()
    assert daemon.pid() == started and daemon.log_file().exists()
    (home() / "engine.json").write_text(json.dumps({"pid": started, "url": "http://127.0.0.1:8765"}))
    assert daemon.stop()
    assert daemon.pid() is None and not (home() / "engine.json").exists()  # the file its clean exit would have removed
    assert not daemon.stop()


def test_a_stale_pid_file_is_not_a_running_engine():
    (home() / "daemon.pid").write_text("999999")
    assert daemon.pid() is None
    (home() / "daemon.pid").write_text(str(__import__("os").getpid()))  # alive, but not an engine
    assert daemon.pid() is None


def test_settings_turn_the_login_item_on_and_off(managers, monkeypatch):
    from fastapi.testclient import TestClient

    from temper_engine.app import create_app

    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    c = TestClient(create_app(token="t"), headers={"Authorization": "Bearer t"})
    assert c.get("/api/daemon").json() == {"login": False, "pid": None}
    assert c.put("/api/daemon/login", json={"enabled": True}).json()["login"]
    assert not c.put("/api/daemon/login", json={"enabled": False}).json()["login"]
