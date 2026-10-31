import contextlib
import json
import sys

import httpx
import pytest

from temper_engine import cli, daemon, lifecycle


class FakeEngine:
    """Answers each question with the product it asks for, except when the model is "wrong"."""

    def __init__(self):
        self.bodies = []

    def post(self, path, json):
        self.bodies.append(json)
        a, b = [int(x) for x in json["messages"][-1]["content"].split("What is ")[1].rstrip("?").split(" * ")]
        answer = a * b + (1 if json.get("model") == "wrong" else 0)
        return httpx.Response(200, json={"model": "default_model", "choices": [{"message": {"content": f"So it's {answer}."}}]})


def test_eval_scores_answers_and_compares_models(tmp_path, monkeypatch, capsys):
    data = tmp_path / "q.jsonl"
    data.write_text("".join(json.dumps({"question": f"What is {a} * 3?", "answer": str(a * 3)}) + "\n" for a in range(1, 5)) + json.dumps({"question": "no answer"}) + "\n")
    fake = FakeEngine()
    monkeypatch.setattr(cli, "running", lambda: {"url": "http://x", "token": "t", "pid": 1})
    monkeypatch.setattr(cli, "_client", lambda engine: contextlib.nullcontext(fake))
    assert cli.main(["eval", str(data), "--model", "good", "--compare", "wrong", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [(s["model"], s["right"], s["asked"]) for s in out["summary"]] == [("good", 4, 4), ("wrong", 0, 4)]
    assert all(b["temperature"] == 0 and b["chat_template_kwargs"] == {"enable_thinking": False} for b in fake.bodies)
    assert cli.main(["eval", str(data), "--model", "wrong", "--limit", "2"]) == 0
    printed = capsys.readouterr().out
    assert "0/2" in printed and "expected '3', answered '4'" in printed
    empty = tmp_path / "e.jsonl"
    empty.write_text(json.dumps({"question": "hi"}) + "\n")
    assert cli.main(["eval", str(empty)]) != 0 and "expected answer" in capsys.readouterr().err


@pytest.fixture
def mac(tmp_path, monkeypatch):
    """A Mac user folder with Temper's home and app folders in it, and no engine or login item."""
    user = tmp_path / "Users" / "someone"
    support = user / "Library" / "Application Support" / lifecycle.IDENTIFIER
    (support / "runs").mkdir(parents=True)
    (support / "runs" / "x.json").write_text("{}")
    (user / "Library" / "Caches" / lifecycle.IDENTIFIER).mkdir(parents=True)
    monkeypatch.setattr(lifecycle.Path, "home", lambda: user)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("TEMPER_HOME", str(support))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(cli, "running", lambda: (_ for _ in ()).throw(cli.CliError("not running")))
    monkeypatch.setattr(daemon, "login_item", lambda: False)
    monkeypatch.setattr(daemon, "pid", lambda: None)
    monkeypatch.setattr(lifecycle, "app_bundle", lambda: None)
    return user, support


def test_uninstall_lists_then_removes_temper_and_nothing_else(mac, tmp_path, capsys):
    user, support = mac
    (tmp_path / "hf").mkdir()
    (user / "keep.txt").write_text("mine")
    assert cli.main(["uninstall", "--dry-run"]) == 0
    listed = capsys.readouterr().out
    assert str(support) in listed and "Caches" in listed and "stay in" in listed and support.exists()
    assert cli.main(["uninstall"]) != 0 and "--yes" in capsys.readouterr().err  # nobody at the terminal to ask
    assert cli.main(["uninstall", "--yes"]) == 0
    assert not support.exists() and not (user / "Library" / "Caches" / lifecycle.IDENTIFIER).exists()
    assert (user / "keep.txt").read_text() == "mine" and (tmp_path / "hf").exists()


def test_uninstall_keeps_data_when_asked_and_never_removes_the_home_folder(mac, monkeypatch):
    user, support = mac
    assert lifecycle.plan(keep_data=True) == []
    monkeypatch.setenv("TEMPER_HOME", str(user))  # a mistake that must not wipe the user's files
    assert user not in lifecycle.plan() and not lifecycle.safe(user) and not lifecycle.safe(user.parent)


def test_update_says_whether_theres_a_newer_release(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "latest", lambda: {"version": "9.0.0", "notes": "", "url": "https://example.com/r"})
    assert cli.main(["update", "--check"]) == 1 and "Temper 9.0.0 is out" in capsys.readouterr().out
    monkeypatch.setattr(cli, "running", lambda: (_ for _ in ()).throw(cli.CliError("not running")))
    ran = []
    monkeypatch.setattr(lifecycle, "run_installer", lambda: ran.append(1) or 0)
    assert cli.main(["update", "--yes"]) == 0 and ran == [1]
    monkeypatch.setattr(lifecycle, "latest", lambda: {"version": cli.__version__, "notes": "", "url": ""})
    assert cli.main(["update"]) == 0 and "newest release" in capsys.readouterr().out
    assert lifecycle.newer("0.10.0", "0.9.1") and not lifecycle.newer("0.1.0", "0.1.0")
