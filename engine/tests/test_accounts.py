import json
import os

import pytest
from fastapi.testclient import TestClient

from temper_engine import accounts, cli, home, jobs
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}


@pytest.fixture
def app():
    with TestClient(create_app(token="t")) as c:
        yield c


def sign_in(c, name, password):
    r = c.post("/api/accounts/login", json={"name": name, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_accounts_are_off_until_an_admin_is_added(app):
    assert app.get("/api/accounts/status").json() == {"enabled": False}
    assert app.post("/api/accounts", headers=H, json={"name": "bob", "password": "password1"}).status_code == 400  # an admin first
    assert app.post("/api/accounts", headers=H, json={"name": "ann", "password": "short", "admin": True}).status_code == 400
    assert app.post("/api/accounts", headers=H, json={"name": "Ann!", "password": "password1", "admin": True}).status_code == 400
    assert app.post("/api/accounts", headers=H, json={"name": "ann", "password": "password1", "admin": True}).json()["admin"]
    assert app.get("/api/accounts/status").json() == {"enabled": True}
    stored = (home() / "users.json").read_text()
    assert "password1" not in stored and json.loads(stored)[0]["hash"]
    if os.name != "nt":
        assert (home() / "users.json").stat().st_mode & 0o777 == 0o600


def test_signing_in_and_out(app):
    accounts.add("ann", "password1", admin=True)
    assert app.post("/api/accounts/login", json={"name": "ann", "password": "wrong-one"}).status_code == 401
    assert app.post("/api/accounts/login", json={"name": "nobody", "password": "password1"}).status_code == 401
    ann = sign_in(app, "Ann", "password1")
    assert app.get("/api/accounts/me", headers=ann).json()["account"]["name"] == "ann"
    assert app.get("/api/keys", headers=ann).status_code == 200  # an admin reaches everything
    assert "ts-" not in (home() / "sessions.json").read_text()  # only the token's hash is kept
    app.post("/api/accounts/logout", headers=ann)
    assert app.get("/api/accounts/me", headers=ann).status_code == 401


def test_guessing_is_slowed_down(app):
    accounts.add("ann", "password1", admin=True)
    accounts._failures.clear()
    for _ in range(10):
        app.post("/api/accounts/login", json={"name": "ann", "password": "wrong-one"})
    assert app.post("/api/accounts/login", json={"name": "ann", "password": "password1"}).status_code == 429
    accounts._failures.clear()


def test_members_have_their_own_chats_and_jobs_and_nothing_else(app):
    accounts.add("ann", "password1", admin=True)
    accounts.add("bob", "password2")
    bob = sign_in(app, "bob", "password2")
    owners = app.post("/api/chats", headers=H, json={"title": "the owner's"}).json()
    bobs = app.post("/api/chats", headers=bob, json={"title": "bob's"}).json()
    assert [c["title"] for c in app.get("/api/chats", headers=bob).json()["chats"]] == ["bob's"]
    assert "bob's" not in [c["title"] for c in app.get("/api/chats", headers=H).json()["chats"]]
    assert [c["title"] for c in app.get("/api/chats", headers=sign_in(app, "ann", "password1")).json()["chats"]] == ["the owner's"]
    assert app.get(f"/api/chats/{owners['id']}", headers=bob).status_code == 404
    assert app.delete(f"/api/chats/{owners['id']}", headers=bob).status_code == 404
    assert app.put(f"/api/chats/{bobs['id']}/messages", headers=bob, json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 200
    # Nothing that reads this computer's files or changes its settings.
    for method, path in (("GET", "/api/keys"), ("GET", "/api/files"), ("POST", "/api/train/start"), ("POST", f"/api/chats/{bobs['id']}/export"), ("POST", "/api/chats/folders/add"), ("GET", "/api/accounts"), ("POST", "/mcp")):
        assert app.request(method, path, headers=bob).status_code == 403, path
    assert app.post("/api/chat/generations", headers=bob, json={"body": {"messages": []}, "tools": ["python"]}).status_code == 403
    # Jobs: a member's own only.
    mine = jobs.registry.create("test", "the owner's job")
    assert mine.id not in [j["id"] for j in app.get("/api/jobs", headers=bob).json()]
    assert app.get(f"/api/jobs/{mine.id}", headers=bob).status_code == 404
    assert app.get(f"/api/jobs/{mine.id}", headers=H).status_code == 200
    jobs.registry.jobs.pop(mine.id)


def test_passwords_change_and_reset(app, monkeypatch, capsys):
    accounts.add("ann", "password1", admin=True)
    accounts.add("bob", "password2")
    bob = sign_in(app, "bob", "password2")
    assert app.put("/api/accounts/me/password", headers=bob, json={"current": "nope", "password": "password3"}).status_code == 403
    assert app.put("/api/accounts/me/password", headers=bob, json={"current": "password2", "password": "password3"}).status_code == 200
    assert app.get("/api/accounts/me", headers=bob).status_code == 401  # signed out everywhere
    # Reset from the CLI, for a forgotten one.
    answers = iter(["password4", "password4"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert cli.main(["accounts", "reset-password", "bob"]) == 0
    sign_in(app, "bob", "password4")
    assert cli.main(["accounts"]) == 0
    assert "bob" in capsys.readouterr().out
    # The only admin stays while there are others.
    assert app.delete("/api/accounts/ann", headers=H).status_code == 400
    assert [a["name"] for a in app.delete("/api/accounts/bob", headers=H).json()["accounts"]] == ["ann"]
