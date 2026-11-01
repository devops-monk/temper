"""Accounts, for a Temper served to several people (`temper serve`): an admin and the people they add.

Off until the first admin is added (Settings → Accounts, or `temper accounts add NAME --admin`). Then the browser
interface asks for a name and password instead of the engine's key, and each person signs in to their own chats and
jobs. Admins see the computer owner's chats, the ones the desktop app shows, and everything else; members can chat
with the models and run their own jobs, and nothing else: no settings, keys, files or tools that run on this computer.
The engine's own key (the desktop app's) is still accepted, as the owner.

Passwords are kept as scrypt hashes in users.json, and signed-in sessions as SHA-256 hashes of their tokens in
sessions.json, both readable only by this user. `temper accounts reset-password NAME` sets a forgotten one.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import home

router = APIRouter(prefix="/api/accounts", tags=["accounts"])

SESSION_DAYS = 30
MIN_PASSWORD = 8
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}")
# Who the request is from: the signed-in account, or None for the engine's own key (the owner).
current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("account", default=None)
_lock = threading.Lock()
_failures: dict[str, list[float]] = {}  # name: recent failed sign-ins, to slow down guessing

# What a member may reach: chatting, their own chats and jobs, and what the chat page reads to draw itself.
MEMBER_READ = ("/api/accounts/me", "/api/system", "/api/runtime", "/api/jobs", "/api/chats", "/api/chat", "/api/attachments",
               "/api/presets", "/api/prompts", "/api/compare", "/api/stats", "/api/catalog", "/v1")
MEMBER_WRITE = ("/api/accounts/me", "/api/accounts/logout", "/api/chats", "/api/chat", "/api/attachments", "/api/compare",
                "/api/runtime/load", "/api/runtime/unload", "/api/jobs", "/v1")
# Within those, what reads or writes this computer's files, or the owner's folders.
MEMBER_DENIED = re.compile(r"^/api/chats/(import|folders)(/|$)|^/api/chats/[^/]+/export$")


def _users_file():
    return home() / "users.json"


def _sessions_file():
    return home() / "sessions.json"


def _read(path, empty):
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, type(empty)) else empty
    except (OSError, ValueError):
        return empty


def _write(path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    if os.name != "nt":
        tmp.chmod(0o600)
    tmp.replace(path)


def users() -> list[dict]:
    return _read(_users_file(), [])


def enabled() -> bool:
    return any(u.get("admin") for u in users())


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


def _check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"Use a password of at least {MIN_PASSWORD} characters.")


def public(user: dict) -> dict:
    return {"name": user["name"], "admin": bool(user.get("admin")), "created": user.get("created")}


# --- Changing accounts (also from the CLI, which works on the files directly) -------------------------


def add(name: str, password: str, admin: bool = False) -> dict:
    name = name.strip().lower()
    if not NAME.fullmatch(name):
        raise ValueError("A name can use lowercase letters, digits, dots, dashes and underscores, up to 32 characters.")
    _check_password(password)
    with _lock:
        everyone = users()
        if any(u["name"] == name for u in everyone):
            raise ValueError(f"There's an account called {name} already.")
        salt = secrets.token_bytes(16)
        user = {"name": name, "salt": salt.hex(), "hash": _hash(password, salt), "admin": admin, "created": time.time()}
        _write(_users_file(), [*everyone, user])
    return public(user)


def set_password(name: str, password: str) -> None:
    """Sets a new password and signs the account out everywhere."""
    _check_password(password)
    with _lock:
        everyone = users()
        user = next((u for u in everyone if u["name"] == name), None)
        if not user:
            raise KeyError(name)
        salt = secrets.token_bytes(16)
        user["salt"], user["hash"] = salt.hex(), _hash(password, salt)
        _write(_users_file(), everyone)
        _drop_sessions(name)


def remove(name: str) -> None:
    with _lock:
        everyone = users()
        user = next((u for u in everyone if u["name"] == name), None)
        if not user:
            raise KeyError(name)
        if user.get("admin") and sum(1 for u in everyone if u.get("admin")) == 1 and len(everyone) > 1:
            raise ValueError("Remove the other accounts first, or make another admin: this is the only one.")
        _write(_users_file(), [u for u in everyone if u["name"] != name])
        _drop_sessions(name)


def _drop_sessions(name: str) -> None:
    _write(_sessions_file(), {k: s for k, s in _read(_sessions_file(), {}).items() if s.get("user") != name})


# --- Signing in -------------------------------------------------------------------------------------


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def verify(name: str, password: str) -> dict | None:
    user = next((u for u in users() if u["name"] == name.strip().lower()), None)
    if not user:
        _hash(password, b"0" * 16)  # as slow as a real check, so a name that doesn't exist doesn't answer faster
        return None
    return user if hmac.compare_digest(_hash(password, bytes.fromhex(user["salt"])), user["hash"]) else None


def sign_in(user: dict) -> str:
    token = "ts-" + secrets.token_urlsafe(32)
    with _lock:
        now = time.time()
        sessions = {k: s for k, s in _read(_sessions_file(), {}).items() if now - s.get("created", 0) < SESSION_DAYS * 86400}
        sessions[_digest(token)] = {"user": user["name"], "created": now}
        _write(_sessions_file(), sessions)
    return token


def session(token: str) -> dict | None:
    """The account a session token signs in, if it's current."""
    if not token.startswith("ts-"):
        return None
    found = _read(_sessions_file(), {}).get(_digest(token))
    if not found or time.time() - found.get("created", 0) > SESSION_DAYS * 86400:
        return None
    return next((u for u in users() if u["name"] == found["user"]), None)


def sign_out(token: str) -> None:
    with _lock:
        sessions = _read(_sessions_file(), {})
        if sessions.pop(_digest(token), None):
            _write(_sessions_file(), sessions)


def member_may(method: str, path: str) -> bool:
    allowed = MEMBER_READ if method in ("GET", "HEAD") else MEMBER_WRITE
    if MEMBER_DENIED.match(path) and not (method == "GET" and path.endswith("/export")):  # the export's text is fine
        return False
    return any(path == p or path.startswith(p + "/") for p in allowed)


# --- Who's asking -----------------------------------------------------------------------------------


def owner() -> str | None:
    """Whose chats and jobs a request sees: a member's own, or (None) the owner's, for the engine's key and admins."""
    account = current.get()
    return None if account is None or account.get("admin") else account["name"]


def is_admin() -> bool:
    account = current.get()
    return account is None or bool(account.get("admin"))


def require_admin() -> None:
    if not is_admin():
        raise HTTPException(403, "Only an admin can do that.")


# --- The API ----------------------------------------------------------------------------------------


class SignIn(BaseModel):
    name: str
    password: str


class NewAccount(BaseModel):
    name: str
    password: str = Field(min_length=1)
    admin: bool = False


class NewPassword(BaseModel):
    password: str
    current: str | None = None  # needed to change your own


@router.get("/status")
def status():
    """Open to anyone: whether the interface should ask for a name and password."""
    return {"enabled": enabled()}


@router.post("/login")
def login(req: SignIn):
    key = req.name.strip().lower()
    now = time.time()
    with _lock:
        recent = [t for t in _failures.get(key, []) if now - t < 300]
        _failures[key] = recent
    if len(recent) >= 10:
        raise HTTPException(429, "Too many wrong passwords. Wait a few minutes and try again.")
    user = verify(req.name, req.password)
    if not user:
        with _lock:
            _failures.setdefault(key, []).append(now)
        raise HTTPException(401, "Wrong name or password.")
    with _lock:
        _failures.pop(key, None)
    return {"token": sign_in(user), "account": public(user)}


@router.post("/logout")
def logout(request: Request):
    sign_out(request.headers.get("authorization", "").removeprefix("Bearer ").strip())
    return {"ok": True}


@router.get("/me")
def me():
    account = current.get()
    return {"account": public(account) if account else None, "admin": is_admin(), "enabled": enabled()}


@router.put("/me/password")
def change_own_password(req: NewPassword):
    account = current.get()
    if account is None:
        raise HTTPException(400, "The engine's key isn't an account. Sign in to change your password.")
    if not req.current or not verify(account["name"], req.current):
        raise HTTPException(403, "Your current password isn't right.")
    try:
        set_password(account["name"], req.password)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.get("")
def list_accounts():
    require_admin()
    return {"accounts": [public(u) for u in users()], "enabled": enabled()}


@router.post("")
def create(req: NewAccount):
    require_admin()
    if not enabled() and not req.admin:
        raise HTTPException(400, "Add an admin first; the accounts after it can be members.")
    try:
        return add(req.name, req.password, req.admin)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.put("/{name}/password")
def reset_password(name: str, req: NewPassword):
    require_admin()
    try:
        set_password(name, req.password)
    except KeyError as e:
        raise HTTPException(404, "No such account.") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.delete("/{name}")
def delete(name: str):
    require_admin()
    account = current.get()
    if account and account["name"] == name:
        raise HTTPException(400, "You can't remove your own account while signed in to it.")
    try:
        remove(name)
    except KeyError as e:
        raise HTTPException(404, "No such account.") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return list_accounts()
