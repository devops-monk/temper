"""Model Context Protocol servers whose tools chat models can call.

Servers are kept in mcp.json in Temper's home, in the format other apps use, so a config can be
pasted in as it is: {"mcpServers": {"<name>": {"command", "args", "env", "cwd"} or {"url", "headers",
"type": "http" | "sse"}}}. Temper adds "enabled" and "tools": {<tool>: "off" | "ask" | "auto"}, each
tool's approval (ask, until the person chooses otherwise).

The client speaks MCP's JSON-RPC over stdio (a process Temper starts), Streamable HTTP, and the older
HTTP+SSE transport. A server is connected when its tools are first wanted and stays connected while
the engine runs. Requests the server makes back (sampling, roots, elicitation) are declined; pings
are answered.

A chat turns on whole servers; their tools join the built-in ones as mcp__<server>__<tool>. When the
tools turned on would crowd the context (more than CATALOG_LIMIT), the model gets a search_tools
tool instead, BM25 over the tools' names and descriptions, and the tools it finds become callable.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
from collections import Counter
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import __version__, home

router = APIRouter(prefix="/api/mcp", tags=["chat"])

PROTOCOL = "2025-06-18"
TIMEOUT = 60  # seconds for a request
CONNECT_TIMEOUT = 30
CATALOG_LIMIT = 24  # more tools than this are found by search rather than all sent
MAX_RESULT = 16_000
MASK = "••••••"


def config_file() -> Path:
    return home() / "mcp.json"


def load() -> dict[str, dict]:
    try:
        data = json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    servers = data.get("mcpServers", data) if isinstance(data, dict) else {}
    return {k: v for k, v in servers.items() if isinstance(v, dict)}


def save(servers: dict[str, dict]) -> None:
    path = config_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def transport(cfg: dict) -> str:
    if cfg.get("command"):
        return "stdio"
    kind = str(cfg.get("type") or cfg.get("transport") or "").lower()
    if kind in ("sse",):
        return "sse"
    return "http"


class McpError(Exception):
    pass


# --- Connections ----------------------------------------------------------------------------------


class Connection:
    """One server: sends JSON-RPC requests and matches the answers to them."""

    def __init__(self, name: str, cfg: dict, client: httpx.Client | None = None) -> None:
        self.name, self.cfg = name, cfg
        self.kind = transport(cfg)
        self.pending: dict[int, Future] = {}
        self.lock = threading.Lock()
        self.next_id = 0
        self.proc: subprocess.Popen | None = None
        self.http = client or httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10), follow_redirects=True)
        self.session: str | None = None
        self.endpoint: str | None = None  # where the older SSE transport takes requests
        self.stderr: list[str] = []
        self.tools: list[dict] = []
        self.info: dict = {}
        self.closed = False

    # Sending ---------------------------------------------------------------------------------

    def request(self, method: str, params: dict | None = None, timeout: float = TIMEOUT) -> dict:
        with self.lock:
            self.next_id += 1
            rid = self.next_id
        future: Future = Future()
        self.pending[rid] = future
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})})
            message = future.result(timeout=timeout)
        except FutureTimeout:
            raise McpError(f"{self.name} didn't answer {method} within {timeout:g} seconds.") from None
        finally:
            self.pending.pop(rid, None)
        if "error" in message:
            error = message["error"] or {}
            raise McpError(f"{self.name}: {error.get('message', 'error')}" + (f" ({error['data']})" if error.get("data") else ""))
        return message.get("result") or {}

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})

    def _send(self, message: dict) -> None:
        if self.closed:
            raise McpError(f"{self.name} is disconnected.")
        if self.kind == "stdio":
            assert self.proc and self.proc.stdin
            try:
                self.proc.stdin.write(json.dumps(message) + "\n")
                self.proc.stdin.flush()
            except OSError as e:
                raise McpError(f"{self.name} stopped: {self._last_words() or e}") from None
        elif self.kind == "sse":
            assert self.endpoint
            r = self.http.post(self.endpoint, json=message, headers=self._headers())
            if r.status_code >= 400:
                raise McpError(f"{self.name} answered {r.status_code}: {r.text[:200]}")
        else:
            self._post(message)

    def _headers(self) -> dict:
        headers = {str(k): str(v) for k, v in (self.cfg.get("headers") or {}).items()}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        if self.info:
            headers["MCP-Protocol-Version"] = self.info.get("protocolVersion", PROTOCOL)
        return headers

    def _post(self, message: dict) -> None:
        """Streamable HTTP: the answer comes back as JSON or as an event stream."""
        headers = {**self._headers(), "Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        with self.http.stream("POST", self.cfg["url"], json=message, headers=headers) as r:
            if sid := r.headers.get("mcp-session-id"):
                self.session = sid
            if r.status_code == 202 or "id" not in message and r.status_code < 300:
                return
            if r.status_code >= 400:
                body = r.read().decode(errors="replace")
                raise McpError(f"{self.name} answered {r.status_code}: {body[:200]}" + (" Check its headers (an API key?)." if r.status_code in (401, 403) else ""))
            if r.headers.get("content-type", "").startswith("text/event-stream"):
                for data in _sse(r.iter_lines()):
                    self._receive(data[1])
            else:
                self._receive(r.read().decode(errors="replace"))

    # Receiving -------------------------------------------------------------------------------

    def _receive(self, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            return
        for m in message if isinstance(message, list) else [message]:
            if not isinstance(m, dict):
                continue
            if "method" in m and "id" in m:  # the server asking something of the client
                threading.Thread(target=self._answer, args=(m,), daemon=True).start()
            elif "id" in m and (future := self.pending.get(m["id"])) and not future.done():
                future.set_result(m)

    def _answer(self, m: dict) -> None:
        reply = {"jsonrpc": "2.0", "id": m["id"]}
        reply |= {"result": {}} if m["method"] == "ping" else {"error": {"code": -32601, "message": "Temper doesn't support that."}}
        try:
            self._send(reply)
        except (McpError, httpx.HTTPError):
            pass

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            if line.strip():
                self._receive(line)
        self._fail(f"{self.name} stopped" + (f": {w}" if (w := self._last_words()) else "."))

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self.stderr = [*self.stderr[-40:], line.rstrip()]

    def _last_words(self) -> str:
        return " ".join(self.stderr[-3:])[-400:]

    def _read_events(self, response: httpx.Response) -> None:
        try:
            for event, data in _sse(response.iter_lines()):
                if event == "endpoint":
                    self.endpoint = urljoin(self.cfg["url"], data.strip())
                    self._endpoint_ready.set()
                else:
                    self._receive(data)
        except httpx.HTTPError:
            pass
        finally:
            response.close()
            self._fail(f"{self.name} closed its event stream.")

    def _fail(self, why: str) -> None:
        self.closed = True
        for future in list(self.pending.values()):
            if not future.done():
                future.set_result({"error": {"message": why}})

    # Lifecycle -------------------------------------------------------------------------------

    def start(self) -> None:
        if self.kind == "stdio":
            cmd = [str(self.cfg["command"]), *[str(a) for a in self.cfg.get("args") or []]]
            env = {**os.environ, **{str(k): str(v) for k, v in (self.cfg.get("env") or {}).items()}}
            try:
                self.proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, cwd=self.cfg.get("cwd") or None,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as e:
                raise McpError(f"Couldn't start {cmd[0]}: {e.strerror or e}. Is it installed and on PATH?") from None
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
        elif self.kind == "sse":
            self._endpoint_ready = threading.Event()
            try:
                response = self.http.send(self.http.build_request("GET", self.cfg["url"], headers={**self._headers(), "Accept": "text/event-stream"}), stream=True)
            except httpx.HTTPError as e:
                raise McpError(f"Couldn't reach {self.cfg['url']}: {e}") from None
            if response.status_code >= 400:
                response.close()
                raise McpError(f"{self.cfg['url']} answered {response.status_code}.")
            threading.Thread(target=self._read_events, args=(response,), daemon=True).start()
            if not self._endpoint_ready.wait(CONNECT_TIMEOUT):
                raise McpError(f"{self.name} didn't say where to send requests.")
        try:
            self.info = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "Temper", "version": __version__}}, timeout=CONNECT_TIMEOUT)
        except httpx.HTTPError as e:
            raise McpError(f"Couldn't reach {self.cfg.get('url')}: {e}") from None
        self.notify("notifications/initialized")
        self.refresh()

    def refresh(self) -> list[dict]:
        tools, cursor = [], None
        for _ in range(50):
            page = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += [t for t in page.get("tools") or [] if isinstance(t, dict) and t.get("name")]
            if not (cursor := page.get("nextCursor")):
                break
        self.tools = tools
        return tools

    def call(self, tool: str, arguments: dict) -> tuple[str, bool]:
        try:
            result = self.request("tools/call", {"name": tool, "arguments": arguments}, timeout=TIMEOUT * 5)
        except httpx.HTTPError as e:
            raise McpError(f"{self.name} stopped answering: {e}") from None
        return _as_text(result), bool(result.get("isError"))

    def close(self) -> None:
        self.closed = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin and self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
        if self.kind != "stdio" and self.session:
            try:
                self.http.delete(self.cfg["url"], headers=self._headers(), timeout=5)
            except httpx.HTTPError:
                pass


def _sse(lines):
    """(event, data) pairs from server-sent event lines."""
    event, data = "message", []
    for line in lines:
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)


def _as_text(result: dict) -> str:
    parts = []
    for item in result.get("content") or []:
        kind = item.get("type")
        if kind == "text":
            parts.append(item.get("text", ""))
        elif kind == "resource" and isinstance(item.get("resource"), dict):
            res = item["resource"]
            parts.append(res.get("text") or f"[a resource: {res.get('uri', '')}]")
        elif kind == "resource_link":
            parts.append(f"[{item.get('name') or 'a resource'}: {item.get('uri', '')}]")
        elif kind in ("image", "audio"):
            parts.append(f"[{'an image' if kind == 'image' else 'a sound'} ({item.get('mimeType', '')}), which Temper can't pass on yet]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(p for p in parts if p) or "(The tool returned nothing.)"
    return text if len(text) <= MAX_RESULT else text[:MAX_RESULT] + "\n[The rest of the result was cut.]"


class Manager:
    """The connected servers, by name; connecting is lazy and one at a time per server."""

    def __init__(self) -> None:
        self.connections: dict[str, Connection] = {}
        self.errors: dict[str, str] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.client: httpx.Client | None = None  # tests put a mock transport here

    def connection(self, name: str) -> Connection:
        cfg = load().get(name)
        if cfg is None:
            raise McpError(f"There's no MCP server called {name}.")
        with self.locks.setdefault(name, threading.Lock()):
            conn = self.connections.get(name)
            if conn and not conn.closed and conn.cfg == cfg:
                return conn
            if conn:
                conn.close()
            conn = Connection(name, cfg, self.client)
            try:
                conn.start()
            except McpError as e:
                conn.close()
                self.errors[name] = str(e)
                self.connections.pop(name, None)
                raise
            self.errors.pop(name, None)
            self.connections[name] = conn
            return conn

    def tools(self, name: str) -> list[dict]:
        return self.connection(name).tools

    def disconnect(self, name: str) -> None:
        if conn := self.connections.pop(name, None):
            conn.close()
        self.errors.pop(name, None)

    def close_all(self) -> None:
        for name in list(self.connections):
            self.disconnect(name)


manager = Manager()


# --- As chat tools --------------------------------------------------------------------------------

PREFIX = "mcp__"
SEARCH = "search_tools"


def tool_name(server: str, tool: str) -> str:
    """The name a model calls: mcp__<server>__<tool>, in the characters and length tool names allow."""
    clean = lambda s: re.sub(r"[^A-Za-z0-9_-]", "_", s)  # noqa: E731
    return f"{PREFIX}{clean(server)}__{clean(tool)}"[:64]


def split(name: str) -> tuple[str, str] | None:
    """The server and tool a mcp__ name stands for."""
    if not name.startswith(PREFIX):
        return None
    for server, cfg in load().items():
        conn = manager.connections.get(server)
        for t in conn.tools if conn else []:
            if tool_name(server, t["name"]) == name:
                return server, t["name"]
    return None


def mode(server: str, tool: str) -> str:
    return str((load().get(server, {}).get("tools") or {}).get(tool, "ask"))


def catalog(servers: list[str]) -> list[dict]:
    """The tools a chat's servers offer, turned on in their settings: [{name, server, tool, description, parameters, mode}].
    Servers that can't be reached are left out (their error shows in Settings)."""
    out = []
    configs = load()
    for server in servers:
        if server not in configs or configs[server].get("enabled") is False:
            continue
        try:
            tools = manager.tools(server)
        except McpError:
            continue
        for t in tools:
            if (m := mode(server, t["name"])) == "off":
                continue
            out.append({"name": tool_name(server, t["name"]), "server": server, "tool": t["name"], "description": t.get("description") or t.get("title") or "", "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}, "mode": m})
    return out


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    # snake_case and camelCase names split into words
    return _WORD.findall(re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ").lower())


def search(entries: list[dict], query: str, limit: int = 6) -> list[dict]:
    """The entries that best match `query`, by Okapi BM25 over name, description and parameter names."""
    docs = [_words(" ".join([e["tool"], e["server"], e["description"], *(e["parameters"].get("properties") or {}).keys()])) for e in entries]
    if not docs:
        return []
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(w for d in docs for w in set(d))
    terms = _words(query)
    k1, b = 1.5, 0.75

    def score(doc: list[str]) -> float:
        tf = Counter(doc)
        return sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg)) for t in terms if t in tf)

    ranked = sorted(((score(d), i) for i, d in enumerate(docs)), key=lambda x: (-x[0], x[1]))
    return [entries[i] for s, i in ranked[:limit] if s > 0]


def search_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": SEARCH,
            "description": "Finds tools for a task among the many available. Search first, then call the tools it returns by their names.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "What you want to do, in a few words"}}, "required": ["query"]},
        },
    }


# --- API ------------------------------------------------------------------------------------------


def _masked(cfg: dict) -> dict:
    """The config for the app to show: values of env and headers hidden, since they're often keys."""
    return {**cfg, **{k: {n: MASK for n in cfg[k]} for k in ("env", "headers") if isinstance(cfg.get(k), dict)}}


def _summary(name: str, cfg: dict) -> dict:
    conn = manager.connections.get(name)
    tools = [{"name": t["name"], "description": t.get("description") or "", "mode": mode(name, t["name"])} for t in (conn.tools if conn else [])]
    state = "off" if cfg.get("enabled") is False else "connected" if conn and not conn.closed else "error" if name in manager.errors else "idle"
    return {"name": name, "transport": transport(cfg), "config": _masked(cfg), "enabled": cfg.get("enabled") is not False, "state": state, "error": manager.errors.get(name), "tools": tools, "server": (conn.info.get("serverInfo") if conn else None)}


@router.get("")
def list_servers():
    return {"servers": [_summary(n, c) for n, c in load().items()]}


class ServerConfig(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[\w .-]+$")
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    cwd: str | None = None
    url: str | None = None
    type: str | None = None
    headers: dict[str, str] = {}


def _clean(req: ServerConfig, old: dict) -> dict:
    if req.command and not req.args and " " in req.command.strip():  # a whole command line typed into one box
        req.command, req.args = parse_command(req.command)
    if not req.command and not req.url:
        raise HTTPException(422, "Give a command to run, or the server's URL.")
    if req.url and not re.match(r"^https?://", req.url):
        raise HTTPException(422, "The URL should start with http:// or https://.")
    # Hidden values the app sent back unchanged keep what was saved.
    keep = lambda k, new: {n: (old.get(k) or {}).get(n, "") if v == MASK else v for n, v in new.items()}  # noqa: E731
    cfg = {"command": req.command, "args": req.args, "env": keep("env", req.env), "cwd": req.cwd} if req.command else {"url": req.url, "type": req.type or "http", "headers": keep("headers", req.headers)}
    cfg = {k: v for k, v in cfg.items() if v not in (None, [], {})}
    return {**cfg, **{k: old[k] for k in ("enabled", "tools") if k in old}}


@router.post("")
def add_server(req: ServerConfig):
    servers = load()
    servers[req.name] = _clean(req, servers.get(req.name, {}))
    save(servers)
    manager.disconnect(req.name)
    return connect(req.name)


class Paste(BaseModel):
    text: str


# --- Ready-made servers --------------------------------------------------------------------------

# Each asks for at most one thing (`needs`: a folder or a token) and may need a program installed.
PRESETS = [
    {
        "id": "filesystem",
        "name": "files",
        "title": "Files",
        "description": "Read, search and write files in one folder you choose, and nowhere else.",
        "needs": "folder",
        "requires": "node",
        "url": "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
    },
    {
        "id": "github",
        "name": "github",
        "title": "GitHub",
        "description": "Issues, pull requests, code search and repositories, through GitHub's own MCP server.",
        "needs": "token",
        "requires": None,
        "url": "https://github.com/github/github-mcp-server",
    },
    {
        "id": "blender",
        "name": "blender",
        "title": "Blender",
        "description": "Build and change 3D scenes in a running Blender. Needs the blender-mcp add-on installed and connected in Blender.",
        "needs": None,
        "requires": None,
        "url": "https://github.com/ahujasid/blender-mcp",
    },
]


def preset_config(preset: str, value: str | None) -> dict:
    """The server config a ready-made server needs, with the folder or token given."""
    if preset == "filesystem":
        folder = Path(value or "").expanduser()
        if not value or not folder.is_dir():
            raise HTTPException(422, "Choose a folder for the files server to work in.")
        if not shutil.which("npx"):
            raise HTTPException(409, "The files server runs with Node.js, which isn't installed. Install it from nodejs.org, then add the server.")
        return {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", str(folder.resolve())]}
    if preset == "github":
        if not value or not value.strip():
            raise HTTPException(422, "Paste a GitHub personal access token (github.com/settings/tokens).")
        return {"url": "https://api.githubcopilot.com/mcp/", "type": "http", "headers": {"Authorization": f"Bearer {value.strip()}"}}
    if preset == "blender":
        from .llamacpp import uv_binary

        try:
            uv = uv_binary()
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        return {"command": uv, "args": ["tool", "run", "blender-mcp"]}
    raise HTTPException(404, f"There's no ready-made server called {preset}.")


@router.get("/presets")
def list_presets():
    added = load()
    return {"presets": [{**p, "added": p["name"] in added, "ready": not p["requires"] or bool(shutil.which("npx"))} for p in PRESETS]}


class AddPreset(BaseModel):
    value: str | None = None  # the folder or token it needs


@router.post("/presets/{preset}")
def add_preset(preset: str, req: AddPreset):
    entry = next((p for p in PRESETS if p["id"] == preset), None)
    if not entry:
        raise HTTPException(404, f"There's no ready-made server called {preset}.")
    servers = load()
    servers[entry["name"]] = preset_config(preset, req.value)
    save(servers)
    manager.disconnect(entry["name"])
    return connect(entry["name"])


@router.post("/import")
def import_servers(req: Paste):
    """Adds servers from pasted JSON: {"mcpServers": {...}}, {"servers": {...}}, or one {"name": {...}} map."""
    try:
        data = json.loads(req.text)
    except ValueError as e:
        raise HTTPException(422, f"That isn't JSON: {e}") from None
    found = data.get("mcpServers") or data.get("servers") or data if isinstance(data, dict) else None
    found = {k: v for k, v in (found or {}).items() if isinstance(v, dict) and (v.get("command") or v.get("url"))}
    if not found:
        raise HTTPException(422, 'No servers found. Paste JSON like {"mcpServers": {"name": {"command": "…"}}}.')
    servers = load()
    for name, cfg in found.items():
        servers[name] = {k: v for k, v in cfg.items() if k in ("command", "args", "env", "cwd", "url", "type", "headers")}
    save(servers)
    return {"added": sorted(found), **list_servers()}


class ServerUpdate(BaseModel):
    enabled: bool | None = None
    tools: dict[str, str] | None = None  # tool -> off | ask | auto


@router.put("/{name}")
def update_server(name: str, req: ServerUpdate):
    servers = load()
    if name not in servers:
        raise HTTPException(404, f"There's no MCP server called {name}.")
    cfg = servers[name]
    if req.enabled is not None:
        cfg["enabled"] = req.enabled
        if not req.enabled:
            manager.disconnect(name)
    if req.tools:
        if bad := [v for v in req.tools.values() if v not in ("off", "ask", "auto")]:
            raise HTTPException(422, f"{bad[0]!r} isn't off, ask or auto.")
        cfg["tools"] = {**(cfg.get("tools") or {}), **req.tools}
    save(servers)
    return _summary(name, cfg)


@router.post("/{name}/connect")
def connect(name: str):
    """Connects (again) and lists the server's tools; a server that can't be reached says why in `error`."""
    servers = load()
    if name not in servers:
        raise HTTPException(404, f"There's no MCP server called {name}.")
    manager.disconnect(name)
    if servers[name].get("enabled") is not False:
        try:
            manager.connection(name)
        except McpError:
            pass
    return _summary(name, servers[name])


@router.delete("/{name}")
def remove_server(name: str):
    servers = load()
    if servers.pop(name, None) is None:
        raise HTTPException(404, f"There's no MCP server called {name}.")
    manager.disconnect(name)
    save(servers)
    return list_servers()


def parse_command(line: str) -> tuple[str, list[str]]:
    """A command line typed into the app, as a command and its arguments."""
    parts = shlex.split(line, posix=os.name != "nt")
    if not parts:
        raise HTTPException(422, "Give the command that starts the server.")
    return parts[0], parts[1:]
