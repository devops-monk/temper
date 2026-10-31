"""The `temper` commands, run from a terminal while the app (or `temper serve`) is running.

    temper models                                              the models on this computer
    temper load <model or path> [--context N]                  load one for chat and the API
    temper chat ["prompt"]                                     ask the loaded model once, or chat until /bye
    temper train <config.yaml> [--set key=value ...] [--dry-run]   fine-tune, following the run
    temper runs                                                past training runs
    temper export <run> [--format gguf] [--quant q4_k_m] ...   export a run, following the export
    temper serve [--port N] [--host H]                         run the engine without the app (servers, CI)
    temper start claude|codex|opencode|aider [their own arguments]   start an agent on the model loaded in Temper
    temper mcp add claude|codex|cursor|vscode                  register Temper's MCP server with an agent
    temper doctor --mcp                                        check that server, and which agents have it

The running engine leaves its address and key in engine.json in Temper's home (readable only by
this user), which is how these find it. The key stays the same across restarts, so agent configs
written here keep working.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import sys
import time
import uuid
from pathlib import Path

import httpx
import psutil

from . import home

RUNNING = "engine.json"


class CliError(Exception):
    pass


# --- Finding the running engine ------------------------------------------------------------------


def write_running(url: str, token: str) -> None:
    """Called by the engine as it starts."""
    path = home() / RUNNING
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "token": token, "pid": os.getpid()}, f)


def clear_running() -> None:
    """Called by the engine as it stops; leaves the file alone if another engine has since written it."""
    path = home() / RUNNING
    try:
        if json.loads(path.read_text()).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def running() -> dict:
    """The running engine's url and token."""
    try:
        found = json.loads((home() / RUNNING).read_text())
    except (OSError, ValueError):
        found = None
    if not found or not psutil.pid_exists(found.get("pid", -1)):
        raise CliError("Temper isn't running. Open the Temper app, then run this again.")
    try:
        httpx.get(f"{found['url']}/api/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        raise CliError(f"Temper's engine isn't answering at {found['url']}. Restart it from the app, then run this again.") from None
    return found


def loaded_model(engine: dict) -> str:
    with _client(engine) as c:
        # Loaded ones only: /v1/models also offers what a request could load on demand.
        models = [m["id"] for m in _check(c.get("/v1/models")).get("data") or [] if m.get("loaded", True)]
    if not models:
        raise CliError("No model is loaded in Temper. Load one on the Models page, then run this again.")
    return models[0]


# --- temper start <agent> ------------------------------------------------------------------------

# What each agent is started with: its command, the arguments and environment that point it at
# Temper, and how to install it.
INSTALL = {
    "codex": "npm install -g @openai/codex",
    "opencode": "npm install -g opencode-ai",
    "aider": "uv tool install aider-chat",
    "claude": "npm install -g @anthropic-ai/claude-code",
}


def launch(agent: str, engine: dict, model: str) -> tuple[list[str], dict]:
    """The command line and extra environment that start `agent` on Temper's model."""
    v1 = f"{engine['url']}/v1"
    if agent == "codex":
        provider = f'{{ name = "Temper", base_url = "{v1}", env_key = "TEMPER_API_KEY", wire_api = "chat" }}'
        return ["codex", "-c", 'model_provider="temper"', "-c", f"model_providers.temper={provider}", "-m", model], {"TEMPER_API_KEY": engine["token"]}
    if agent == "opencode":
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"temper": {"npm": "@ai-sdk/openai-compatible", "name": "Temper", "options": {"baseURL": v1, "apiKey": "{env:TEMPER_API_KEY}"}, "models": {model: {"name": model}}}},
        }
        return ["opencode", "--model", f"temper/{model}"], {"OPENCODE_CONFIG_CONTENT": json.dumps(config), "TEMPER_API_KEY": engine["token"]}
    if agent == "aider":
        return ["aider", "--model", f"openai/{model}"], {"OPENAI_API_BASE": v1, "OPENAI_API_KEY": engine["token"]}
    if agent == "claude":
        # Claude Code adds /v1/messages itself; every model it asks for (main and background) is the loaded one.
        env = {"ANTHROPIC_BASE_URL": engine["url"], "ANTHROPIC_AUTH_TOKEN": engine["token"], "ANTHROPIC_API_KEY": "", "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model}
        return ["claude"], {**env, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    raise CliError(f"Temper doesn't know how to start {agent!r}. It can start: claude, codex, opencode, aider.")


def start(agent: str, args: list[str]) -> int:
    engine = running()
    model = loaded_model(engine)
    command, env = launch(agent, engine, model)
    exe = shutil.which(command[0])
    if not exe:
        raise CliError(f"{command[0]} isn't installed. Install it with: {INSTALL[agent]}")
    print(f"Starting {agent} on {model} from Temper ({engine['url']}).", file=sys.stderr)
    env = {**os.environ, **env}
    if os.name == "nt":  # Windows has no exec; wait for the agent instead
        return subprocess.call([exe, *command[1:], *args], env=env)
    os.execve(exe, [command[0], *command[1:], *args], env)
    return 0  # not reached


# --- temper mcp add <agent> ----------------------------------------------------------------------


def _config_dir(app: str) -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support" / app
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData/Roaming") / app
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / app


def json_config(agent: str) -> tuple[Path, str]:
    """The JSON file an agent keeps MCP servers in, and the key they sit under."""
    if agent == "cursor":
        return Path.home() / ".cursor/mcp.json", "mcpServers"
    return _config_dir("Code") / "User/mcp.json", "servers"  # vscode


def codex_config() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"


CODEX_TABLE = re.compile(r"^\[mcp_servers\.temper\]\n(?:(?!\[).*\n?)*", re.M)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    path.write_text(text, encoding="utf-8")


def add(agent: str) -> str:
    """Registers Temper's MCP server with `agent`; returns what was done."""
    engine = running()
    url, header = f"{engine['url']}/mcp", f"Bearer {engine['token']}"
    if agent == "claude":
        if not shutil.which("claude"):
            raise CliError(f"Claude Code isn't installed. Install it with: {INSTALL['claude']}")
        subprocess.run(["claude", "mcp", "remove", "--scope", "user", "temper"], capture_output=True)
        done = subprocess.run(["claude", "mcp", "add", "--scope", "user", "--transport", "http", "temper", url, "--header", f"Authorization: {header}"], capture_output=True, text=True)
        if done.returncode:
            raise CliError(f"claude mcp add failed: {(done.stderr or done.stdout).strip()}")
        return "Added Temper to Claude Code for every project (claude mcp list shows it)."
    if agent == "codex":
        path = codex_config()
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        table = f'[mcp_servers.temper]\nurl = "{url}"\nbearer_token_env_var = "TEMPER_API_KEY"\n'
        text = CODEX_TABLE.sub(lambda _: table + "\n", text, count=1) if CODEX_TABLE.search(text) else (text.rstrip() + "\n\n" if text.strip() else "") + table
        _write(path, text)
        return f"Added Temper to {path}. Codex reads its key from TEMPER_API_KEY:\n  export TEMPER_API_KEY={engine['token']}"
    if agent in ("cursor", "vscode"):
        path, key = json_config(agent)
        try:
            config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except ValueError:
            raise CliError(f"{path} isn't valid JSON, so Temper left it alone. Fix it, then run this again.") from None
        entry = {"url": url, "headers": {"Authorization": header}} | ({"type": "http"} if agent == "vscode" else {})
        config.setdefault(key, {})["temper"] = entry
        _write(path, json.dumps(config, indent=2) + "\n")
        return f"Added Temper to {path}."
    raise CliError(f"Temper doesn't know where {agent!r} keeps MCP servers. It can add itself to: claude, codex, cursor, vscode.")


# --- temper doctor --mcp -------------------------------------------------------------------------


def registered(engine: dict) -> list[tuple[str, str, bool]]:
    """For each agent: where Temper's MCP server is (or isn't) set up, and whether it points at this engine."""
    url, out = f"{engine['url']}/mcp", []
    claude = Path.home() / ".claude.json"
    try:
        entry = json.loads(claude.read_text(encoding="utf-8")).get("mcpServers", {}).get("temper")
    except (OSError, ValueError):
        entry = None
    out.append(("Claude Code", f"in {claude}" if entry else "not added", bool(entry) and entry.get("url") == url))
    codex = codex_config()
    text = codex.read_text(encoding="utf-8") if codex.exists() else ""
    m = CODEX_TABLE.search(text)
    out.append(("Codex", f"in {codex}" if m else "not added", bool(m) and f'"{url}"' in m.group(0)))
    for agent, label in (("cursor", "Cursor"), ("vscode", "VS Code")):
        path, key = json_config(agent)
        try:
            entry = json.loads(path.read_text(encoding="utf-8")).get(key, {}).get("temper") if path.exists() else None
        except ValueError:
            entry = None
        out.append((label, f"in {path}" if entry else "not added", bool(entry) and entry.get("url") == url))
    return out


def doctor_mcp() -> int:
    from .mcp import Connection, McpError

    engine = running()
    url = f"{engine['url']}/mcp"
    conn = Connection("temper", {"url": url, "headers": {"Authorization": f"Bearer {engine['token']}"}})
    try:
        conn.start()
        print(f"✓ Temper's MCP server  {url} answers, with {len(conn.tools)} tools: {', '.join(t['name'] for t in conn.tools)}")
        ok = True
    except McpError as e:
        print(f"✗ Temper's MCP server  {e}")
        ok = False
    finally:
        conn.close()
    for label, where, current in registered(engine):
        mark = "✓" if current else "!" if where != "not added" else "·"
        note = "" if current or where == "not added" else " (points at another address or key: run temper mcp add again)"
        print(f"{mark} {label:<20} {where}{note}")
    print("\nAdd Temper to an agent with: temper mcp add claude|codex|cursor|vscode")
    return 0 if ok else 1


# --- Entry ---------------------------------------------------------------------------------------

# --- temper models | load | chat | train | runs | export | serve ----------------------------------


def _client(engine: dict) -> httpx.Client:
    return httpx.Client(base_url=engine["url"], headers={"Authorization": f"Bearer {engine['token']}"}, timeout=httpx.Timeout(30, read=None))


def _check(r: httpx.Response):
    """The reply's JSON, or the engine's own explanation as a CliError."""
    if r.status_code < 400:
        return r.json()
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = r.text
    if isinstance(detail, list):  # a validation error: say which setting and why
        detail = "; ".join(f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
    raise CliError(str(detail))


def _size(n: int | None) -> str:
    return f"{(n or 0) / 1e9:.1f} GB" if (n or 0) >= 1e8 else f"{(n or 0) / 1e6:.0f} MB"


def models_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper models", description="The models on this computer.")
    p.add_argument("--json", action="store_true", help="print JSON")
    a = p.parse_args(args)
    with _client(running()) as c:
        models = _check(c.get("/api/models/local"))
    if a.json:
        print(json.dumps(models, indent=2))
        return 0
    if not models:
        print("No models yet. Download one on the Models page.")
    for m in models:
        print(f"{m['id']}  {_size(m['size'])}" + ("  (missing)" if m.get("missing") else ""))
        for g in m.get("gguf") or []:
            print(f"    {g['name']}  {_size(g['size'])}")
    return 0


def resolve_model(models: list[dict], wanted: str, quant: str | None = None) -> tuple[str, str]:
    """The path to load and its name, from what was typed: a path, a GGUF's file name, or (part of) a model's name."""
    from .model_names import pick

    try:
        return pick(models, wanted, quant)
    except LookupError as e:
        raise CliError(f"{e} `temper models` lists them.") from e


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _table(rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip())


def _estimate(c: httpx.Client, path: str, context: int | None) -> dict | None:
    """What loading a GGUF would take (memory.py); None for what the estimate can't read, like an MLX folder."""
    r = c.post("/api/memory/chat", json={"path": path, **({"context": context} if context else {})})
    return r.json() if r.status_code < 400 else None


def load_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper load", description="Load a model for chat and the API.")
    p.add_argument("model", help="a model's name, a GGUF file name, or a path")
    p.add_argument("--quant", help="which of the model's GGUF files, like Q8_0")
    p.add_argument("--context", type=int, help="context length in tokens")
    p.add_argument("--gpu-layers", type=int, metavar="N", help="layers on the GPU (0 runs on the CPU; default all)")
    p.add_argument("--alias", help="the name requests use for it (default its file or repo name)")
    p.add_argument("--ttl", type=int, metavar="SECONDS", help="unload after this long unused (default the idle time in Settings; 0 never)")
    p.add_argument("--alongside", action="store_true", help="keep chat's model loaded and load this one next to it")
    p.add_argument("--estimate-only", action="store_true", help="print the memory it would take, without loading it")
    p.add_argument("-y", "--yes", action="store_true", help="load even when it doesn't look like it fits")
    a = p.parse_args(args)
    with _client(running()) as c:
        path, name = resolve_model(_check(c.get("/api/models/local")), a.model, a.quant)
        est = _estimate(c, path, a.context)
        if a.estimate_only:
            if est is None:
                raise CliError(f"Temper can't estimate {name}'s memory; estimates cover GGUF files.")
            free = est["budget"]["free"]
            print(f"{name} at {est['context']} tokens of context needs about {_size(est['total'])}: {_size(est['weights'])} of weights, "
                  f"{_size(est['kv_cache'])} of cache and {_size(est['overhead'])} of working memory. {_size(free)} is free, "
                  + {"fits": "so it fits.", "tight": "so close other apps first.", "too_large": "which isn't enough."}[est["verdict"]])
            return 0
        if est and est["verdict"] != "fits" and not a.yes:
            raise CliError(f"{name} needs about {_size(est['total'])} and {_size(est['budget']['free'])} is free. "
                           "Lower --context, pick a smaller --quant, or pass -y to try anyway.")
        body = {"path": path, "name": a.alias or name, "alongside": a.alongside}
        body |= {k: v for k, v in (("context", a.context), ("gpu_layers", a.gpu_layers), ("ttl", a.ttl)) if v is not None}
        _check(c.post("/api/runtime/load", json=body))
        print(f"Loading {body['name']}", end="", flush=True, file=sys.stderr)
        while (state := _loaded_state(c, path))["state"] == "loading":
            print(".", end="", flush=True, file=sys.stderr)
            time.sleep(1)
        print(file=sys.stderr)
    if state["state"] != "ready":
        raise CliError(state.get("error") or "The model didn't load.")
    print(f"Loaded {state['model']}" + (f" with {state['context']} tokens of context" if state.get("context") else "")
          + (" alongside chat's model" if not state.get("chat") else ""))
    return 0


def _loaded_state(c: httpx.Client, path: str) -> dict:
    """The status of the server holding `path`, chat's or alongside it; failed when it vanished while loading."""
    for m in _check(c.get("/api/runtime/loaded"))["models"]:
        if m.get("path") == path:
            return m
    state = _check(c.get("/api/runtime"))  # a failed load leaves no loaded server, only chat's error
    return state if state.get("path") == path else {"state": "failed", "error": state.get("error")}


def ps_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper ps", description="The models loaded now.")
    p.add_argument("--json", action="store_true", help="print JSON")
    a = p.parse_args(args)
    with _client(running()) as c:
        listing = _check(c.get("/api/runtime/loaded"))
    models, idle = listing["models"], listing["idle_minutes"] * 60
    if a.json:
        print(json.dumps(listing, indent=2))
        return 0
    if not models:
        print("No models are loaded. `temper load <model>` loads one" + (", and API requests that name one load it." if listing.get("load_on_demand") else "."))
        return 0
    rows = [["MODEL", "ROLE", "ENGINE", "CONTEXT", "STATE", "IDLE", "UNLOADS IN"]]
    for m in models:
        limit = m["ttl"] if m.get("ttl") is not None else idle
        left = None if not limit or m.get("idle") is None else limit - m["idle"]
        role = "chat" if m.get("chat") else "on demand" if m.get("on_demand") else "alongside"
        rows.append([m.get("model") or Path(m.get("path") or "").name, role, m.get("engine") or "-", str(m.get("context") or "-"), m["state"],
                     _duration(m.get("idle")), "never" if not limit else _duration(left)])
    _table(rows)
    return 0


def unload_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper unload", description="Unload a model, or all of them.")
    p.add_argument("model", nargs="?", help="its name or path, as `temper ps` shows it")
    p.add_argument("-a", "--all", action="store_true", help="unload every model")
    a = p.parse_args(args)
    if not a.model and not a.all:
        p.error("name a model, or pass --all")
    with _client(running()) as c:
        models = _check(c.get("/api/runtime/loaded"))["models"]
        if a.all:
            chosen = sorted(models, key=lambda m: bool(m.get("chat")))  # chat's last, so nothing moves into its place
        else:
            w = a.model.lower()
            chosen = [m for m in models if w in ((m.get("model") or "").lower(), (m.get("path") or "").lower(), Path(m.get("path") or "").name.lower())]
            if not chosen:
                raise CliError(f"No loaded model called {a.model!r}. `temper ps` lists them.")
        for m in chosen:
            _check(c.post("/api/runtime/loaded/unload", json={"path": m["path"]}))
            print(f"Unloaded {m.get('model') or m['path']}")
    if a.all and not chosen:
        print("No models were loaded.")
    return 0


DIM, PLAIN = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def _reply(c: httpx.Client, messages: list[dict], model: str | None = None, think: str = "auto") -> tuple[str, str, dict]:
    """Streams one reply to stdout and returns it, its reasoning and its stats.

    A thinking model's reasoning is left out, or shown dimmed with think="on"; think="off" asks the
    model not to think, where its template allows that.
    """
    from .anthropic_api import ThinkFilter

    body: dict = {"messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    if think == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}  # templates without the switch ignore it
    think_filter, text, reasoning, usage = ThinkFilter(), "", "", {}
    start, first, shown = time.monotonic(), None, False
    with c.stream("POST", "/v1/chat/completions", json=body) as r:
        if r.status_code >= 400:
            r.read()
            _check(r)
        for line in r.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(line[5:])
            except ValueError:
                continue
            usage = chunk.get("usage") or usage
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if thought := delta.get("reasoning_content") or delta.get("reasoning") or "":
                first = first or time.monotonic()
                reasoning += thought
                if think == "on":
                    print(f"{DIM}{thought}{PLAIN}", end="", flush=True)
                    shown = True
            part = think_filter.feed(delta.get("content") or "")
            if not text:
                part = part.lstrip()  # what a thinking block leaves before the answer
            if part:
                first = first or time.monotonic()
                if shown:
                    print("\n")  # the answer starts below the reasoning
                    shown = False
                text += part
                print(part, end="", flush=True)
    if rest := (think_filter.flush() if text else think_filter.flush().lstrip()):
        text += rest
        print(rest, end="")
    print()
    took = time.monotonic() - start
    out = usage.get("completion_tokens")
    stats = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out,
        "seconds": round(took, 2),
        "first_token_seconds": round(first - start, 2) if first else None,
        "tokens_per_second": round(out / (took - (first - start)), 1) if out and first and took > first - start else None,
    }
    return text, reasoning, stats


CHAT_HELP = """Commands:
  /model [name]        the model to talk to: a loaded one, or a downloaded one to load (no name lists them)
  /system [text]       set the system prompt (no text clears it)
  /think on|off|auto   show a thinking model's reasoning, ask it not to think, or think quietly (the default)
  /stats               tokens, speed and time to first token of the last reply
  /save [title]        keep this conversation in Temper's chats
  /clear               start over
  /bye                 end (or Ctrl-D)
Start a line with \"\"\" to write several lines, and end with \"\"\" on a line of its own."""


class Repl:
    """`temper chat` with no prompt: a conversation, with the commands in CHAT_HELP."""

    def __init__(self, c: httpx.Client, system: str | None) -> None:
        self.c, self.system = c, system
        self.model: str | None = None  # None: chat's model
        self.think, self.history, self.stats, self.chat_id = "auto", [], None, None

    def messages(self) -> list[dict]:
        return ([{"role": "system", "content": self.system}] if self.system else []) + self.history

    def read(self) -> str | None:
        """One message: a line, or a block between \"\"\" lines. None at the end of input."""
        try:
            line = input("> ")
            if not line.lstrip().startswith('"""'):
                return line
            lines = [line.lstrip()[3:]]
            while not lines[-1].rstrip().endswith('"""') or len(lines) == 1 and lines[0].strip() == "":
                lines.append(input(". "))
            lines[-1] = lines[-1].rstrip()[:-3]
            return "\n".join(lines).strip("\n")
        except EOFError:
            print(file=sys.stderr)
            return None

    def command(self, line: str) -> bool:
        """Runs a /command; False when it ends the chat."""
        name, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if name in ("bye", "exit", "quit"):
            return False
        if name == "help":
            print(CHAT_HELP)
        elif name == "clear":
            self.history, self.chat_id = [], None
            print("Started over.")
        elif name == "system":
            self.system = arg or None
            print(f"System prompt set." if arg else "System prompt cleared.")
        elif name == "think":
            if arg not in ("on", "off", "auto"):
                print(f"/think on, off or auto (now {self.think}).")
            else:
                self.think = arg
                print({"on": "Showing the reasoning.", "off": "Asking the model not to think.", "auto": "Thinking quietly."}[arg])
        elif name == "model":
            self.pick_model(arg)
        elif name == "stats":
            self.show_stats()
        elif name == "save":
            self.save(arg)
        else:
            print(f"There's no /{name}. /help lists the commands.")
        return True

    def pick_model(self, wanted: str) -> None:
        models = _check(self.c.get("/v1/models")).get("data") or []
        if not wanted:
            for m in models:
                mark = "*" if m["id"] == self.model or (self.model is None and m is models[0] and m.get("loaded", True)) else " "
                print(f"{mark} {m['id']}" + ("" if m.get("loaded", True) else "  (loads when chosen)"))
            return
        match = next((m for m in models if m["id"].lower() == wanted.lower()), None) or next((m for m in models if wanted.lower() in m["id"].lower()), None)
        if match is None:
            print(f"No model called {wanted!r}. /model lists them.")
            return
        self.model = match["id"]
        print(f"Talking to {self.model}" + ("" if match.get("loaded", True) else "; it loads with your next message") + ".")

    def show_stats(self) -> None:
        s = self.stats
        if not s:
            print("No reply yet.")
            return
        parts = [f"{s['completion_tokens']} tokens" if s["completion_tokens"] is not None else None,
                 f"{s['tokens_per_second']} tokens/s" if s["tokens_per_second"] else None,
                 f"first token after {s['first_token_seconds']} s" if s["first_token_seconds"] is not None else None,
                 f"{s['seconds']} s in all",
                 f"{s['prompt_tokens']} tokens of prompt" if s["prompt_tokens"] is not None else None]
        print(", ".join(p for p in parts if p))

    def save(self, title: str) -> None:
        if not self.history:
            print("Nothing to save yet.")
            return
        body = {"messages": self.messages(), "model": self.model}
        if self.chat_id:
            _check(self.c.put(f"/api/chats/{self.chat_id}/messages", json=body))
            if title:
                _check(self.c.patch(f"/api/chats/{self.chat_id}", json={"title": title}))
        else:
            chat = _check(self.c.post("/api/chats", json={**body, "title": title or None}))
            self.chat_id = chat["id"]
        print("Saved in Temper's chats. /save again keeps it up to date.")

    def run(self) -> int:
        print("Chatting with the model loaded in Temper. /help lists the commands; /bye (or Ctrl-D) ends.", file=sys.stderr)
        while (line := self.read()) is not None:
            if not line.strip():
                continue
            if line.startswith("/") and not line.startswith("//"):
                if not self.command(line.strip()):
                    return 0
                continue
            self.history.append({"role": "user", "content": line.removeprefix("/")})  # // sends a line that starts with /
            try:
                text, reasoning, self.stats = _reply(self.c, self.messages(), self.model, self.think)
            except CliError as e:
                self.history.pop()
                print(e, file=sys.stderr)
                continue
            self.history.append({"role": "assistant", "content": text, **({"reasoning": reasoning} if reasoning else {}), **({"model": self.model} if self.model else {})})
        return 0


def chat_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper chat", description="Ask the loaded model once, or chat with it until /bye.")
    p.add_argument("prompt", nargs="*", help="ask this once and print the reply (or pipe it in)")
    p.add_argument("--system", help="a system prompt")
    p.add_argument("--model", help="a loaded model to ask, or a downloaded one to load (default chat's model)")
    a = p.parse_args(args)
    engine = running()
    if not a.model:
        loaded_model(engine)
    with _client(engine) as c:
        prompt = " ".join(a.prompt) or (sys.stdin.read() if not sys.stdin.isatty() else "")
        if prompt.strip():
            messages = [{"role": "system", "content": a.system}] if a.system else []
            _reply(c, [*messages, {"role": "user", "content": prompt}], a.model)
            return 0
        repl = Repl(c, a.system)
        repl.model = a.model
        return repl.run()


def _bar(done: float, total: float, speed: float | None = None) -> None:
    """A download's progress on one line, redrawn in place; nothing when stderr isn't a terminal."""
    if not sys.stderr.isatty():
        return
    share = min(1.0, done / total) if total else 0.0
    rate = f"  {speed / 1e6:.1f} MB/s" if speed else ""
    print(f"\r  [{'#' * int(share * 30):<30}] {share:4.0%}  {_size(done)} of {_size(total)}{rate}   ", end="", flush=True, file=sys.stderr)


def follow(c: httpx.Client, job_id: str, logs: bool) -> dict:
    """Prints a job's progress until it ends, and returns how it ended. Ctrl-C stops following, not the job."""
    try:
        with c.stream("GET", f"/api/jobs/{job_id}/events") as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:])
                kind = ev["event"]
                if kind == "log" and logs:
                    print(ev.get("line", ""))
                elif kind == "phase":
                    print(f"· {ev.get('name')}")
                elif kind == "metrics" and ev.get("loss") is not None:
                    total = f"/{ev['total_steps']}" if ev.get("total_steps") else ""
                    eta = f"  {int(ev['eta']) // 60}m{int(ev['eta']) % 60:02d}s left" if ev.get("eta") else ""
                    print(f"  step {ev.get('step')}{total}  loss {ev['loss']:.4f}{eta}")
                elif kind == "eval" and ev.get("val_loss") is not None:
                    print(f"  step {ev.get('step')}  held-out loss {ev['val_loss']:.4f}")
                elif kind == "progress" and ev.get("total"):
                    _bar(ev["received"], ev["total"], ev.get("speed"))
                elif kind == "error":
                    print(ev.get("message", ""), file=sys.stderr)
                elif kind == "status":
                    if sys.stderr.isatty():
                        print(file=sys.stderr)  # off the progress bar's line
                    return ev
    except KeyboardInterrupt:
        print("\nStopped following. It carries on in Temper; the app shows it.", file=sys.stderr)
        raise
    return _check(c.get(f"/api/jobs/{job_id}"))


PATH_SETTINGS = ("dataset", "eval_dataset", "base_model", "resume_from", "output_dir")


def train_config(path: str | None, sets: list[str]) -> dict:
    """A training config from a YAML file plus `key=value` overrides (values read as YAML, so 2e-4 is a number)."""
    import yaml

    cfg: dict = {}
    if path:
        try:
            cfg = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        except OSError as e:
            raise CliError(f"Couldn't read {path}: {e.strerror}") from None
        except yaml.YAMLError as e:
            raise CliError(f"{path} isn't valid YAML: {e}") from None
        if not isinstance(cfg, dict):
            raise CliError(f"{path} should be settings, one per line, like `base_model: Qwen/Qwen3-0.6B`.")
    for item in sets:
        key, eq, value = item.partition("=")
        if not eq or not key.strip():
            raise CliError(f"--set takes key=value, like --set learning_rate=1e-4 (got {item!r}).")
        cfg[key.strip()] = yaml.safe_load(value) if value.strip() else None
    # The engine runs elsewhere: local paths are made whole, against the config's folder, then this one.
    base = Path(path).expanduser().resolve().parent if path else Path.cwd()
    for key in PATH_SETTINGS:
        value = cfg.get(key)
        if isinstance(value, str) and not value.startswith(("~", "/", "http://", "https://", "s3://", "gs://")):
            for root in (base, Path.cwd()):
                if (root / value).exists():
                    cfg[key] = str((root / value).resolve())
                    break
    return cfg


def train_cmd(args: list[str]) -> int:
    from pydantic import ValidationError

    from .training import TrainConfig

    p = argparse.ArgumentParser(prog="temper train", description="Fine-tune a model from a YAML config, following the run.")
    p.add_argument("config", nargs="?", help="a YAML file of training settings (the same names as the Train page's)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one setting; repeatable")
    p.add_argument("--dry-run", action="store_true", help="check the config and print it in full, without training")
    p.add_argument("--logs", action="store_true", help="print the trainer's own output too")
    a = p.parse_args(args)
    raw = train_config(a.config, a.set)
    unknown = sorted(set(raw) - set(TrainConfig.model_fields))
    if unknown:
        raise CliError(f"Unknown settings: {', '.join(unknown)}. The settings are: {', '.join(TrainConfig.model_fields)}.")
    try:
        cfg = TrainConfig(**raw)
    except ValidationError as e:
        raise CliError("; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())) from None
    if a.dry_run:
        import yaml

        print(yaml.safe_dump(cfg.model_dump(exclude_none=True), sort_keys=False).rstrip())
        print("\nThe config is valid. Run it without --dry-run to train.", file=sys.stderr)
        return 0
    with _client(running()) as c:
        job = _check(c.post("/api/train", json=raw))
        print(f"Training {job['title']}", file=sys.stderr)
        end = follow(c, job["id"], a.logs)
    if end.get("status") != "done":
        raise CliError(end.get("error") or f"Training {end.get('status')}.")
    out = (end.get("result") or {}).get("output_dir") or (end.get("result") or {}).get("run_id") or ""
    print(f"Done. {out}".rstrip())
    return 0


def runs_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper runs", description="Training runs, newest first.")
    p.add_argument("--json", action="store_true", help="print JSON")
    a = p.parse_args(args)
    with _client(running()) as c:
        runs = _check(c.get("/api/train/runs"))
    if a.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print("No training runs yet.")
    for r in runs:
        loss = f"  loss {r['loss']:.4f}" if r.get("loss") is not None else ""
        held = f"  held-out {r['val_loss']:.4f}" if r.get("val_loss") is not None else ""
        steps = f"  {r['steps']} steps" if r.get("steps") else ""
        print(f"{r['id']}  {r['status']}  {r['base_model']}{steps}{loss}{held}")
    return 0


def export_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper export", description="Export a training run, following the export.")
    p.add_argument("run", help="the run's id, or part of it (see `temper runs`)")
    p.add_argument("--format", action="append", dest="formats", help="gguf, lora, merged_16bit, merged_4bit or mlx; repeatable (default gguf)")
    p.add_argument("--quant", action="append", dest="quants", help="a GGUF quant like q4_k_m or q8_0; repeatable (default q4_k_m)")
    p.add_argument("--imatrix", action="store_true", help="calibrate the quants on the run's training data")
    a = p.parse_args(args)
    with _client(running()) as c:
        runs = [r["id"] for r in _check(c.get("/api/train/runs"))]
        found = [r for r in runs if r == a.run] or [r for r in runs if a.run in r]
        if len(found) != 1:
            raise CliError(f"No run matches {a.run!r}. `temper runs` lists them." if not found else f"{a.run!r} matches {len(found)} runs: {', '.join(found[:5])}. Give more of the id.")
        body = {"run_id": found[0], "formats": a.formats or ["gguf"], "gguf_quants": a.quants or ["q4_k_m"], "imatrix": a.imatrix}
        job = _check(c.post("/api/export", json=body))
        print(f"Exporting {found[0]}", file=sys.stderr)
        end = follow(c, job["id"], logs=True)
    if end.get("status") != "done":
        raise CliError(end.get("error") or f"The export {end.get('status')}.")
    result = end.get("result") or {}
    print(f"Done: {result.get('output_dir', '')}")
    for f in result.get("gguf") or []:
        print(f"  {f}")
    return 0


def stable_key() -> str:
    """The engine's key, the same one the app uses (its api-key file), made if there isn't one yet."""
    path = home() / "api-key"
    try:
        key = path.read_text().strip()
        if len(key) >= 32 and key.isalnum():
            return key
    except OSError:
        pass
    key = uuid.uuid4().hex
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    return key


LADDER = ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs", "q3_k_m", "iq3_m", "q2_k")


def _pick_quant(c: httpx.Client, repo: str, ggufs: list[dict]) -> tuple[dict, dict | None]:
    """The best quant that fits this computer at 8K context, by the engine's own estimate; the smallest when none does."""
    ranked = sorted((g for g in ggufs if g.get("complete", True)), key=lambda g: next((i for i, q in enumerate(LADDER) if q in g["name"].lower()), len(LADDER)))
    for g in ranked:
        if not any(q in g["name"].lower() for q in LADDER):
            continue
        r = c.post("/api/memory/chat", json={"repo_id": repo, "filename": g["name"], "context": 8192})
        if r.status_code >= 400:
            return g, None  # no estimate (a header Temper can't read): take the usual pick
        est = r.json()
        if est["verdict"] == "fits":
            return g, est
    smallest = min(ggufs, key=lambda g: g["size"])
    return smallest, None


def get_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper get", description="Find a model on Hugging Face and download it.")
    p.add_argument("query", nargs="+", help="words to search for, or owner/name")
    p.add_argument("--quant", help="which GGUF to get, like Q8_0 (default: the best that fits this computer)")
    p.add_argument("--mlx", action="store_true", help="search MLX models instead of GGUFs (Apple Silicon)")
    p.add_argument("-y", "--yes", action="store_true", help="take the top search result and don't ask")
    a = p.parse_args(args)
    query = " ".join(a.query)
    with _client(running()) as c:
        if "/" in query and " " not in query:
            repo = query
        else:
            params = {"q": query, "limit": 8, "gguf": not a.mlx, **({"author": "mlx-community"} if a.mlx else {})}
            found = _check(c.get("/api/models/search", params=params))
            if not found:
                raise CliError(f"Nothing on Hugging Face matches {query!r}.")
            if a.yes or len(found) == 1 or not sys.stdin.isatty():
                repo = found[0]["id"]
            else:
                for i, m in enumerate(found, 1):
                    print(f"  {i}. {m['id']}  {m.get('downloads') or 0:,} downloads" + ("  (gated)" if m.get("gated") else ""))
                try:
                    choice = input(f"Which one? [1-{len(found)}, Enter for 1] ").strip() or "1"
                except EOFError:
                    raise CliError("\nNothing chosen.") from None
                if not choice.isdigit() or not 1 <= int(choice) <= len(found):
                    raise CliError("Pick one of the numbers.")
                repo = found[int(choice) - 1]["id"]
        info = _check(c.get("/api/models/repo", params={"id": repo}))
        ggufs = info["gguf"]
        if ggufs:
            if a.quant:
                ggufs = [g for g in ggufs if a.quant.lower() in g["name"].lower()]
                if not ggufs:
                    raise CliError(f"{repo} has no {a.quant} file. Its GGUFs: {', '.join(g['name'] for g in info['gguf'])}")
                chosen, est = ggufs[0], None
            else:
                chosen, est = _pick_quant(c, repo, ggufs)
            patterns, size = chosen["parts"], chosen["size"]
            what = chosen["name"] + (f", which needs about {_size(est['total'])} at 8K context and fits" if est else "")
        else:
            chosen = None
            patterns, size = [], sum(f["size"] or 0 for f in info["files"])
            what = f"the whole repository ({len(info['files'])} files)"
        print(f"{repo}: {what}. Downloading {_size(size)}.")
        job = _check(c.post("/api/models/download", json={"repo_id": repo, "patterns": patterns}))
        end = follow(c, job["id"], logs=False)
    if end.get("status") != "done":
        raise CliError(end.get("error") or f"The download {end.get('status')}.")
    print(f"Done. `temper load {chosen['name'] if ggufs else repo}` loads it.")
    return 0


def import_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper import", description="Add a model you already have: a GGUF, or a model folder.")
    p.add_argument("path", help="a .gguf file or a folder holding a model")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="copy it into Temper's imports folder (default: use it where it is)")
    mode.add_argument("--move", action="store_true", help="move it into Temper's imports folder")
    a = p.parse_args(args)
    path = os.path.abspath(Path(a.path).expanduser())  # not resolve(): cache files are symlinks to extensionless blobs
    how = "copy" if a.copy else "move" if a.move else "link"
    with _client(running()) as c:
        if how != "link":
            print(f"{'Copying' if a.copy else 'Moving'} {Path(path).name} into Temper's imports folder...", file=sys.stderr)
        found = _check(c.post("/api/models/import", json={"path": path, "mode": how}))
    target = Path(found["path"]).name if found["path"].lower().endswith(".gguf") else shlex.quote(found["path"])
    print(f"Added {found['name']} ({found['path']}). `temper load {target}` loads it.")
    return 0


def _log_line(e: dict) -> str:
    when = time.strftime("%H:%M:%S", time.localtime(e["time"]))
    tokens = f"{e['input_tokens'] or 0}→{e['output_tokens'] or 0} tokens" if e.get("input_tokens") is not None or e.get("output_tokens") is not None else ""
    speed = f"{e['tokens_per_second']} tokens/s" if e.get("tokens_per_second") else ""
    took = f"{e['ms'] / 1000:.1f}s" if e.get("ms") is not None else ""
    first = f"first token {e['first_ms'] / 1000:.1f}s" if e.get("first_ms") is not None and e.get("output_tokens") else ""
    cells = [when, e["method"], e["path"], e["key"], e.get("model") or "", str(e.get("status") or ""), took, first, tokens, speed]
    return "  ".join(c for c in cells if c)


def log_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper log", description="Follow the model API's requests as they happen.")
    p.add_argument("--content", action="store_true", help="also show each prompt and reply (kept in memory only, while this runs)")
    p.add_argument("--json", action="store_true", help="one JSON object per request")
    p.add_argument("-n", "--last", type=int, default=10, metavar="N", help="start with the last N requests (default 10)")
    a = p.parse_args(args)
    engine = running()
    params = {"content": "true" if a.content else "false"}
    with _client(engine) as c:
        recent = _check(c.get("/api/keys/log", params={**params, "limit": max(a.last, 1)}))[::-1]
        since = max((e.get("seq", 0) for e in recent), default=0)
        shown: set[str] = set()

        def show(e: dict) -> None:
            if not e.get("done") or e["id"] in shown:
                return  # each request once, when it's answered
            shown.add(e["id"])
            if a.json:
                print(json.dumps(e), flush=True)
                return
            print(_log_line(e), flush=True)
            if a.content:
                for label, text in (("prompt", e.get("prompt")), ("reply", e.get("reply"))):
                    if text:
                        print(f"    {label}: " + text.strip().replace("\n", "\n            "), flush=True)

        for e in recent[-a.last:] if a.last else []:
            show(e)
        if not a.json:
            print(f"Following {engine['url']}/v1" + (" with prompts and replies" if a.content else "") + ". Ctrl-C stops.", file=sys.stderr)
        try:
            with c.stream("GET", "/api/keys/log/stream", params={**params, "since": since}) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        show(json.loads(line[5:]))
        except KeyboardInterrupt:
            print(file=sys.stderr)
    return 0


def daemon_cmd(args: list[str]) -> int:
    from . import daemon

    p = argparse.ArgumentParser(prog="temper daemon", description="Run Temper's engine in the background, with no window or terminal.")
    sub = p.add_subparsers(dest="action", required=True)
    up = sub.add_parser("up", help="start it in the background")
    up.add_argument("--host", default="127.0.0.1", help="the address to listen on (default this computer only)")
    up.add_argument("--port", type=int, default=8765)
    sub.add_parser("down", help="stop the background engine")
    sub.add_parser("status", help="whether it's running, and whether it starts at login")
    sub.add_parser("install", help="start it at login (and now)")
    sub.add_parser("uninstall", help="stop starting it at login")
    a = p.parse_args(args)

    def answering() -> str | None:
        try:
            return running()["url"]
        except CliError:
            return None

    if a.action == "up":
        if url := answering():
            print(f"Temper is running already, at {url}." + ("" if daemon.pid() else " It's the app's engine; it stops when the app quits."))
            return 0
        started = daemon.start(a.host, a.port)
        url = daemon.wait_until_answering(answering)
        if url is None:
            raise CliError(f"The engine didn't come up. Its log: {daemon.log_file()}")
        print(f"Temper's engine is running in the background at {url} (pid {started}). `temper daemon down` stops it.")
    elif a.action == "down":
        if daemon.stop():
            print("Stopped the background engine.")
        elif answering():
            raise CliError("The running engine is the app's (or one started at login); quit the app, or run `temper daemon uninstall`.")
        else:
            print("No background engine was running.")
    elif a.action == "status":
        url, found, login = answering(), daemon.pid(), daemon.login_item()
        whose = "in the background" if found else "in the app (or from the login item)" if url else None
        print(f"Engine: {'running ' + whose + ' at ' + url if url else 'not running'}")
        print(f"Starts at login: {'yes' if login else 'no'}  (log: {daemon.log_file()})")
    elif a.action == "install":
        try:
            where = daemon.install()
        except RuntimeError as e:
            raise CliError(str(e)) from e
        print(f"Temper's engine now starts at login ({where}). `temper daemon uninstall` undoes it.")
    elif a.action == "uninstall":
        print("Temper's engine no longer starts at login." if daemon.uninstall() else "It wasn't set to start at login.")
    return 0


def engines_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper engines", description="The inference engines that run models, and what this computer gives them.")
    p.add_argument("--json", action="store_true", help="print JSON")
    sub = p.add_subparsers(dest="action")
    use = sub.add_parser("use", help="load model folders (not GGUF files) with this engine")
    use.add_argument("engine", help="mlx, transformers, vllm or sglang; `auto` picks by format again")
    sub.add_parser("install", help="install an engine into its own environment").add_argument("engine")
    sub.add_parser("remove", help="remove an installed engine").add_argument("engine")
    sub.add_parser("update", help="install the llama.cpp build this version of Temper uses, for this hardware")
    sub.add_parser("survey", help="what each engine can use here: GPU, memory, backends").add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    a = p.parse_args(args)
    with _client(running()) as c:
        engines = _check(c.get("/api/runtime/engines"))["engines"]
        names = {e["id"]: e for e in engines}

        def known(engine: str) -> str:
            found = next((k for k, e in names.items() if engine.lower() in (k, e["name"].lower())), None)
            if found is None:
                raise CliError(f"No engine called {engine}. These are: {', '.join(names)}.")
            return found

        if a.action is None:
            llama = _check(c.get("/api/llamacpp"))
            if a.json:
                print(json.dumps({"engines": engines, "llamacpp": llama}, indent=2))
                return 0
            rows = [["ENGINE", "VERSION", "STATUS", "LOADS"]]
            for e in engines:
                status = "installed" + (", preferred" if e["preferred"] else "") if e["available"] else e["unsupported"] or ("not installed" + (" (installable)" if e["installable"] else ""))
                rows.append([e["id"], e["version"] or "-", status, ", ".join(e["formats"])])
            _table(rows)
            if llama["available"] and not llama["managed"]:
                print(f"\nllama.cpp is the one at {llama['server']}; `temper engines update` installs Temper's own build ({llama['tag']}).")
            return 0
        if a.action == "use":
            engine = None if a.engine.lower() == "auto" else known(a.engine)
            engines = _check(c.put("/api/runtime/engines/preferred", json={"engine": engine}))["engines"]
            print(f"Model folders now load on {names[engine]['name']}; GGUF files still load on llama.cpp." if engine else "Back to picking the engine by the model's format.")
            return 0
        if a.action == "install":
            engine = known(a.engine)
            if names[engine]["available"]:
                print(f"{names[engine]['name']} is installed already ({names[engine]['version'] or 'version unknown'}).")
                return 0
            if engine == "llama.cpp":
                return engines_cmd(["update"])
            job = _check(c.post(f"/api/runtime/engines/{engine}/install"))
            print(f"Installing {names[engine]['name']} into its own environment; this can take a few minutes.")
            end = follow(c, job["id"], logs=False)
        elif a.action == "remove":
            engine = known(a.engine)
            if engine in ("llama.cpp", "mlx"):
                raise CliError(f"{names[engine]['name']} comes with Temper and can't be removed.")
            _check(c.delete(f"/api/runtime/engines/{engine}"))
            print(f"Removed {names[engine]['name']}.")
            return 0
        elif a.action == "update":
            llama = _check(c.get("/api/llamacpp"))
            if llama["managed"]:
                print(f"llama.cpp {llama['tag']} is installed, the build this version of Temper is tested with. Updating Temper brings newer builds.")
                return 0
            job = _check(c.post("/api/llamacpp/install", json={}))
            print(f"Installing llama.cpp {llama['tag']} for this computer.")
            end = follow(c, job["id"], logs=True)
        else:
            return _survey(c, engines, a.json)
    if end.get("status") != "done":
        raise CliError(end.get("error") or f"It ended {end.get('status')}.")
    print("Done.")
    return 0


def _survey(c: httpx.Client, engines: list[dict], as_json: bool) -> int:
    """`temper engines survey`: the hardware, and which engines can use it."""
    system = _check(c.get("/api/system"))
    if as_json:
        print(json.dumps({"system": system, "engines": engines}, indent=2))
        return 0
    mem = system["memory"]
    rows = [
        ["Computer", f"{system['os']['name']} {system['os']['release']} on {system['os']['machine']}, {system['cpu']['name']} ({system['cpu']['cores']} cores)"],
        ["Memory", f"{_size(mem['total'])}, {_size(mem['total'] - mem['used'])} free"],
    ]
    for g in system["gpus"]:
        shared = " shared with the CPU" if g.get("unified_memory") else ""
        rows.append([f"GPU {g['index']}", f"{g['name']} ({g['backend']}), {_size(g.get('memory_total'))}{shared}" + (f", driver {g['driver']}" if g.get("driver") else "")])
    if not system["gpus"]:
        rows.append(["GPU", "none found; models run on the CPU"])
    rows.append(["Backend", system["backend"]])
    _table(rows)
    print()
    backend = system["backend"]
    rows = [["ENGINE", "RUNS ON"]]
    for e in engines:
        if e["unsupported"]:
            where = f"can't run here: {e['unsupported']}"
        elif e["id"] == "llama.cpp":
            where = f"{backend} GPU; layers that don't fit run on the CPU" if backend != "cpu" else "CPU"
        elif e["id"] == "mlx":
            where = "Apple GPU (Metal), in unified memory"
        else:
            where = "CUDA or ROCm GPUs, split across them" if e["id"] in ("vllm", "sglang") else "GPU through PyTorch when it has one, else CPU"
        rows.append([e["id"], where + ("" if e["available"] or e["unsupported"] else " (not installed)")])
    _table(rows)
    return 0


def _choose(what: str, options: dict[str, str], given: str | None) -> str:
    """`given`, or a pick from a numbered list when someone is at the terminal, else the first option."""
    if given or not sys.stdin.isatty():
        return given or next(iter(options))
    print(f"{what}:")
    keys = list(options)
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k:<11} {options[k]}")
    try:
        choice = input(f"Which one? [1-{len(keys)}, Enter for 1] ").strip() or "1"
    except EOFError:
        raise CliError("Nothing picked.") from None
    if not choice.isdigit() or not 1 <= int(choice) <= len(keys):
        raise CliError(f"Pick a number from 1 to {len(keys)}.")
    return keys[int(choice) - 1]


def create_cmd(args: list[str]) -> int:
    from . import starters

    p = argparse.ArgumentParser(prog="temper create", description="Start a small app that uses Temper's models, with an API key filled in.")
    p.add_argument("folder", help="the new project's folder")
    p.add_argument("--template", "-t", choices=list(starters.TEMPLATES), help="; ".join(f"{k}: {v}" for k, v in starters.TEMPLATES.items()))
    p.add_argument("--lang", "-l", choices=starters.LANGUAGES, help="python (default) or typescript")
    p.add_argument("--model", help="the model it asks for (default: the one loaded in Temper)")
    a = p.parse_args(args)
    folder = Path(os.path.abspath(a.folder))
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise CliError(f"{folder} already exists and isn't empty; pick a new folder.")
    template = _choose("What should it do", starters.TEMPLATES, a.template)
    language = _choose("In which language", {"python": "run with uv or pip", "typescript": "run with Node 20+"}, a.lang)
    engine = running()
    model = a.model
    if not model:
        try:
            model = loaded_model(engine)
        except CliError:
            model = ""  # requests go to whatever is loaded when it runs
    key_name = f"{folder.name} (temper create)"[:60]
    with _client(engine) as c:
        key = _check(c.post("/api/keys", json={"name": key_name}))
    project = starters.files(folder.name, template, language, engine["url"] + "/v1", key["key"], key_name, model)
    starters.write(folder, project)
    run = "uv run main.py" if language == "python" else "npm install && npm start"
    notes = " notes" if template == "rag" else ""
    if notes:
        run += notes if language == "python" else " -- notes"
    print(f"Made {folder} ({template}, {language}), with an API key named \"{key_name}\" in .env" + (f" and {model} as its model." if model else "."))
    print(f"Next:  cd {a.folder} && {run}")
    return 0


def devices_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper devices", description="Other computers running Temper whose models this one can use.")
    p.add_argument("--json", action="store_true", help="print JSON")
    sub = p.add_subparsers(dest="action")
    add = sub.add_parser("add", help="link a computer by its address and one of its named API keys")
    add.add_argument("url", help="its address, as its Connect page shows it (http://192.168.1.20:8765)")
    add.add_argument("--key", required=True, help="a named API key made on that computer")
    add.add_argument("--name", help="what to call it (default: its host name)")
    sub.add_parser("remove", help="unlink a computer").add_argument("name")
    prefer = sub.add_parser("prefer", help="which answers when it and this computer both have the model")
    prefer.add_argument("name", help="a device's name, or `this` for this computer")
    a = p.parse_args(args)
    with _client(running()) as c:
        listed = _check(c.get("/api/devices", params={"fresh": "true"}))

        def named(name: str) -> dict:
            found = next((d for d in listed["devices"] if d["name"].lower() == name.lower()), None)
            if found is None:
                raise CliError(f"No linked device called {name}." + (f" These are: {', '.join(d['name'] for d in listed['devices'])}." if listed["devices"] else ""))
            return found

        if a.action == "add":
            d = _check(c.post("/api/devices", json={"url": a.url, "key": a.key, "name": a.name}))
            print(f"Linked {d['name']} at {d['url']}, with {len(d['models'])} models. Ask for one as <model>@{d['name']}.")
        elif a.action == "remove":
            _check(c.delete(f"/api/devices/{named(a.name)['id']}"))
            print(f"Unlinked {a.name}.")
        elif a.action == "prefer":
            device = None if a.name.lower() == "this" else named(a.name)
            _check(c.put("/api/devices/preferred", json={"device": device and device["id"]}))
            print(f"{device['name']} answers for models both computers have." if device else "This computer answers for its own models.")
        elif a.json:
            print(json.dumps(listed, indent=2))
        elif not listed["devices"]:
            print("No devices are linked. `temper devices add <address> --key <key>` links one; make the key on that computer under Connect → API keys.")
        else:
            rows = [["DEVICE", "ADDRESS", "STATUS", "MODELS"]]
            for d in listed["devices"]:
                status = ("online" if d["online"] else d["error"]) + (", preferred" if d["preferred"] else "")
                rows.append([d["name"], d["url"], status, ", ".join(m["id"] + ("" if m["loaded"] else " (would load)") for m in d["models"]) or "-"])
            _table(rows)
    return 0


def transcribe_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper transcribe", description="Speech to text with Whisper, on this computer.")
    p.add_argument("file", help="an audio or video file (anything FFmpeg reads)")
    p.add_argument("--format", "-f", choices=["txt", "srt", "vtt", "json"], default="txt", help="what to print or write (default txt)")
    p.add_argument("--language", "-l", help="the spoken language's code, like en or de (default: detected)")
    p.add_argument("--translate", action="store_true", help="English text from speech in any language")
    p.add_argument("--output", "-o", help="write it to this file instead of printing it")
    a = p.parse_args(args)
    path = os.path.abspath(a.file)
    with _client(running()) as c:
        job = _check(c.post("/api/speech/jobs", json={"path": path, "language": a.language, "task": "translate" if a.translate else "transcribe"}))
        print(f"Transcribing {Path(path).name}…", file=sys.stderr)
        end = follow(c, job["id"], logs=False)
    if end.get("status") != "done":
        raise CliError(end.get("error") or f"It ended {end.get('status')}.")
    files = (end.get("result") or {}).get("files") or {}
    text = Path(files[a.format]).read_text(encoding="utf-8")
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8")
        print(f"Wrote {a.output}. All formats are in {Path(files['txt']).parent}.", file=sys.stderr)
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


def speech_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper speech", description="Speech to text: whether it's installed, and its model.")
    sub = p.add_subparsers(dest="action")
    sub.add_parser("install", help="install Whisper into its own environment")
    sub.add_parser("remove", help="remove it")
    use = sub.add_parser("use", help="pick the model, downloading it if needed")
    use.add_argument("model", help="a model id, or turbo, small or large")
    a = p.parse_args(args)
    with _client(running()) as c:
        st = _check(c.get("/api/speech"))
        if a.action == "install":
            if st["installed"]:
                print(f"Speech to text ({st['backend']}) is installed already.")
                return 0
            end = follow(c, _check(c.post("/api/speech/install"))["id"], logs=False)
            if end.get("status") != "done":
                raise CliError(end.get("error") or f"It ended {end.get('status')}.")
            print(f"Installed speech to text ({st['backend']}).")
        elif a.action == "remove":
            _check(c.delete("/api/speech"))
            print("Removed speech to text. Downloaded models stay; delete them on the Models page.")
        elif a.action == "use":
            words = {"turbo": "Large v3 Turbo", "small": "Small", "large": "Large v3"}
            m = next((m for m in st["models"] if a.model in (m["id"], m["id"].split("/")[-1]) or words.get(a.model.lower()) == m["name"]), None)
            if m is None:
                raise CliError(f"Pick one of: {', '.join(m['id'] for m in st['models'])} (or turbo, small, large).")
            if not m["downloaded"]:
                print(f"Downloading {m['id']} ({_size(m['size'])})…", file=sys.stderr)
                end = follow(c, _check(c.post("/api/models/download", json={"repo_id": m["id"]}))["id"], logs=False)
                if end.get("status") != "done":
                    raise CliError(end.get("error") or f"The download ended {end.get('status')}.")
            _check(c.put("/api/speech/model", json={"model": m["id"]}))
            print(f"Speech to text now uses {m['name']} ({m['id']}).")
        else:
            _table([
                ["Speech to text", f"{st['backend']}, " + ("installed" if st["installed"] else "not installed (`temper speech install`)")],
                ["Model", st["model"] + ("" if any(m["id"] == st["model"] and m["downloaded"] for m in st["models"]) else " (not downloaded: `temper speech use`)")],
            ])
            print()
            _table([["MODEL", "SIZE", ""], *[[m["id"], _size(m["size"]), "downloaded" if m["downloaded"] else ""] for m in st["models"]]])
    return 0


def say_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper say", description="Read text aloud with Kokoro, on this computer.")
    p.add_argument("text", nargs="?", help="what to say (default: read from stdin)")
    p.add_argument("--voice", "-v", help="a voice id, like af_heart or bm_george (default: the one picked on the Audio page)")
    p.add_argument("--speed", type=float, default=1.0, help="0.5 to 2 (default 1)")
    p.add_argument("--output", "-o", help="write the audio here (.wav, .mp3, .opus, .flac or .aac) instead of playing it")
    p.add_argument("--voices", action="store_true", help="list the voices")
    p.add_argument("--install", action="store_true", help="install text to speech and download Kokoro")
    a = p.parse_args(args)
    with _client(running()) as c:
        if a.install:
            st = _check(c.get("/api/tts"))
            steps = [(not st["installed"], "/api/tts/install", None, "text to speech"), (not st["downloaded"], "/api/models/download", {"repo_id": st["repo"]}, f"Kokoro ({_size(st['size'])})")]
            for needed, path, body, what in steps:
                if needed:
                    print(f"Installing {what}…", file=sys.stderr)
                    end = follow(c, _check(c.post(path, json=body))["id"], logs=False)
                    if end.get("status") != "done":
                        raise CliError(end.get("error") or f"It ended {end.get('status')}.")
            print('Text to speech is ready. Try: temper say "Hello there"')
            return 0
        if a.voices:
            listed = _check(c.get("/api/tts/voices"))
            _table([["VOICE", "NAME", "LANGUAGE", ""], *[[v["id"], v["name"], v["language"], "picked" if v["id"] == listed["voice"] else ""] for v in listed["voices"]]])
            return 0
        text = a.text if a.text is not None else sys.stdin.read()
        if a.output:
            fmt = Path(a.output).suffix.lstrip(".").lower() or "wav"
            fmt = {"ogg": "opus", "m4a": "aac"}.get(fmt, fmt)
            r = c.post("/v1/audio/speech", json={"input": text, "voice": a.voice or "", "speed": a.speed, "response_format": fmt})
            if r.status_code != 200:
                _check(r)
            Path(a.output).write_bytes(r.content)
            print(f"Wrote {a.output} ({r.headers.get('x-duration', '?')} s).", file=sys.stderr)
            return 0
        r = c.post("/api/tts/speak", json={"text": text, "voice": a.voice, "speed": a.speed})
        if r.status_code != 200:
            _check(r)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(r.content)
    if sys.platform == "darwin":
        player = ["afplay", f.name]
    elif os.name == "nt":
        player = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{f.name}').PlaySync()"]
    else:
        player = ["aplay", "-q", f.name]
    try:
        subprocess.run(player, check=False)
    except FileNotFoundError:
        raise CliError(f"Nothing here plays audio ({player[0]}); use -o to save it to a file.") from None
    finally:
        os.unlink(f.name)
    return 0


def stems_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper stems", description="Split a song into vocals, drums, bass and the rest with Demucs, on this computer.")
    p.add_argument("file", nargs="?", help="an audio or video file (anything FFmpeg reads)")
    p.add_argument("--output", "-o", help="copy the stems into this folder too")
    p.add_argument("--install", action="store_true", help="install stem separation and download its model")
    a = p.parse_args(args)
    with _client(running()) as c:
        if a.install:
            st = _check(c.get("/api/stems"))
            steps = [(not st["installed"], "/api/stems/install", None, "stem separation (PyTorch and Demucs)"), (not st["downloaded"], "/api/models/download", {"repo_id": st["repo"]}, f"Demucs's model ({_size(st['size'])})")]
            for needed, path, body, what in steps:
                if needed:
                    print(f"Installing {what}…", file=sys.stderr)
                    end = follow(c, _check(c.post(path, json=body))["id"], logs=False)
                    if end.get("status") != "done":
                        raise CliError(end.get("error") or f"It ended {end.get('status')}.")
            print("Stem separation is ready. Try: temper stems song.mp3")
            return 0
        if not a.file:
            p.error("give a file to split, or --install")
        job = _check(c.post("/api/stems/jobs", json={"path": os.path.abspath(a.file)}))
        end = follow(c, job["id"], logs=False)
    if end.get("status") != "done":
        raise CliError(end.get("error") or f"It ended {end.get('status')}.")
    result = end.get("result") or {}
    stems = result.get("stems") or {}
    if a.output:
        Path(a.output).mkdir(parents=True, exist_ok=True)
        for path in stems.values():
            shutil.copy2(path, a.output)
    where = a.output or result.get("folder")
    _table([[name, str(Path(where) / Path(path).name)] for name, path in stems.items()])
    return 0


def status_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper status", description="Whether Temper is running, and what it's doing.")
    p.add_argument("--json", action="store_true", help="print JSON")
    a = p.parse_args(args)
    try:
        engine = running()
    except CliError as e:
        print(json.dumps({"running": False, "detail": str(e)}) if a.json else e)
        return 1
    with _client(engine) as c:
        out = {
            "running": True,
            "url": engine["url"],
            "version": _check(c.get("/api/health")).get("version"),
            "loaded": _check(c.get("/api/runtime/loaded")),
            "jobs": [j for j in _check(c.get("/api/jobs")) if j["status"] == "running"],
            "lan": _check(c.get("/api/lan")),
            "discovery": _check(c.get("/api/discovery")),
            "storage": _check(c.get("/api/storage")),
        }
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    loaded = out["loaded"]
    models = ", ".join(f"{m.get('model')} ({'chat' if m.get('chat') else 'on demand' if m.get('on_demand') else 'alongside'})" for m in loaded["models"])
    jobs = ", ".join(j["title"] + (f" {j['progress']:.0%}" if j.get("progress") is not None else "") for j in out["jobs"])
    lan, disc, disk = out["lan"], out["discovery"], out["storage"]
    reach = ["this computer"] + ([f"the network at {lan['url']}"] if lan.get("running") and lan.get("url") else [])
    also = [f"{name}'s address ({d['port']})" for name, d in disc.items() if d.get("running")]
    _table([
        ["Temper", f"{out['version']}, running at {out['url']}/v1, reachable from {' and '.join(reach)}" + (f"; also answering at {', '.join(also)}" if also else "")],
        ["Models", models or ("none loaded; requests that name a model load it" if loaded.get("load_on_demand") else "none loaded")],
        ["Jobs", jobs or "none running"],
        ["Disk", f"Temper uses {_size(sum(disk['usage'].values()))}; {_size(disk['disk']['free'])} free"],
    ])
    return 0


def version_cmd(args: list[str]) -> int:
    from . import __version__

    p = argparse.ArgumentParser(prog="temper version", description="Temper's version.")
    p.add_argument("--json", action="store_true", help="print JSON")
    a = p.parse_args(args)
    try:
        engine = _check(httpx.get(f"{running()['url']}/api/health", timeout=3)).get("version")
    except (CliError, httpx.HTTPError):
        engine = None  # not running: this copy's version is all there is to say
    if a.json:
        print(json.dumps({"version": __version__, "running_engine": engine}))
    else:
        print(f"temper {__version__}" + (f" (the running engine is {engine})" if engine and engine != __version__ else ""))
    return 0


def serve_cmd(args: list[str]) -> int:
    from .__main__ import run_engine

    p = argparse.ArgumentParser(prog="temper serve", description="Run Temper's engine without the app, for a server or CI.")
    p.add_argument("--host", default="127.0.0.1", help="the address to listen on (default 127.0.0.1, this computer only)")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(args)
    try:
        engine = running()
    except CliError:
        engine = None
    if engine:
        raise CliError(f"Temper is already running at {engine['url']}.")
    key = stable_key()
    print(f"Temper's engine: http://{a.host}:{a.port}  (key in {home() / 'api-key'}; Ctrl-C stops it)", file=sys.stderr)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("It's reachable from other computers: anyone with the key can use and manage it. Prefer named API keys for apps.", file=sys.stderr)
    run_engine(a.host, a.port, key)
    return 0


COMMANDS = {"models": models_cmd, "load": load_cmd, "ps": ps_cmd, "unload": unload_cmd, "status": status_cmd, "version": version_cmd, "get": get_cmd, "import": import_cmd, "log": log_cmd, "daemon": daemon_cmd, "engines": engines_cmd, "create": create_cmd, "devices": devices_cmd, "transcribe": transcribe_cmd, "speech": speech_cmd, "say": say_cmd, "stems": stems_cmd, "chat": chat_cmd, "train": train_cmd, "runs": runs_cmd, "export": export_cmd, "serve": serve_cmd}


USAGE = """usage:
  temper models [--json]
  temper load <model or path> [--quant Q] [--context N] [--alongside] [--ttl S] [--alias NAME] [--gpu-layers N] [--estimate-only] [-y]
  temper ps [--json]
  temper unload <model> | --all
  temper get <search words | owner/name> [--quant Q] [--mlx] [-y]
  temper import <file or folder> [--copy | --move]
  temper status [--json]
  temper log [--content] [--json] [-n N]
  temper version
  temper chat ["prompt"] [--system "..."] [--model M]
  temper train <config.yaml> [--set key=value ...] [--dry-run]
  temper runs [--json]
  temper export <run> [--format gguf] [--quant q4_k_m] [--imatrix]
  temper serve [--host H] [--port N]
  temper daemon up|down|status|install|uninstall
  temper create <folder> [--template chat|tools|rag] [--lang python|typescript] [--model M]
  temper transcribe <file> [--format txt|srt|vtt|json] [--language xx] [--translate] [-o out]
  temper say ["text"] [--voice V] [--speed S] [-o out.mp3] | --voices | --install
  temper stems <file> [-o folder] | --install
  temper speech [install | remove | use turbo|small|large]
  temper devices [--json] | add <address> --key K [--name N] | remove <name> | prefer <name>|this
  temper engines [--json] | use <engine>|auto | install|remove <engine> | update | survey
  temper start claude|codex|opencode|aider [agent arguments]
  temper mcp add claude|codex|cursor|vscode
  temper doctor --mcp"""


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        if argv[:1] and argv[0] in COMMANDS:
            return COMMANDS[argv[0]](argv[1:])
        if argv[:1] == ["start"] and len(argv) >= 2:
            return start(argv[1], argv[2:])
        if argv[:2] == ["mcp", "add"] and len(argv) == 3:
            print(add(argv[2]))
            return 0
        if argv[:2] == ["doctor", "--mcp"]:
            return doctor_mcp()
    except CliError as e:
        print(e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    print(USAGE, file=sys.stderr)
    return 2
