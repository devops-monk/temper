"""The `temper` commands for coding agents, run from a terminal while the app is open.

    temper start codex|opencode|aider [their own arguments]   start an agent on the model loaded in Temper
    temper mcp add claude|codex|cursor|vscode                  register Temper's MCP server with an agent
    temper doctor --mcp                                        check that server, and which agents have it

The running engine leaves its address and key in engine.json in Temper's home (readable only by
this user), which is how these find it. The key stays the same across restarts, so agent configs
written here keep working.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
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
    r = httpx.get(f"{engine['url']}/v1/models", headers={"Authorization": f"Bearer {engine['token']}"}, timeout=5)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data") or []]
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
        raise CliError("Claude Code talks the Anthropic API, which Temper doesn't serve yet. You can still give it Temper's tools: temper mcp add claude")
    raise CliError(f"Temper doesn't know how to start {agent!r}. It can start: codex, opencode, aider.")


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

USAGE = """usage:
  temper start codex|opencode|aider [agent arguments]
  temper mcp add claude|codex|cursor|vscode
  temper doctor --mcp"""


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
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
    print(USAGE, file=sys.stderr)
    return 2
