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
import shutil
import subprocess
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
        models = [m["id"] for m in _check(c.get("/v1/models")).get("data") or []]
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
    with _client(running()) as c:
        models = _check(c.get("/api/models/local"))
    if not models:
        print("No models yet. Download one on the Models page.")
    for m in models:
        print(f"{m['id']}  {_size(m['size'])}" + ("  (missing)" if m.get("missing") else ""))
        for g in m.get("gguf") or []:
            print(f"    {g['name']}  {_size(g['size'])}")
    return 0


def resolve_model(models: list[dict], wanted: str, quant: str | None = None) -> tuple[str, str]:
    """The path to load and its name, from what was typed: a path, a GGUF's file name, or a model's name
    (its Q4_K_M file when it has GGUFs, unless `quant` says which)."""
    if Path(wanted).expanduser().exists():
        path = Path(wanted).expanduser()
        return str(path), path.name
    for m in models:
        for g in m.get("gguf") or []:
            if g["name"].lower() == wanted.lower():
                return g["path"], g["name"]
    w = wanted.lower()
    model = next((m for m in models if m["id"].lower() == w), None) or next((m for m in models if w in m["id"].lower()), None)
    if model is None:
        raise CliError(f"No model called {wanted!r} on this computer. `temper models` lists them.")
    ggufs = [g for g in model.get("gguf") or [] if g.get("complete", True)]
    if quant:
        ggufs = [g for g in ggufs if quant.lower() in g["name"].lower()]
        if not ggufs:
            raise CliError(f"{model['id']} has no {quant} file. `temper models` lists its files.")
    if ggufs:
        g = next((g for g in ggufs if "q4_k_m" in g["name"].lower()), ggufs[0])
        return g["path"], g["name"]
    if model.get("safetensors") and model.get("path"):
        return model["path"], model["id"]
    raise CliError(f"{model['id']} has nothing Temper can load.")


def load_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper load", description="Load a model for chat and the API.")
    p.add_argument("model", help="a model's name, a GGUF file name, or a path")
    p.add_argument("--quant", help="which of the model's GGUF files, like Q8_0")
    p.add_argument("--context", type=int, help="context length in tokens")
    a = p.parse_args(args)
    with _client(running()) as c:
        path, name = resolve_model(_check(c.get("/api/models/local")), a.model, a.quant)
        _check(c.post("/api/runtime/load", json={"path": path, "name": name, **({"context": a.context} if a.context else {})}))
        print(f"Loading {name}", end="", flush=True, file=sys.stderr)
        while (state := _check(c.get("/api/runtime")))["state"] == "loading":
            print(".", end="", flush=True, file=sys.stderr)
            time.sleep(1)
        print(file=sys.stderr)
    if state["state"] != "ready":
        raise CliError(state.get("error") or "The model didn't load.")
    print(f"Loaded {state['model']}" + (f" with {state['context']} tokens of context" if state.get("context") else ""))
    return 0


def _reply(c: httpx.Client, messages: list[dict]) -> str:
    """Streams one reply to stdout, leaving out a thinking model's reasoning, and returns it."""
    from .anthropic_api import ThinkFilter

    think, text = ThinkFilter(), ""
    with c.stream("POST", "/v1/chat/completions", json={"messages": messages, "stream": True}) as r:
        if r.status_code >= 400:
            r.read()
            _check(r)
        for line in r.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                continue
            part = think.feed(delta.get("content") or "")
            if not text:
                part = part.lstrip()  # what a thinking block leaves before the answer
            if part:
                text += part
                print(part, end="", flush=True)
    if rest := (think.flush() if text else think.flush().lstrip()):
        text += rest
        print(rest, end="")
    print()
    return text


def chat_cmd(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="temper chat", description="Ask the loaded model once, or chat with it until /bye.")
    p.add_argument("prompt", nargs="*", help="ask this once and print the reply (or pipe it in)")
    p.add_argument("--system", help="a system prompt")
    a = p.parse_args(args)
    engine = running()
    loaded_model(engine)
    messages = [{"role": "system", "content": a.system}] if a.system else []
    with _client(engine) as c:
        prompt = " ".join(a.prompt) or (sys.stdin.read() if not sys.stdin.isatty() else "")
        if prompt.strip():
            _reply(c, [*messages, {"role": "user", "content": prompt}])
            return 0
        print("Chatting with the model loaded in Temper. /clear starts over, /bye (or Ctrl-D) ends.", file=sys.stderr)
        history = list(messages)
        while True:
            try:
                line = input("> ")
            except EOFError:
                print(file=sys.stderr)
                return 0
            if line.strip() in ("/bye", "/exit", "/quit"):
                return 0
            if line.strip() == "/clear":
                history = list(messages)
                continue
            if not line.strip():
                continue
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": _reply(c, history)})


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
                elif kind == "error":
                    print(ev.get("message", ""), file=sys.stderr)
                elif kind == "status":
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
    with _client(running()) as c:
        runs = _check(c.get("/api/train/runs"))
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


COMMANDS = {"models": models_cmd, "load": load_cmd, "chat": chat_cmd, "train": train_cmd, "runs": runs_cmd, "export": export_cmd, "serve": serve_cmd}


USAGE = """usage:
  temper models
  temper load <model or path> [--quant Q] [--context N]
  temper chat ["prompt"] [--system "..."]
  temper train <config.yaml> [--set key=value ...] [--dry-run]
  temper runs
  temper export <run> [--format gguf] [--quant q4_k_m] [--imatrix]
  temper serve [--host H] [--port N]
  temper start claude|codex|opencode|aider [agent arguments]
  temper mcp add claude|codex|cursor|vscode
  temper doctor --mcp"""


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        if argv[:1] in (["models"], ["load"], ["chat"], ["train"], ["runs"], ["export"], ["serve"]):
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
