"""Tools a chat model can call, and the repairs that make small models' tool calls usable.

A chat turns tools on by name. With llama.cpp the tools go to llama-server natively (it renders them
with the model's chat template, or a generic JSON format for templates without tools, and parses the
calls). Other engines get them in the system prompt instead, with a <tool_call> format to write, and
Temper parses that from the reply text.

Models get tool calls wrong in a few common ways, which Temper repairs rather than failing on:
arguments that aren't quite JSON (single quotes, trailing commas, a missing brace, a JSON string
inside a string, code fences), calls written into the reply text in some model family's own format
(<tool_call>, <|python_tag|>, [TOOL_CALLS], <function=…>, a bare JSON object) instead of as a tool
call, and tool names that are nearly right. What can't be repaired goes back to the model as the
tool's result, so it can try again.

A reply's tool steps are kept with it as [{id, name, arguments, result, error, at}], where `at` is
the offset in the reply's text where the step happened; `expand` turns them back into the messages
the model sees on later turns.
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import operator
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter

router = APIRouter(prefix="/api/tools", tags=["chat"])

# Rounds of tool calls in one reply; the round after the last is offered no tools, so the model answers.
MAX_ROUNDS = 8
MAX_RESULT = 16_000  # characters of a tool's result sent back to the model


@dataclass
class Tool:
    name: str
    label: str
    description: str
    parameters: dict
    run: Callable[[dict, dict], str]  # (arguments, context: {"chat": id}) -> result
    # Whether a call runs straight away ("auto") or waits for the person to allow it ("ask"), until they choose.
    mode: str = "auto"
    available: Callable[[], bool] = lambda: True


# --- Built-in tools -------------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {k: getattr(math, k) for k in ("pi", "e", "tau", "inf")}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log10", "log2", "exp", "floor", "ceil", "factorial", "radians", "degrees")} | {"abs": abs, "round": round, "min": min, "max": max}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("That power is too large to work out.")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > 1000:
            raise ValueError("That factorial is too large to work out.")
        return _FUNCS[node.func.id](*args)
    raise ValueError(f"The calculator can't evaluate {ast.unparse(node)!r}; use numbers, + - * / // % **, and functions like sqrt or log.")


def calculator(args: dict, ctx: dict | None = None) -> str:
    expression = str(args.get("expression", "")).replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    if not expression:
        raise ValueError("Give an expression to work out, like 12 * (3 + 4).")
    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except SyntaxError:
        raise ValueError(f"{expression!r} isn't an arithmetic expression.") from None
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return f"{expression} = {value}"


def current_time(args: dict, ctx: dict | None = None) -> str:
    zone = args.get("timezone")
    try:
        now = datetime.now(ZoneInfo(zone)) if zone else datetime.now().astimezone()
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"{zone!r} isn't a time zone name; use one like Europe/London or Asia/Kolkata.") from None
    return now.strftime("%A %d %B %Y, %H:%M:%S ") + (zone or now.tzname() or "local time") + f" (UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})"


def _code(language: str) -> Callable[[dict, dict], str]:
    def run(args: dict, ctx: dict) -> str:
        from . import sandbox

        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Give the code to run as a string in `code`.")
        return sandbox.run(language, code, ctx.get("chat"))

    return run


def _node() -> bool:
    from . import sandbox

    return sandbox.node() is not None


_CODE_NOTE = (
    " It runs on this computer in a sandbox with no network, in a working folder kept for this chat (files you write there"
    " are there next time). It returns what the code prints, so print the results you need. Runs stop after 30 seconds."
)

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            "calculator",
            "Calculator",
            "Works out an arithmetic expression exactly. Use it for any sum you would otherwise do in your head.",
            {"type": "object", "properties": {"expression": {"type": "string", "description": "The expression, for example (17.5 * 12) / 3 or sqrt(2) ** 3"}}, "required": ["expression"]},
            calculator,
        ),
        Tool(
            "current_time",
            "Date and time",
            "Gives the current date and time, on this computer's clock or in a named time zone.",
            {"type": "object", "properties": {"timezone": {"type": "string", "description": "An IANA time zone such as America/New_York; leave out for local time"}}},
            current_time,
        ),
        Tool(
            "run_python",
            "Python",
            "Runs Python 3 code and returns its output. Use it for calculations, data work and checking code." + _CODE_NOTE,
            {"type": "object", "properties": {"code": {"type": "string", "description": "The Python program to run"}}, "required": ["code"]},
            _code("python"),
            mode="ask",
        ),
        Tool(
            "run_javascript",
            "JavaScript",
            "Runs JavaScript (Node.js, as an ES module) and returns its output." + _CODE_NOTE,
            {"type": "object", "properties": {"code": {"type": "string", "description": "The JavaScript program to run"}}, "required": ["code"]},
            _code("javascript"),
            mode="ask",
            available=_node,
        ),
    )
}


def available() -> list[Tool]:
    return [t for t in TOOLS.values() if t.available()]


def schemas(names: list[str]) -> list[dict]:
    """The OpenAI `tools` list for the named tools."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for n in names if (t := TOOLS.get(n))]


def prompt(names: list[str]) -> str:
    """The system prompt text that offers the tools to a model that gets them as text rather than natively."""
    offered = [t for n in names if (t := TOOLS.get(n))]
    if not offered:
        return ""
    lines = "\n".join(f"- {t.name}: {t.description} Arguments (JSON schema): {json.dumps(t.parameters)}" for t in offered)
    return (
        "You can call tools. To call one, write only this, on its own line, then stop:\n"
        '<tool_call>{"name": "<tool name>", "arguments": {<arguments>}}</tool_call>\n'
        "The result comes back in a <tool_response> message, and you continue from there. "
        "Call a tool when it helps answer; otherwise answer directly.\n\nTools:\n" + lines
    )


# --- Repairs --------------------------------------------------------------------------------------


def _balance(text: str) -> str:
    """Closes brackets and a string left open, as a reply cut off mid-call leaves them."""
    stack, quoted, escaped = [], False, False
    for ch in text:
        if quoted:
            escaped, quoted = (False, quoted) if escaped else (ch == "\\", ch != '"')
        elif ch == '"':
            quoted = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return text + ('"' if quoted else "") + "".join(reversed(stack))


def parse_arguments(raw: object) -> dict | None:
    """A tool call's arguments as a dict, repairing near-JSON; None when they can't be read."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    candidates = [text, re.sub(r",\s*([}\]])", r"\1", text)]
    candidates.append(_balance(candidates[-1]))
    for c in candidates:
        try:
            value = json.loads(c)
        except ValueError:
            try:  # Python-style: single quotes, True/None
                value = ast.literal_eval(c)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if isinstance(value, str):  # a JSON object encoded twice
            return parse_arguments(value) if value.strip().startswith("{") else None
        if isinstance(value, dict):
            return value
    return None


def resolve_name(name: str, known: list[str]) -> str | None:
    """The known tool a model meant: exact, then ignoring case and a "functions." prefix, then a near spelling."""
    if name in known:
        return name
    plain = re.sub(r"^(functions|tools)\.", "", name.strip()).lower().replace("-", "_").replace(" ", "_")
    for k in known:
        if k.lower() == plain:
            return k
    close = difflib.get_close_matches(plain, known, n=1, cutoff=0.75)
    return close[0] if close else None


@dataclass
class Call:
    name: str
    arguments: object  # a dict, or the raw text the model wrote
    id: str = ""


# A call written into the reply text, in each model family's format; the JSON is the group.
_TEXT_CALLS = [
    re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S),  # Hermes, Qwen
    re.compile(r"<\|python_tag\|>\s*(.*?)\s*(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.S),  # Llama 3.1
    re.compile(r"\[TOOL_CALLS\]\s*(.*?)\s*(?:</s>|$)", re.S),  # Mistral
    re.compile(r"<function=([\w.-]+)>\s*(.*?)\s*(?:</function>|$)", re.S),  # Llama 3 custom
    re.compile(r"```(?:json|tool_call)?\s*(\{.*?\"(?:name|function)\".*?\})\s*```", re.S),  # a fenced call
]


def _calls_in(value: object, known: list[str]) -> list[Call]:
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            return []
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") or (item.get("function") if isinstance(item.get("function"), str) else None)
        if not isinstance(name, str):
            return []
        calls.append(Call(name, fn.get("arguments", fn.get("parameters", {}))))
    return calls


def extract_calls(text: str, known: list[str]) -> tuple[list[Call], str]:
    """Tool calls a model wrote into its reply text, and the text without them."""
    for pattern in _TEXT_CALLS:
        found: list[Call] = []
        for m in pattern.finditer(text):
            if pattern.groups == 2:  # <function=name>{arguments}
                found.append(Call(m.group(1), m.group(2)))
                continue
            body = parse_arguments(m.group(1)) if m.group(1).lstrip().startswith("{") else None
            if body is None:
                try:
                    body = json.loads(_balance(m.group(1)))
                except ValueError:
                    found.append(Call("", m.group(1)))  # a call it meant to make; run() says why it couldn't
                    continue
            found += _calls_in(body, known) or [Call("", m.group(1))]
        if found:
            return found, pattern.sub("", text).strip()
    # A reply that is nothing but a JSON call to a known tool.
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        body = parse_arguments(stripped) if stripped.startswith("{") else None
        if body is None:
            try:
                body = json.loads(stripped)
            except ValueError:
                body = None
        calls = _calls_in(body, known) if body is not None else []
        if calls and all(resolve_name(c.name, known) for c in calls):
            return calls, ""
    return [], text


# --- Running --------------------------------------------------------------------------------------


def run(call: Call, known: list[str], ctx: dict | None = None) -> dict:
    """Runs a call; returns its step {id, name, arguments, result, error}. A call that can't run says why in `result`."""
    step = {"id": call.id or f"call_{uuid.uuid4().hex[:10]}", "name": call.name, "arguments": call.arguments, "result": None, "error": False}
    if not call.name:
        return {**step, "name": "?", "error": True, "result": f'The tool call couldn\'t be read: {str(call.arguments)[:300]}. Write it as <tool_call>{{"name": "<tool>", "arguments": {{...}}}}</tool_call>.'}
    name = resolve_name(call.name, known)
    if not name:
        return {**step, "error": True, "result": f"There is no tool called {call.name!r}. The tools are: {', '.join(known)}."}
    args = parse_arguments(call.arguments)
    if args is None:
        return {**step, "name": name, "error": True, "result": f"The arguments weren't valid JSON: {str(call.arguments)[:300]}. Call {name} again with a JSON object."}
    step = {**step, "name": name, "arguments": args}
    try:
        result = TOOLS[name].run(args, ctx or {})
    except Exception as e:  # noqa: BLE001 — any failure goes back to the model as the result
        return {**step, "error": True, "result": f"{name} failed: {e}"}
    if len(result) > MAX_RESULT:
        result = result[:MAX_RESULT] + "\n[The rest of the result was cut.]"
    return {**step, "result": result}


# --- Messages -------------------------------------------------------------------------------------


def round_messages(text: str, steps: list[dict], native: bool) -> list[dict]:
    """One round of a reply, its text and the tools it called, as the model sees it next round."""
    if native:
        calls = [{"id": s["id"], "type": "function", "function": {"name": s["name"], "arguments": json.dumps(s["arguments"]) if isinstance(s["arguments"], dict) else str(s["arguments"])}} for s in steps]
        return [{"role": "assistant", "content": text, "tool_calls": calls}, *({"role": "tool", "tool_call_id": s["id"], "name": s["name"], "content": s["result"] or ""} for s in steps)]
    written = "\n".join(f"<tool_call>{json.dumps({'name': s['name'], 'arguments': s['arguments']})}</tool_call>" for s in steps)
    results = "\n".join(f'<tool_response name="{s["name"]}">\n{s["result"] or ""}\n</tool_response>' for s in steps)
    return [{"role": "assistant", "content": "\n".join(t for t in (text.strip(), written) if t)}, {"role": "user", "content": results}]


def expand(messages: list[dict], native: bool) -> list[dict]:
    """Messages with each reply's saved tool steps turned back into the rounds the model saw."""
    out = []
    for m in messages:
        steps = m.get("tools") if m.get("role") == "assistant" else None
        m = {k: v for k, v in m.items() if k != "tools"}
        if not steps:
            out.append(m)
            continue
        content, start = m.get("content") or "", 0
        for at in sorted({s.get("at", 0) for s in steps}):
            out += round_messages(content[start:at], [s for s in steps if s.get("at", 0) == at], native)
            start = at
        out.append({**m, "content": content[start:]})
    return out


def flatten(messages: list[dict]) -> list[dict]:
    """Native tool rounds as plain text turns, alternating, for counting tokens or summarising."""
    out: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "tool":
            role, content = "user", f'<tool_response name="{m.get("name", "")}">\n{content}\n</tool_response>'
        elif m.get("tool_calls"):
            written = "\n".join(f"<tool_call>{json.dumps({'name': c['function']['name'], 'arguments': c['function']['arguments']})}</tool_call>" for c in m["tool_calls"])
            content = "\n".join(t for t in (content or "", written) if t)
        if out and out[-1]["role"] == role and isinstance(content, str) and isinstance(out[-1]["content"], str):
            out[-1] = {**out[-1], "content": out[-1]["content"] + "\n" + content}
        else:
            out.append({**{k: v for k, v in m.items() if k not in ("tool_calls", "tool_call_id", "name")}, "role": role, "content": content})
    return out


@router.get("")
def list_tools():
    from . import sandbox

    return {"tools": [{"name": t.name, "label": t.label, "description": t.description, "mode": t.mode} for t in available()], "sandbox": sandbox.kind()}
