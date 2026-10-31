"""Starter projects for `temper create`: a small app wired to Temper's API, in Python or TypeScript.

Each one talks to Temper through the OpenAI SDK, reading the address, a named API key and the model from
.env (kept out of git). "chat" is a streaming conversation, "tools" lets the model call two functions, and
"rag" answers questions from a folder of notes, picking the passages to quote by keyword overlap so it
needs no embedding model.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES = {
    "chat": "a streaming chat in the terminal",
    "tools": "the model calls your functions (the time in a city, the files in a folder)",
    "rag": "answers questions from a folder of notes, quoting where it found them",
}
LANGUAGES = ("python", "typescript")

# --- Python ---------------------------------------------------------------------------------------

PY_PROJECT = """[project]
name = "__SLUG__"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["openai>=1.40", "python-dotenv>=1.0"]
"""

PY_CLIENT = '''import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.environ["TEMPER_BASE_URL"], api_key=os.environ["TEMPER_API_KEY"])
# A model you've downloaded loads on request; any other name goes to the model loaded in Temper.
MODEL = os.environ.get("TEMPER_MODEL") or "local"
'''

PY_MAIN = {
    "chat": '''"""__NAME__: a chat with a model running in Temper."""

from temper import MODEL, client

messages = [{"role": "system", "content": "You are a helpful assistant. Keep answers short."}]
print(f"Chatting with {MODEL} through Temper. An empty line quits.")
while True:
    try:
        text = input("> ").strip()
    except EOFError:
        break
    if not text:
        break
    messages.append({"role": "user", "content": text})
    reply = ""
    for chunk in client.chat.completions.create(model=MODEL, messages=messages, stream=True):
        piece = chunk.choices[0].delta.content if chunk.choices else None
        if piece:
            reply += piece
            print(piece, end="", flush=True)
    print()
    messages.append({"role": "assistant", "content": reply})
''',
    "tools": '''"""__NAME__: a model running in Temper that calls Python functions."""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from temper import MODEL, client


def current_time(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%A %H:%M")


def list_files(folder: str = ".") -> list[str]:
    return sorted(os.listdir(folder))[:50]


FUNCTIONS = {"current_time": current_time, "list_files": list_files}
TOOLS = [
    {"type": "function", "function": {"name": "current_time", "description": "The local time in a time zone.",
     "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "An IANA zone, like Europe/London"}}, "required": ["timezone"]}}},
    {"type": "function", "function": {"name": "list_files", "description": "The names of the files in a folder on this computer.",
     "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}}}},
]


def ask(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(5):  # a few rounds of calls, then it has to answer
        reply = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS).choices[0].message
        if not reply.tool_calls:
            return reply.content or ""
        messages.append(reply.model_dump(exclude_none=True))
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [{call.function.name}({args})]")
            try:
                result = FUNCTIONS[call.function.name](**args)
            except Exception as e:  # the model sees the error and can try again
                result = f"error: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return "(it kept calling tools without answering)"


print(f"Ask {MODEL} something that needs a tool, like \\"What time is it in Tokyo?\\". An empty line quits.")
while (question := input("> ").strip()):
    print(ask(question))
''',
    "rag": '''"""__NAME__: answers questions from a folder of notes with a model running in Temper."""

import math
import re
import sys
from collections import Counter
from pathlib import Path

from temper import MODEL, client

FOLDER = Path(sys.argv[1] if len(sys.argv) > 1 else "notes")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def passages(folder: Path, size: int = 800) -> list[tuple[str, str]]:
    """(file, passage) pairs: each .md or .txt file split at blank lines into pieces of about `size` characters."""
    out = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        piece = ""
        for para in re.split(r"\\n\\s*\\n", path.read_text(errors="replace")):
            if piece and len(piece) + len(para) > size:
                out.append((str(path.relative_to(folder)), piece.strip()))
                piece = ""
            piece += para + "\\n\\n"
        if piece.strip():
            out.append((str(path.relative_to(folder)), piece.strip()))
    return out


def best(question: str, found: list[tuple[str, str]], k: int = 4) -> list[tuple[str, str]]:
    """The passages sharing the most (and the rarest) words with the question."""
    counts = [Counter(words(text)) for _, text in found]
    seen_in = Counter(w for c in counts for w in c)
    def score(c: Counter) -> float:
        return sum(math.log(1 + len(found) / seen_in[w]) * (1 + math.log(c[w])) for w in set(words(question)) if w in c)
    ranked = sorted(range(len(found)), key=lambda i: score(counts[i]), reverse=True)
    return [found[i] for i in ranked[:k] if score(counts[i]) > 0]


found = passages(FOLDER)
print(f"Read {len(found)} passages from {FOLDER}/. Ask about them; an empty line quits.")
while (question := input("> ").strip()):
    quotes = best(question, found)
    context = "\\n\\n".join(f"[{name}]\\n{text}" for name, text in quotes) or "(nothing in the notes matches)"
    messages = [
        {"role": "system", "content": "Answer from the notes below, and name the file each fact comes from in brackets. "
                                      "If the notes don't say, say so.\\n\\n" + context},
        {"role": "user", "content": question},
    ]
    for chunk in client.chat.completions.create(model=MODEL, messages=messages, stream=True):
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
''',
}

# --- TypeScript -----------------------------------------------------------------------------------

TS_PACKAGE = """{
  "name": "__SLUG__",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": { "start": "tsx --env-file=.env main.ts" },
  "dependencies": { "openai": "^4.77.0" },
  "devDependencies": { "tsx": "^4.19.0", "typescript": "^5.7.0", "@types/node": "^22.10.0" }
}
"""

TS_CONFIG = """{
  "compilerOptions": { "target": "ES2022", "module": "NodeNext", "moduleResolution": "NodeNext", "strict": true, "skipLibCheck": true, "noEmit": true }
}
"""

TS_CLIENT = """import OpenAI from "openai";

export const client = new OpenAI({ baseURL: process.env.TEMPER_BASE_URL, apiKey: process.env.TEMPER_API_KEY });
// A model you've downloaded loads on request; any other name goes to the model loaded in Temper.
export const MODEL = process.env.TEMPER_MODEL || "local";
"""

TS_MAIN = {
    "chat": """// __NAME__: a chat with a model running in Temper.
import * as readline from "node:readline/promises";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { client, MODEL } from "./temper.js";

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const messages: ChatCompletionMessageParam[] = [{ role: "system", content: "You are a helpful assistant. Keep answers short." }];
console.log(`Chatting with ${MODEL} through Temper. An empty line quits.`);
rl.setPrompt("> ");
rl.prompt();
for await (const line of rl) {
  const text = line.trim();
  if (!text) break;
  messages.push({ role: "user", content: text });
  let reply = "";
  for await (const chunk of await client.chat.completions.create({ model: MODEL, messages, stream: true })) {
    const piece = chunk.choices[0]?.delta?.content;
    if (piece) {
      reply += piece;
      process.stdout.write(piece);
    }
  }
  console.log();
  messages.push({ role: "assistant", content: reply });
  rl.prompt();
}
rl.close();
""",
    "tools": """// __NAME__: a model running in Temper that calls TypeScript functions.
import { readdirSync } from "node:fs";
import * as readline from "node:readline/promises";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "openai/resources/chat/completions";
import { client, MODEL } from "./temper.js";

const functions: Record<string, (args: any) => unknown> = {
  current_time: ({ timezone }) => new Date().toLocaleString("en-GB", { timeZone: timezone, weekday: "long", hour: "2-digit", minute: "2-digit" }),
  list_files: ({ folder = "." }) => readdirSync(folder).sort().slice(0, 50),
};
const tools: ChatCompletionTool[] = [
  { type: "function", function: { name: "current_time", description: "The local time in a time zone.",
    parameters: { type: "object", properties: { timezone: { type: "string", description: "An IANA zone, like Europe/London" } }, required: ["timezone"] } } },
  { type: "function", function: { name: "list_files", description: "The names of the files in a folder on this computer.",
    parameters: { type: "object", properties: { folder: { type: "string" } } } } },
];

async function ask(question: string): Promise<string> {
  const messages: ChatCompletionMessageParam[] = [{ role: "user", content: question }];
  for (let round = 0; round < 5; round++) { // a few rounds of calls, then it has to answer
    const reply = (await client.chat.completions.create({ model: MODEL, messages, tools })).choices[0].message;
    if (!reply.tool_calls?.length) return reply.content ?? "";
    messages.push(reply);
    for (const call of reply.tool_calls) {
      const args = JSON.parse(call.function.arguments || "{}");
      console.log(`  [${call.function.name}(${JSON.stringify(args)})]`);
      let result: unknown;
      try {
        result = functions[call.function.name](args);
      } catch (e) { // the model sees the error and can try again
        result = `error: ${(e as Error).message}`;
      }
      messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
    }
  }
  return "(it kept calling tools without answering)";
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
console.log(`Ask ${MODEL} something that needs a tool, like "What time is it in Tokyo?". An empty line quits.`);
rl.setPrompt("> ");
rl.prompt();
for await (const line of rl) {
  if (!line.trim()) break;
  console.log(await ask(line.trim()));
  rl.prompt();
}
rl.close();
""",
    "rag": """// __NAME__: answers questions from a folder of notes with a model running in Temper.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, relative } from "node:path";
import * as readline from "node:readline/promises";
import { client, MODEL } from "./temper.js";

const folder = process.argv[2] ?? "notes";
const words = (text: string) => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

/** [file, passage] pairs: each .md or .txt file split at blank lines into pieces of about `size` characters. */
function passages(dir: string, size = 800): [string, string][] {
  const out: [string, string][] = [];
  for (const name of readdirSync(dir).sort()) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      out.push(...passages(path, size).map(([f, t]) => [join(name, f), t] as [string, string]));
      continue;
    }
    if (![".md", ".txt"].includes(extname(name).toLowerCase())) continue;
    let piece = "";
    for (const para of readFileSync(path, "utf8").split(/\\n\\s*\\n/)) {
      if (piece && piece.length + para.length > size) {
        out.push([relative(dir, path), piece.trim()]);
        piece = "";
      }
      piece += para + "\\n\\n";
    }
    if (piece.trim()) out.push([relative(dir, path), piece.trim()]);
  }
  return out;
}

const found = passages(folder);
const counts = found.map(([, text]) => words(text).reduce((m, w) => m.set(w, (m.get(w) ?? 0) + 1), new Map<string, number>()));
const seenIn = new Map<string, number>();
for (const c of counts) for (const w of c.keys()) seenIn.set(w, (seenIn.get(w) ?? 0) + 1);

/** The passages sharing the most (and the rarest) words with the question. */
function best(question: string, k = 4): [string, string][] {
  const asked = new Set(words(question));
  const score = (c: Map<string, number>) =>
    [...asked].filter((w) => c.has(w)).reduce((s, w) => s + Math.log(1 + found.length / seenIn.get(w)!) * (1 + Math.log(c.get(w)!)), 0);
  return found.map((p, i) => [p, score(counts[i])] as const).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).slice(0, k).map(([p]) => p);
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
console.log(`Read ${found.length} passages from ${folder}/. Ask about them; an empty line quits.`);
rl.setPrompt("> ");
rl.prompt();
for await (const line of rl) {
  const q = line.trim();
  if (!q) break;
  const context = best(q).map(([name, text]) => `[${name}]\\n${text}`).join("\\n\\n") || "(nothing in the notes matches)";
  const stream = await client.chat.completions.create({
    model: MODEL,
    stream: true,
    messages: [
      { role: "system", content: "Answer from the notes below, and name the file each fact comes from in brackets. If the notes don't say, say so.\\n\\n" + context },
      { role: "user", content: q },
    ],
  });
  for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
  console.log();
  rl.prompt();
}
rl.close();
""",
}

NOTES = """# About these notes

This folder is what the app answers from. Put your own Markdown or text files here (folders inside are read
too), and ask about them. Each answer names the file it quotes.

The app splits each file at blank lines into passages, and sends the model the few passages that share the
most words with your question. Rare words count for more than common ones.
"""

README = """# __NAME__

__ABOUT__, using a model running on this computer in [Temper](https://github.com/devops-monk/temper).

## Run it

Temper needs to be open (or `temper daemon up`), with a model loaded or downloaded.

```bash
__RUN__
```

## Settings

`.env` holds Temper's address, an API key named "__KEY_NAME__", and the model to use. It's in `.gitignore`;
don't commit it. Revoke the key in Temper under Connect → API keys when you no longer need it.

- `TEMPER_MODEL`: a model you've downloaded loads when the app asks for it (`temper models` lists them). Any
  other name goes to the model loaded in Temper.
- From another computer: turn on *Serve the model API on this network* in Temper and put its address in
  `TEMPER_BASE_URL`.

The app uses the OpenAI SDK, so it works with any OpenAI-compatible server by changing `.env`.
"""

RUN = {
    "python": "uv run main.py__ARGS__        # or: pip install openai python-dotenv && python main.py__ARGS__",
    "typescript": "npm install\nnpm start__TS_ARGS__",
}


def files(name: str, template: str, language: str, base_url: str, key: str, key_name: str, model: str) -> dict[str, str]:
    """The project's files, by path."""
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-") or "temper-app"
    args = " notes" if template == "rag" else ""
    if language == "python":
        out = {"pyproject.toml": PY_PROJECT, "temper.py": PY_CLIENT, "main.py": PY_MAIN[template], ".gitignore": ".env\n.venv/\n__pycache__/\n"}
    else:
        out = {"package.json": TS_PACKAGE, "tsconfig.json": TS_CONFIG, "temper.ts": TS_CLIENT, "main.ts": TS_MAIN[template], ".gitignore": ".env\nnode_modules/\n"}
    if template == "rag":
        out["notes/about.md"] = NOTES
    run = RUN[language].replace("__ARGS__", args).replace("__TS_ARGS__", " -- notes" if args else "")
    about = TEMPLATES[template][0].upper() + TEMPLATES[template][1:]
    out["README.md"] = README.replace("__ABOUT__", about).replace("__RUN__", run).replace("__KEY_NAME__", key_name)
    out[".env"] = f"TEMPER_BASE_URL={base_url}\nTEMPER_API_KEY={key}\nTEMPER_MODEL={model}\n"
    return {path: text.replace("__NAME__", name).replace("__SLUG__", slug) for path, text in out.items()}


def write(folder: Path, project: dict[str, str]) -> None:
    for path, text in project.items():
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (folder / ".env").chmod(0o600)
