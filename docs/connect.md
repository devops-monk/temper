# Connect

Load a model, and anything that speaks the OpenAI or Anthropic API can use it. **Connect** shows ready-made
settings for each tool, with this computer's address and key filled in.

## The API

| | |
|-|-|
| Address | `http://127.0.0.1:8765/v1` (Temper keeps port 8765 when it is free) |
| OpenAI-compatible | `/v1/chat/completions`, `/v1/models`; other `/v1` paths pass through to llama.cpp's server |
| Anthropic-compatible | `/v1/messages`, `/v1/messages/count_tokens` |
| Key | a bearer token: the one Connect shows, or an API key you create |

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer $TEMPER_KEY" -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Loading models by name

With *Load models on demand* on (the default), a request whose `model` names a model you've downloaded loads
it first, alongside any others. Use the model's whole name, such as `Qwen/Qwen3-4B-GGUF`, or a GGUF's file
name. `/v1/models` lists everything that can load this way. The model unloads after the idle time in Settings, or after
the request's `ttl` in seconds:

```json
{"model": "Qwen/Qwen3-4B-GGUF", "ttl": 600, "messages": [{"role": "user", "content": "Hello"}]}
```

When memory runs short, models loaded this way make room for each other, least recently used first. Models you
loaded yourself are never unloaded to make room. Names that match nothing, like a client's default `gpt-4o`, go to
the loaded model as before.

To start an app of your own already wired to this API, run `temper create my-app` (see [the command line](cli.md#starter-projects)).

## Coding agents

```bash
temper start claude      # Claude Code on the loaded model
temper start codex       # or opencode, aider
```

`temper start` points the agent at Temper for that session only and changes none of its settings. The Connect
page also has settings to paste for Codex, OpenCode, Continue (VS Code), Aider, Cursor, the OpenAI and
Anthropic SDKs, and curl.

Agents send long prompts, so pick a model with a large context (32K or more) and enough memory to hold it.

## Tools for agents (MCP)

Temper is also an MCP server: an agent can list, load and unload your local models, ask one a question,
ask for a typed decision (below), and follow or cancel Temper's jobs (downloads, training, exports). Add it with
`temper mcp add claude` (or `codex`, `cursor`, `vscode`).

## Decisions

When your code needs a value rather than prose (is this spam, which team gets this ticket, how positive is
this review), ask for a decision. The answer is yes or no, one of a set of choices, the labels that apply,
or a whole-number score, with a one-sentence reason. The model is held to that shape where its server
supports structured output (llama.cpp, vLLM, SGLang); otherwise Temper reads the answer from the reply, and
a reply it can't read is an error, not a guess. Answers are asked for at temperature 0, so the same text
gets the same answer.

```bash
curl http://127.0.0.1:8765/v1/decide -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{
  "question": "How should support prioritise this ticket?",
  "kind": "choice", "choices": ["low", "normal", "urgent"],
  "text": "Nobody can sign in since the update."
}'
# {"kind": "choice", "answer": "urgent", "reason": "No one can sign in.", "seconds": 0.7}
```

`kind` is `yes_no` (the answer is `true` or `false`), `choice`, `labels` (a list) or `score` (with `"scale": [1, 5]`;
the default is 1 to 10). `instructions` says how to decide: definitions, examples, edge cases. `texts` in place of
`text` decides each of up to 100 texts and answers with `results`. `model` picks a model loaded alongside chat's,
or one on a linked device.

**Saved decisions** keep the question and its options under a name, so an app sends only the text:
`POST /v1/decisions/ticket-priority` with `{"text": "..."}`. Make them under *Connect → Decisions*, where you can
also try one on some text, or with `temper decide`:

```bash
temper decide "Is this a refund request?" -t "It broke, I want my money back"     # yes  (reason)
temper decide "What is it?" -c bug,feature,praise --lines < feedback.txt           # one answer per line
temper decide "How formal is it?" --score 1-5 -t "yo" --json
temper decide "Is this spam?" --save is-spam --instructions "Colleagues are never spam."
temper decide -s is-spam < message.txt
temper decide --list
```

A saved decision can also be answered by a **classifier** you trained on examples (see
[Classifiers](training.md#classifiers)): set `"classifier": "tickets"` with `kind` `choice` (its labels) or
`yes_no` (when its labels are yes and no). It answers in milliseconds, with its `confidence` and a score
for each label.

Named API keys work for decisions, since they're under `/v1`.

## API keys

The key Connect shows is this computer's own: it stays the same across restarts and opens all of Temper,
not just the model API. For other apps and devices, create a named key under **Connect → API keys**
instead. A named key opens only `/v1`, and works until you revoke it. The monitor beside the keys shows
each request, which key made it, the tokens it used and how fast the reply came. *Show prompts and replies*
adds what was asked and answered, held in memory only while it's on; `temper log` follows the same from a
terminal. Keys are stored only as hashes, so copy a new key when it is shown.

## From another device

*Serve the model API on this network* opens the API to phones and other computers on your Wi-Fi.
Only `/v1` is reachable from them, and only with an API key. Scan the QR code to copy the address.
Turn it off when you don't need it.

## Linked devices

A laptop can use a desktop's models, and its GPU. On the desktop, turn on *Serve the model API on this network*
and create an API key. On the laptop, add the desktop under **Connect → Linked devices** with its address
and that key, or from a terminal:

```bash
temper devices add 192.168.1.20:8765 --key tk-… --name desk
temper devices                     # each device, whether it's online, and its models
temper devices prefer desk         # or `this`
```

The desktop's models show on the laptop's Models page as "on desk", and in `/v1/models` as `<model>@desk`.
A request for `<model>@desk` runs on the desktop. A request for a model by name alone runs on the laptop if
the laptop has that model, and on the desktop if only the desktop has it. When both have it, the
**preferred** device answers. Chat, on the Chat page, uses the laptop's own models. `temper chat --model <model>@desk` and
any app using the API can use the desktop's. The key is kept in `devices.json` in Temper's folder, readable
only by you; revoke it on the desktop to cut the link off.

## Apps that expect Ollama or llama.cpp

Some apps only look for Ollama (port 11434) or a llama.cpp server (port 8080). Under **Connect →
Discovery**, Temper can answer on those ports, so these apps find the loaded model with no setup. It only
starts where nothing else is already listening.
