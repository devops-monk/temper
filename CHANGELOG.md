# Changelog

## 0.2.0 (unreleased)

Temper now makes pictures, listens and speaks, and runs like a service your other apps can lean on.

### Chat
- Saved chats with search, folders, pins, export and imports from other apps.
- Thinking shown as it happens, with per-model thinking controls; generation presets.
- Compare mode: one prompt to two to four models side by side. Load several models at once.
- Attachments: images and audio for models that take them, documents as text.
- Auto-compaction when a chat fills the context, an explanation (and fixes) when a message doesn't fit,
  replies that keep generating when the window closes, a prompt queue, a prompt library and AGENTS.md.
- Tools: chat models can call tools, run Python and JavaScript in a sandbox (asking first), and use MCP
  servers. Small models' broken tool calls are repaired.
- Find in chat, keyboard shortcuts, interface zoom, and the app in English, Hindi, Spanish, Chinese and
  Japanese.

### Models
- MLX, Transformers, vLLM and SGLang engines next to llama.cpp, with load settings (context, KV cache,
  flash attention, GPU offload, speculative decoding).
- A catalog with each publisher's recommended settings; Hugging Face, a mirror or ModelScope as the source;
  resumable downloads; the Hugging Face token kept in the system keychain.
- Use your own model files and the models you downloaded with Ollama; convert pre-quantized and
  Mistral-format checkpoints to GGUF; see disk usage and choose where models live.
- A memory estimate before loading or training, and a Doctor that checks the setup.
- Temper updates itself from GitHub Releases; `curl | sh` and `irm | iex` installers.

### Images and audio
- **Images**: make pictures from a prompt or change one you have, with LoRAs, seeds and a gallery. LCM
  Dreamshaper (the default, about 2 seconds a picture on an M1 Pro), SD-Turbo and SSD-1B. OpenAI's
  `/v1/images/generations` and `temper image`.
- **Speech to text**: dictate into chat, transcribe files into text, subtitles or JSON, and translate to
  English, with Whisper running on this computer. OpenAI's `/v1/audio/transcriptions` and `translations`,
  and `temper transcribe`.
- **Text to speech**: replies read aloud in 54 voices and nine languages with Kokoro, a gallery of saved
  clips, OpenAI's `/v1/audio/speech` and `temper say`.
- **Stems**: split a song into vocals, drums, bass and the rest, plus the song without its vocals, on the
  new Audio page or with `temper stems`.

### Small models for everyday decisions
- **Decisions**: typed answers for your apps (yes or no, one of a set, labels or a score, each with a reason)
  at `/v1/decide`, saved by name at `/v1/decisions/{name}`, from MCP and with `temper decide`.
- **Classifiers**: teach a small model your labels from a file of examples in seconds. It answers in
  milliseconds with its confidence, shows how it did on held-out examples, and can answer saved decisions.
- **Embedding models**: tune one on pairs from your data so search finds what you mean, measured before and
  after. `/v1/embeddings` now answers with the embedding models on this computer.

### Running like a service
- Models load on demand when a request names one, and unload when idle.
- Run Temper's engine in the background and at login, without the window (`temper daemon`).
- `temper ps`, `load`, `unload`, `get`, `import`, `status`, `version`, `log`, with `--json`.
- A request log with speed, and prompts and replies when you ask for them.
- `temper engines`: versions, a preferred engine, and a pinned llama.cpp update.
- `temper create`: starter apps (chat, tools, RAG) in Python or TypeScript, wired to Temper.
- Linked devices: use another computer's models through this one, as `model@device`.

### Agents, tools and research
- An Anthropic-compatible API (`/v1/messages`), so Claude Code can use local models.
- MCP: add MCP servers for chat models to use, and serve Temper's models and jobs to other agents.
- Skills, a web page reader, web search with a choice of provider, a terminal tool, and a check of what the
  code sandbox stops. `temper start claude` (or codex, opencode, aider).
- Deep research: searched, read and cited reports.
- Knowledge bases: search your own documents from chat, kept in sync with their folders, with scanned PDFs,
  web pages and YouTube transcripts.

### Training and export
- Preference training (DPO, ORPO, SimPO, and KTO from replies marked good or bad), full and embedding training on Apple Silicon, and training text
  from documents.
- Batch size, LoRA rank and context fitted to this computer; a token preview; loss on replies only.
- Evaluation while training (sample replies, early stopping), run history, checkpoints, resuming and
  chatting with any checkpoint.
- Exports to Ollama and the MLX format, importance-matrix quants, and publishing fine-tunes, exports and
  datasets to Hugging Face. Datasets from web addresses and public S3 and GCS files.

### Everywhere else
- A Library page for everything Temper keeps, with folders, favourites and rename.
- API keys and a request monitor; LAN access with a QR code; answering as Ollama and llama.cpp so other
  apps find Temper by themselves.
- A tray icon, "Open with Temper" for `.gguf` files, `temper://` links, and dropping files on any page.
- A guided tour, statistics kept on this computer, no telemetry, and crash reports sent only by hand.
- Guides for every part of the app, and accessibility fixes in both themes.

## 0.1.0 (2026-10-24)

The first release: chat with local models, fine-tune with LoRA on an NVIDIA or AMD GPU or an Apple Silicon
Mac, map and clean datasets or generate them with Data Recipes, export to GGUF and chat with the result,
and a first-run setup that installs the engine for you.
