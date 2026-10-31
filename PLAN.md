# Temper: build plan

> **Temper**: run, fine-tune and ship local AI models from one desktop app.
>
> Tempering is the step after hardening steel. It tunes the metal for a specific job,
> which is what fine-tuning does to a base model. The app's palette comes from steel's
> temper colours: straw, bronze, violet and blue.

This plan lists every feature Temper will have: training, chat, data, export, serving, the CLI and
the desktop app. Each feature is one item, grouped into phases and ordered by dependency. Tick an
item when it ships.

**Coverage audit (2026-10-24):** items added by a feature audit against the leading local
fine-tuning tools are marked *(audit)*.

Legend: `[x]` built and checked · `[~]` built in part, or built but not yet run end to end · `[ ]` not started

> **Status (v0.1.0):** the shell, UI, engine API, hardware probe, hub search/download and dataset preview
> were tested on macOS (Apple M1 Pro). On Apple Silicon the whole loop has run end to end: MLX LoRA
> training (Qwen3-4B, 60 steps), GGUF export with llama.cpp's converter (Q4_K_M), then chat with the
> result. The NVIDIA/AMD training and export paths have **not** yet run on a GPU, so those
> items stay `[~]`.

---

## Name

**Temper** is the recommended name. Other candidates:

| Name | Why it fits | Watch out for |
|---|---|---|
| **Temper** (recommended) | The metallurgy step that tunes hardened steel for a job. Short, a verb, easy to say. | Generic word, so pair it with "Studio" or "AI" in store listings. |
| Quench | The fast cooling step in forging; suggests speed. | Sounds like a drink brand. |
| Anneal | The ML term ("simulated annealing") and a metallurgy term. | Harder to spell and say. |
| Kestrel | A fast, hovering bird, the opposite of a sloth. | Already the name of the ASP.NET web server. |
| Velox | Latin for "swift". | Less meaning for non-Latin readers. |

Check trademark, domain (`temper.ai`, `gettemper.app`), the GitHub org, Homebrew and winget IDs before release.

## Licensing (read first)

- The GPU training library is **mixed-licence** (parts are AGPL-3.0). Temper never copies or ships it: first-run setup installs it from PyPI on the user's machine, and it runs in a separate worker process. Get legal advice before bundling it or offering Temper as a hosted service.
- **Never copy code from AGPL-licensed apps** used as feature references. Write Temper's implementation from scratch.
- llama.cpp is MIT, stable-diffusion.cpp is MIT, whisper.cpp is MIT, vLLM and SGLang are Apache-2.0.

## Architecture

```
┌──────────────────────── Temper.app / .exe / .AppImage ────────────────────────┐
│  Tauri 2 shell (Rust)                                                         │
│   • window, tray, menus, deep links, auto-update, native file dialogs         │
│   • starts and supervises the engine, picks a free port, passes a token       │
│  ┌──────────── WebView: React 19 + Vite + Tailwind v4 + Zustand ───────────┐  │
│  │  Chat · Models · Train · Datasets · Export · Connect · Settings         │  │
│  └─────────────────────────────── HTTP / SSE ──────────────────────────────┘  │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │ 127.0.0.1:<port>  (bearer token)
┌──────────────────────────────────────▼────────────────────────────────────────┐
│  temper-engine (Python 3.12, FastAPI, managed by uv)                           │
│   • hardware probe   • HF Hub search/download   • jobs + SSE event streams     │
│   • inference router → llama.cpp server / MLX / transformers / vLLM / SGLang   │
│   • training workers  → PyTorch (CUDA, ROCm, Intel XPU), MLX-LM (Apple)        │
│   • export workers    → merged 16-bit, LoRA, GGUF, FP8/NVFP4, Ollama           │
│   • OpenAI- and Anthropic-compatible API on /v1                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

Why this stack:
- **Tauri 2**: installers around 10 MB, native on macOS (DMG, universal), Windows (NSIS/MSI, x64 and ARM64) and Linux (AppImage, deb, rpm).
- **Python engine**: the whole training ecosystem (PyTorch, TRL, PEFT) is Python. The shell installs it on first run with `uv`, so the installer stays small.

## Repo layout

```
temper/
├── PLAN.md                 ← this file
├── README.md
├── assets/icon.svg         ← master icon; `npm run icons` builds every platform size
├── src/                    ← React UI
├── src-tauri/              ← Rust shell + bundler config + generated icons
├── engine/                 ← Python FastAPI engine (pyproject managed by uv)
└── .github/workflows/      ← CI + cross-platform release builds
```

---

## Phase 0: Foundation

- [x] **0.1 Project scaffold**: Tauri 2 + React 19 + TypeScript + Vite + Tailwind v4.
- [x] **0.2 Design system**: temper palette tokens, light and dark themes, bundled fonts (Bricolage Grotesque for UI, JetBrains Mono for code), focus rings, reduced-motion support.
- [x] **0.3 App icon**: master SVG plus generated `.icns`, `.ico` and PNG sizes for every OS.
- [x] **0.4 App shell**: left navigation rail, workspace, settings side panel, status bar with live hardware and engine status.
- [x] **0.5 Engine process**: Tauri starts `temper-engine` on a free port with a random token, restarts it if it crashes, and stops it when the app quits.
- [x] **0.6 Engine API skeleton**: FastAPI with `/api/health`, bearer-token auth and CORS limited to the app origin.
- [x] **0.7 CI**: typecheck and UI build, engine tests on macOS, Windows and Linux, a full shell build on all three, and the headless first-run setup on each.
- [x] **0.8 Release pipeline**: pushing a `v*` tag checks it against the app version, builds DMG (arm64 and x64), NSIS `.exe` and MSI (x64 and ARM64), AppImage, deb and rpm (x64 and ARM64), and publishes the GitHub release once every platform succeeds. v0.1.0 shipped this way.
- [~] **0.9 First-run bootstrap**: on first launch the app downloads a pinned, checksum-verified `uv`, a uv-managed Python 3.12 and the engine's locked packages into app data, with a step-by-step progress screen and plain-language errors. Two profiles: **Chat only** (GGUF, no PyTorch) and **Chat and training** (MLX on Apple Silicon; PyTorch with `--torch-backend=auto` on NVIDIA/AMD). App updates that change the engine re-sync automatically; Settings can add training or reinstall. `temper --setup chat|train [--clean]` installs headlessly, and CI runs it on every OS.
  Setup stops before downloading anything when the disk can't hold about twice the install size (uv keeps a download cache), and says how much is needed and free. Tested on macOS (Apple Silicon): both profiles, a read-only app bundle, a network failure and retry, the update re-sync, and a too-small disk. Still to do: test the NVIDIA/AMD training profile on real hardware and add an Intel XPU profile.
- [ ] **0.10 Code signing**: Apple Developer ID plus notarization, Windows Authenticode (Azure Trusted Signing), Linux GPG-signed repos.
- [~] **0.11 Auto-update**: Tauri updater with signed manifests and stable/nightly channels. Roll back if a staged update fails.
  Done: releases publish signed update bundles and `latest.json`; Temper checks on launch and every 6 hours (can be turned off in Settings), shows a new version in the status bar and Settings, warns before restarting during a running job, and installs and relaunches. On Linux only the AppImage updates itself; deb and rpm installs update through the package manager. Still to do: a nightly channel and rollback.
- [~] **0.12 Distribution channels**: Homebrew cask, winget, Flathub, `.deb`/`.rpm` repos, `curl | sh` and `irm | iex` installers, Docker image (`temper/temper`, plus a ROCm variant).
  Done: `curl -fsSL https://temper.devops-monk.com/install.sh | sh` (macOS, Linux AppImage) and `irm https://temper.devops-monk.com/install.ps1 | iex` (Windows, per-user), both checked against the release's SHA-256 and tested in CI on macOS (Apple Silicon and Intel), Linux and Windows (x64 and ARM64) after every release. Still to do: Homebrew cask, winget, Flathub and package repos (each needs an account or submission), and a Docker image of the engine.

## Phase 1: Hardware and system

- [x] **1.1 Hardware probe**: OS, CPU, RAM, NVIDIA GPUs (`nvidia-smi`), Apple Silicon unified memory and chip name, AMD (`rocm-smi`), Intel XPU.
- [x] **1.2 Status bar meter**: GPU name, VRAM/RAM in use, engine state, polled while the window is visible.
- [ ] **1.3 Multi-GPU**: list every device, choose which GPUs to use per model and per job, NVLink/P2P detection.
- [ ] **1.4 Backends**: NVIDIA CUDA, AMD ROCm (including RDNA2 APUs), Intel XPU, Apple Metal/MLX, Vulkan, CPU, and NPUs (Intel/AMD/Qualcomm).
- [ ] **1.5 iGPU memory carve-out helper**: explain and adjust shared-memory limits on APUs.
- [x] **1.6 Doctor**: `temper doctor` and a Settings → Diagnostics page that checks drivers, CUDA/ROCm versions, disk space, ports and engine packages, with fix suggestions.
  Checks the engine and its package versions, free disk (and the model cache disk), memory, the GPU and driver (NVIDIA driver and CUDA version, ROCm, macOS for MLX, a GPU whose driver is missing), whether PyTorch or MLX can actually use the GPU, whether llama.cpp starts, who holds port 8765, and whether Hugging Face is reachable (proxy aware). Fixes the app can apply have a button: install llama.cpp, add training, reinstall the engine. "Copy report" copies the results for a bug report. CI runs `temper-engine doctor` on every OS.
- [~] **1.7 VRAM estimator**: before load or training, estimate memory for a given model, quantization, context, LoRA rank and batch size, and warn before running out of memory.
  Done: the engine reads the model's shape from `config.json` or the GGUF header (ranged requests for GGUFs on Hugging Face, so nothing is downloaded; hybrid models only count layers that keep a KV cache) and estimates weights, KV cache, activations and optimizer state against what the GPU can use (VRAM, or macOS's GPU share of unified memory). The Train page shows a live breakdown and asks before starting a run that won't fit; the Models page marks each quantization Fits, Tight fit or Too large at the 8K context Temper loads with, and asks before loading one that won't fit. MLX training is fitted to measured runs (within 3% on Qwen3-0.6B and 4B); llama.cpp KV cache matches llama-server. Still to do: measure the PyTorch path on an NVIDIA GPU (its constants come from published figures), and sliding-window KV caches (Gemma).
- [ ] **1.8 In-app runtime updates** *(audit)*: update the bundled llama.cpp build and upgrade `transformers` in place when a newly released model needs it, with rollback.

## Phase 2: Model hub and library

- [x] **2.1 Hugging Face search**: search by text, filter by author, sort by downloads, likes or recency.
- [x] **2.2 Local library**: list models in the HF cache with size and last-used date.
- [x] **2.3 Download manager**: background downloads with byte-level progress, speed and time left. Pause stops mid-file and Resume continues from the partial file (HTTP Range into `blobs/<sha256>.temper-partial`), also after a failure or an engine restart; dropped connections retry with backoff. Finished files are checked against their SHA-256 and stored in the standard Hugging Face cache.
- [x] **2.4 GGUF quant picker**: list every quant file in a repo (Q2_K … Q8_0, UD-dynamic quants, BF16) with size and an estimate of whether it fits in memory.
  Split GGUFs (`-00001-of-00003`, often in a folder per quant) show as one entry with the total size and download together; the fit estimate uses the memory estimator (1.7).
- [x] **2.5 HF token and gated models**: store the token in the OS keychain and guide the user through gated-model access. *(Done: Settings → Hugging Face account checks the token with Hugging Face, saves it to the macOS Keychain, Windows Credential Manager or Secret Service, and the engine gets it as HF_TOKEN, live without a restart; a `huggingface-cli login` token is used when Temper has none. Expanding a gated or private model shows what to do (open its page, add a token, check again); search shows the lock and date again; downloads explain gated errors.)*
- [x] **2.6 ModelScope mirror**: alternative hub for users in regions where Hugging Face is slow or blocked. *(Done: Settings → Model source picks Hugging Face, a Hugging Face mirror (hf-mirror.com or your own; workers get HF_ENDPOINT) or ModelScope, with a connection check. ModelScope search, file lists, memory estimates and resumable, checksum-verified downloads land in the normal model cache, so chat, the library and training (from the cached copy) work unchanged. Datasets and gated models still come from Hugging Face.)*
- [x] **2.7 Model catalog**: curated cards for Qwen, Llama, Gemma, Mistral, gpt-oss, GLM, Kimi, DeepSeek, Phi and others, with recommended settings and chat templates. *(Done: Models → Catalog lists 38 models in 11 families. `engine/scripts/build_catalog.py` reads every fact from the Hub: size, context, licence, gating, images, MoE, whether the chat template thinks (found by rendering it), the publisher's sampling settings (generation_config.json, else quoted from the card), and GGUF/MLX conversions from trusted publishers. Each card shows the fit on this machine, the best fit per family, what's downloaded, and whether the installed trainer supports it; Fine-tune fills in the Train form. Loading a catalog model in Chat applies its publisher's settings (presence penalty added). Chat templates come from each GGUF via llama.cpp.)*
- [x] **2.8 Import local files**: drag in a `.gguf`, a safetensors folder or an MLX folder. *(Done: drop files on the Models page or use Import GGUF file / Import folder. Temper links to them where they are (library.json) instead of copying. Each import is checked: GGUF magic and header (the model's own name), split parts beside it, folders of GGUFs, config.json plus weights, MLX quantization, missing tokenizer, and Hugging Face cache paths (symlinks, repo folders). Imports show in the library with Load or Fine-tune; moved files show as missing; removing one keeps the files. Mistral-format folders get a clear "not yet" (2.11).)*
- [x] **2.9 Disk management**: show cache usage and delete or move models; choose a custom models folder. *(Done: Settings → Storage shows what Temper uses (models, exports, runs, datasets, engine) against free disk space, and changes the models folder, moving models there as a job (instant on the same disk; across disks it copies with links kept, then deletes) or just switching. The folder keeps the Hugging Face cache layout and reaches workers as HF_HUB_CACHE. The library deletes a whole model or one quant (all parts of a split GGUF, blobs freed only when no snapshot links to them), and refuses while it's loaded or downloading.)*
- [x] **2.10 Import from other tools**: chats and settings from Claude Code, Codex, Cursor and Ollama. *(Done for Ollama: Models → On this computer finds Ollama's models (OLLAMA_MODELS, ~/.ollama, the Linux service folder) and adds them without copying; llama.cpp loads the GGUF blob directly. The Modelfile's sampling parameters and system prompt come along and Chat applies them. Chats from Claude Code, Codex and Cursor need somewhere to go, so they moved to 3.4.)*
- [x] **2.11 Pre-quantized and unusual checkpoints** *(audit)*: load FP8 (ModelOpt), MXFP4, NVFP4 and compressed-tensors checkpoints, and Mistral-format folders (`params.json` + `consolidated*.safetensors`). *(Done: chat runs GGUFs, so these load by converting them. Imports and downloads report the pre-quantization from config.json or ModelOpt's hf_quant_config.json (FP8, FP8/NVFP4 ModelOpt, compressed-tensors FP8/INT/NVFP4, MXFP4, GPTQ, BitNet) and whether llama.cpp's converter can read it; AWQ, bitsandbytes, 3-bit GPTQ, mixed compressed-tensors groups and MXFP4 outside gpt-oss/DeepSeek V4/Kimi K3/MiMo get a plain reason instead. Mistral-format folders import as their own kind. Convert to GGUF (Q4_K_M, Q8_0 or F16) runs the pinned converter, with mistral-common for Mistral's format, then llama-quantize, checks disk space first, never changes the source (a linked copy fixes transformers 5 tokenizer configs), and adds the GGUF to the library. Verified with Qwen3-0.6B-FP8 → Q4_K_M, loaded and answering; Mistral format verified up to the converter environment, not on a real checkpoint. Vision projectors aren't converted yet.)*
- [~] **2.12 Library** *(audit)*: one place for uploads, chat attachments, generated images and audio, fine-tuned models and sandbox files, with folders, favourites and rename.
  Done: a Library page lists what Temper keeps (uploads copied in with Add files or by dropping them on the page, prepared and generated datasets with their eval split, fine-tuning runs, exports named after their run, converted GGUFs) with search, favourites, folders (add, rename, remove; files stay), rename (a display name in files.json; files on disk keep theirs, since runs refer to them by path), show in folder and delete (refused while a model is loaded or a job is writing it; converted GGUFs leave the model list too). Each row has its next step: Train on it, Export, or Load in chat. Chat attachments (3.9) are listed too. Still to do: generated images (9.1) and audio (6.9), and sandbox files (4.2) each add a kind to the library when they exist.

## Phase 3: Inference and chat

- [x] **3.1 Chat UI**: streaming responses, Markdown rendering, stop, regenerate, copy, clear.
- [x] **3.2 llama.cpp runtime**: start and supervise `llama-server` for GGUF models; OpenAI-compatible streaming. Temper downloads a pinned llama.cpp release (checksummed binaries for each OS and GPU backend) with one click on the Models page, so nobody needs Homebrew or winget.
- [x] **3.3 Generation settings**: temperature, top-p, top-k, min-p, repeat penalty, max tokens, system prompt.
- [x] **3.4 Chat history**: threads stored in SQLite, search, rename, pin, folders/projects, export as Markdown or JSON. Import conversations from Claude Code, Codex and Cursor (moved from 2.10). *(Done: chats.db in Temper's home holds threads and messages, with FTS5 search over message text (prefix matches, highlighted snippets) plus titles. The Chat page has a sidebar: New chat, search, Pinned, folders (add, rename, remove), recent chats and one group per import source; each chat can be renamed, pinned, moved, exported as Markdown or JSON (save dialog in the app, a download in a browser) or deleted. A thread is saved before the reply streams and again after it, so nothing is lost if generation stops; untitled chats take their first line as the title; the system prompt is saved with the thread. Import chats finds Claude Code (~/.claude/projects), Codex (~/.codex/sessions, both rollout formats) and Cursor (its state database, opened read-only) history, keeps only what the person and assistant wrote (no tool calls, tool output, thinking or injected context), joins replies split around tool calls, and on re-import refreshes chats that grew while keeping the title, folder and pin. Checked against real Claude Code and Cursor history on this Mac; Codex against its documented formats.)*
- [x] **3.5 Reasoning view**: collapsible thinking blocks and a per-model reasoning-effort control. *(Done: llama-server streams a thinking model's reasoning separately, and chat shows it in a block that stays open while the model thinks, folds away when the answer starts ("Thought for 11 s") and opens on click. Reasoning is saved with the reply in chat history and exported inside a folded <details> block. After loading, the engine reads the chat template from llama-server's /props: templates with `enable_thinking` get a Think before answering switch, templates that take `reasoning_effort` (gpt-oss and others) get Low/Medium/High, and any thinking model gets a thinking budget in tokens (`thinking_budget_tokens`). The choices are remembered per model. A reply that runs out of tokens mid-thought says so. Checked with Qwen3-0.6B: thinking shown and saved, switched off, budget honoured; reasoning effort is wired to the template's capability but untested on a real gpt-oss model here.)*
- [x] **3.6 Presets**: saved generation presets per model; import recommended settings from the model card. *(Done: the Generation panel picks Recommended, Temper defaults or a saved preset, marks it (changed) once edited, and saves the system prompt, sampling, max tokens and thinking choices as a preset for this model or every model (presets.json). A preset can load with its model (★), ahead of the recommended settings; Save changes and Delete work in place. Recommended settings now cover models outside the catalog: an Ollama Modelfile, then the catalog's card-quoted settings, then the sampling the converter stored in the GGUF (general.sampling.*, copied from generation_config.json), then generation_config.json fetched from the model's Hugging Face repo, its card's base_model, or the base model the GGUF names (fetched, not cached, so no empty model appears in the library). Thinking controls now appear only when the template has a switch or effort level, or once the model has actually thought.)*
- [x] **3.7 Compare mode**: send one prompt to two or more models side by side (for example base vs fine-tuned). *(Done: Chat has a Compare tab: pick two to four GGUFs (exported fine-tunes listed first, then downloaded, imported and converted models), and each answers the same prompt in its own column, keeping its own thread for follow-ups, with the Generation panel's system prompt and sampling, a thinking switch, and its speed, length and time to first token. Each model gets its own llama.cpp server, reusing chat's when it's the same model; requests are routed by model. When they don't fit in memory together (from the memory estimate) they answer one at a time, each loaded in turn. A model that fails to load fails only its column, with a plain reason (an unsupported architecture is named). Compare's servers stop when the tab closes, after 10 minutes idle, or when chat loads a model. Comparisons aren't saved yet.)*
- [x] **3.8 Prompt queue**: queue follow-up prompts while one is generating; edit and reorder the queue. *(Done: while a reply is written, Enter (or the queue button beside Stop) queues the message instead. Queued messages show above the box and go out in order as each reply finishes; click one to edit it, drag it or move it with the arrow keys to reorder, remove one or clear them all. Stopping a reply or a failed one pauses the queue, with Send next to carry on. Regenerate continues the queue too. Opening another chat clears it.)*
- [x] **3.9 Multimodal input**: images (vision models), PDFs, DOCX, CSV and audio attachments. *(Done: attach with the paperclip, by dropping files on the message box or by pasting an image. Images (PNG, JPEG, GIF, BMP) and audio (WAV, MP3) go to models that take them; PDFs (text per page), Word documents (paragraphs and tables), CSV/TSV and text or code files are read into text and sent inside the message, with an estimate of their tokens and a warning when one may not fit the context. Chat loads a model's projector (an mmproj GGUF beside it) automatically, and downloading a quant fetches the repo's projector with it; the model's /props say whether it sees or hears, and an image for a model that can't is stopped with the reason. Files are kept in attachments/ in Temper's home, listed in the Library, saved with the chat and sent again on later turns. Unreadable formats (WebP, HEIC, .doc, .xlsx, M4A…) say what to convert to. Checked live with SmolVLM-256M (image) and a CSV; audio is covered by tests only so far, and compare mode doesn't take attachments yet.)*
- [x] **3.10 Auto-compaction**: rolling context window that summarises old turns when context fills. *(Done: before each reply the engine counts the prompt with the loaded model's own template and tokenizer (llama-server /apply-template and /tokenize; pictures and sounds estimated). If it leaves less than a quarter of the context (or Max new tokens, if smaller) for the reply, the oldest turns are summarised by the same model, a context-sized chunk at a time when they're long, keeping the most recent turns (up to 40% of the context) word for word; the summary goes into the system prompt and later turns build on it. The chat shows every message, dims the summarised ones, marks where the summary starts (click to read it), shows "Summarising earlier messages…" while it works, and a token meter. The summary is saved with the chat. A toggle in the Generation panel turns it off; a message too big for the context on its own is refused with why. Also: an empty reply now shows a notice instead of vanishing, and failed replies are no longer sent back to the model.)*
- [x] **3.11 Speculative decoding**: draft-model pairing for faster generation. *(Done: the chat panel's Model section offers Off, prompt lookup (n-gram, no extra model), the model's own MTP layers when its GGUF has them, and any local GGUF that can draft for it (same tokenizer, under half the size), with how many tokens to draft at a time. After each reply it shows the speed and how many drafted tokens were accepted, and warns when so few are accepted that drafting slows replies. Measured on an M1 Pro with a Qwen3-4B fine-tune: prompt lookup doubled the speed of a repeated request (44 to 88 tokens/s); Qwen3-0.6B as a draft accepted 21% and was slower, because the fine-tune has drifted from the base model.)*
- [x] **3.12 KV-cache controls**: context length, KV quantization, flash attention, layer offload. *(Done: context length up to the model's maximum, a 16-, 8- or 4-bit KV cache (quantized needs flash attention), flash attention, and how many layers go on the GPU, in the same Model section. Settings are saved per model in the engine and apply wherever the model is loaded (Models page, library, compare); changing them shows the new memory estimate (cache type and draft model included) and reloads with Reload with these. The engine reports the context llama-server set up, which the token meter and attachment warnings use.)*
- [x] **3.13 Extra engines**: MLX (Apple), transformers, and opt-in vLLM and SGLang (Linux + NVIDIA/AMD) with tensor and pipeline parallel. MLX tested live (model folders, and fine-tune runs with their adapter). Transformers, vLLM and SGLang install into their own environments; their commands are unit-tested but not run on this hardware (Transformers needs PyTorch, vLLM and SGLang need a Linux GPU).
- [x] **3.14 Multiple loaded models**: load several models and route between them; unload when idle. Models load alongside chat's (any engine), answer on `/v1` by name, swap into chat without reloading, and unload after a set idle time.
- [x] **3.15 Find in chat, keyboard shortcuts, interface zoom.** ⌘/Ctrl F finds in the chat with highlights; N, L, 1–8, comma, Esc and / shortcuts; zoom from 67% to 200%, kept between launches.
- [~] **3.16 Localisation (i18n)**: string catalog with parity checks; start with English, Hindi, Spanish, Chinese and Japanese. Catalogs in `src/locales`, typed `t()` keys, a language setting, and `npm run check:locales` in CI. Translated so far: navigation, status bar, chat, chat list, find, shortcuts and Settings → Appearance; the other pages still show English.
- [x] **3.17 Resumable generations** *(audit)*: replies keep generating in the engine if the window closes; reopening the chat shows the finished answer. The engine writes and saves each reply; reopening a chat mid-reply replays it and follows it live.
- [x] **3.18 Prompt library** *(audit)*: save, name and reuse prompts. From the composer or by typing `/` in an empty message; `{{variables}}` are asked for on use; most-used first.
- [x] **3.19 AGENTS.md instructions** *(audit)*: a global instructions file plus each project folder's `AGENTS.md`, read the way Codex reads them. Global AGENTS.md in Settings; a chat's project folder adds AGENTS.md (or AGENTS.override.md) from the repo root down, capped at 32 KiB.
- [x] **3.20 Keep chat context across GPU swaps** *(audit)*: save and restore the KV cache when image or video generation temporarily takes the GPU. `POST /api/runtime/suspend` saves llama-server's slots to disk and unloads; `resume` reloads and restores them (tested: 4,904 cached tokens back, 15 s of prompt processing saved). Image and video generation will call it when they land.
- [x] **3.21 Context-fit explanations** *(audit)*: when a prompt doesn't fit the context window, say why and what to change. *(Done: a 413 from prepare carries `fit`, a token breakdown (instructions, earlier turns, the message, each attachment, the reply's room) and fixes with numbers; chat shows it as a bar with fixes, and a shorter reply or auto-compact applies and retries.)*

## Phase 4: Agents, tools and MCP

- [x] **4.1 Tool calling**: native function calling for models that support it, with repair of malformed tool calls. *(Done: tools.py with a calculator and a clock to start; llama.cpp gets tools natively, other engines in the prompt; the engine runs the rounds of calls and results. Repairs near-JSON arguments, calls written as text in five model families' formats, near-miss names, and repeated-call loops; what can't be repaired goes back to the model. Steps are saved with the reply and shown as cards; per-chat tool switches in the Generation panel.)*
- [x] **4.2 Code execution sandbox**: run Python/JS in an isolated sandbox with per-tool approval (Ask / Auto / Off). *(Done: sandbox.py runs code in a per-chat folder with a time limit, a file-size cap and a stripped environment, inside Seatbelt on macOS or bubblewrap on Linux (no network, writes only to its folder, credential folders unreadable); Windows has no OS sandbox yet and says so. Each tool is Off, Ask first or Run per chat; a call that asks waits in the engine until it's allowed, declined or allowed for the rest of the chat. Data libraries for Python install from Settings.)*
- [x] **4.3 MCP client**: add MCP servers (stdio, HTTP, SSE), browse tools, approve them; search over large tool catalogs (BM25) so they fit in context. *(Done: mcp.py, a JSON-RPC client over stdio, Streamable HTTP (sessions, event-stream answers) and the older HTTP+SSE; servers in mcp.json in the common mcpServers format, added by form or pasted JSON, keys masked; each tool Off / Ask first / Run; a chat turns servers on in its Tools. Over 24 tools, the model gets a search_tools tool (BM25) and the tools it finds become callable. OAuth sign-in for remote servers isn't done yet: use a header.)*
- [x] **4.4 Temper as an MCP server**: expose local models and jobs to other agents. *(Done: the engine answers MCP's Streamable HTTP at /mcp behind its token, with temper_status, list_models, load_model, unload_model, ask_model, list_jobs, get_job and cancel_job; Connect shows configs for Cursor/Claude Desktop/VS Code, Claude Code and Codex. The key still changes on restart; a lasting one comes with 4.11.)*
- [x] **4.5 Agent skills**: `SKILL.md` folders and `@skill` mentions in chat. *(Done: skills.py reads skills from Temper's home and a chat's project (.agents/skills, .claude/skills; the project's win); @name anywhere in a chat puts that SKILL.md, with its files' paths, in the system prompt. Typing @ opens a picker; Settings makes, imports and removes skills.)*
- [~] **4.6 Built-in browser agent**: fetch, read and act on web pages. *(Fetch and read done: the read_web_page tool (asks first) turns pages into text with numbered links to follow, in parts for long pages, reads PDFs, marks page text as untrusted, and refuses local and private addresses. Acting on pages (clicking, typing, screenshots) needs a headless browser install and is still to do.)*
- [x] **4.7 Ready-made integrations**: Blender MCP, file system, GitHub. Settings → MCP servers has Ready-made cards: Files (one folder you pick, via the reference filesystem server on npx), GitHub (GitHub's hosted MCP server with a personal access token, kept masked) and Blender (blender-mcp run with uv). Not tried live: no GitHub token or Blender here, and the files server wasn't started to avoid an npx download.
- [~] **4.8 Connect coding agents**: copy-paste config for Codex, OpenCode, VS Code (Continue), Aider, Python and curl, pointing at the local `/v1` endpoint. The key now stays the same across restarts, so pasted configs keep working; `temper start` (4.9) sets an agent up in one step. Claude Code waits on 10.2.
- [x] **4.9 `temper start <agent>` CLI** that does the same from a terminal. `temper start codex|opencode|aider [args]` finds the running engine (it leaves its address and key in `engine.json`, readable only by the user), picks the loaded model and starts the agent pointed at it, passing the agent's own arguments through. Tried live with a stand-in aider; Codex and OpenCode aren't installed here.
- [~] **4.10 Terminal tool and OS sandbox** *(audit)*: a terminal tool alongside Python, and a page showing which OS sandbox (Seatbelt, bubblewrap, AppContainer) protects tool runs on this machine. Done: a Terminal tool (`run_shell`: bash or sh, PowerShell on Windows; asks first) in the same per-chat sandbox, and Settings → Code sandbox → Check the sandbox, which tries to reach the internet, write outside its folder and read a credential folder from inside it, and lists what was stopped (verified on macOS: all stopped). Left: Windows has no OS sandbox yet (AppContainer isn't built), so there only the time limit applies and the page says so.
- [x] **4.11 `temper mcp add <agent>`** *(audit)*: register Temper's MCP server with Claude Code, Codex and other agents, and check it with `temper doctor --mcp`. Done: `temper mcp add claude|codex|cursor|vscode` (Claude Code through `claude mcp add --scope user`; the others by editing their config, keeping a `.bak`), and `temper doctor --mcp` connects to the server, lists its tools and shows which agents have it and whether they point at this engine. Claude Code tried live in a throwaway home: it connected.

## Phase 5: Search, research and RAG

- [x] **5.1 Web search tool**: private web search with configurable providers. A `web_search` tool (asks first) returns numbered results with addresses and snippets, to read in full with `read_web_page`. Settings → Web search picks the provider: DuckDuckGo (default, no key), SearXNG (your instance), Brave Search or Tavily (your key, kept masked). Only the search words go to the provider. DuckDuckGo tried live; SearXNG, Brave and Tavily are tested against mocked answers (no instance or keys here).
- [x] **5.2 Deep research**: multi-step research runs with citations, a review screen and a report export. A Research page: the loaded model plans searches, reads each source one at a time (so small models keep up) and notes what bears on the question, may search again for what's missing (Quick, Standard, Thorough), then writes a report from the notes alone with [n] citations that jump to the source. Review: untick sources and write the report again. Export to Markdown with a source list. Runs are kept and listed. Tried live with a 4B MLX model: 3 searches, 5 sources, a correct cited report.
- [x] **5.3 Knowledge base (RAG)**: add folders and files (PDF, DOCX, PPTX, XLSX, Markdown, code); chunk, embed (local embedding GGUF) and index in SQLite + FTS5 + vectors; hybrid retrieval with citations. Settings → Knowledge bases: name a base and pick folders; Temper reads PDF, Word, PowerPoint, Excel, HTML, Markdown, text and code, cuts them into passages and keeps them in SQLite with FTS5 and a vector each (an embedding GGUF in its own llama-server: Qwen3 Embedding 0.6B by default, Nomic or BGE Small for English). Search merges word and meaning rankings; re-indexing reads only changed files. Chat models get a `search_knowledge` tool that returns passages with their files to cite. Tried live with BGE Small on README and PLAN.
- [x] **5.4 Folder sync**: watch folders and re-index on change, honouring `.gitignore`. Each knowledge base can be kept in sync (on by default): while the engine runs it checks the folders every minute and re-indexes only what was added, changed or removed. Folders honour .gitignore files, nested ones and `!` exceptions included (pathspec), and skip hidden and build folders.
- [x] **5.5 OCR for scanned PDFs**: vision-model transcription first, Tesseract as a fallback. PDF pages with no text layer are rendered (PDFium via pypdfium2) and read by the chat model when it can see images, else by Tesseract when installed, in knowledge bases (every page) and chat attachments (up to 20 pages). Settings → Knowledge bases says which applies. Tesseract tested live on a generated scan; the vision path is tested with a stand-in model (no vision model downloaded here).
- [x] **5.6 YouTube transcripts and URL ingestion.** A YouTube address given to `read_web_page` returns the video's transcript with times (its own captions first, else automatic ones; youtube-transcript-api). Knowledge bases take web pages and videos by address alongside folders: fetched when added and again on a manual re-index, each passage headed by the page's title. Transcript fetching tried live; knowledge ingestion of addresses is tested with stand-in pages.

## Phase 6: Fine-tuning

- [x] **6.1 Training wizard UI**: choose base model, dataset, method and hyperparameters, then start, stop and watch.
- [~] **6.2 SFT with LoRA / QLoRA**: 4-bit loading, LoRA adapters and TRL `SFTTrainer` on NVIDIA/AMD; MLX-LM on Apple Silicon.
- [x] **6.3 Live metrics**: loss chart, learning rate, step/epoch, tokens/s, ETA and a log console over SSE.
- [~] **6.4 Full fine-tuning and continued pretraining** (including embedding and `lm_head` training). Done: Full fine-tune is offered on Apple Silicon too (mlx-lm, every layer rather than its default last 16), and a new *Also train the embeddings and output head* option trains `embed_tokens` and `lm_head` for continued pretraining (on MLX by unfreezing them in Temper's trainer wrapper; on GPUs by adding them to the LoRA targets). Both refuse a 4- or 8-bit MLX model, which can't store changed weights. Tested: the rules, and the unfreezing on a tiny MLX model. Not run live: there's no 16-bit model downloaded here, and no NVIDIA GPU.
- [~] **6.5 Preference training**: DPO, ORPO, KTO, SimPO.
  Done: DPO, ORPO and SimPO as LoRA adapters, chosen as the run's objective on the Train page, from rows with a prompt and a chosen and rejected reply (as text or as conversations). On Apple Silicon a small MLX trainer of Temper's own does it and saves adapters as mlx-lm does, so chat, checkpoints and export work as for fine-tuning; runs report how often the model prefers the chosen reply, in training and on held-out pairs. Tested on MLX with Qwen3-0.6B: DPO took held-out accuracy from 25% to 100% in 20 steps, starting at exactly ln 2; ORPO and SimPO trained cleanly. GPU runs use TRL's DPO, ORPO and CPO (SimPO) trainers; not yet run on GPU hardware. Still to do: KTO (unpaired good/bad examples).
- [ ] **6.6 Reinforcement learning**: GRPO, GSPO, Dr. GRPO, DAPO, with reward-function editor and vLLM fast generation.
- [ ] **6.7 FP8 and NVFP4 training** on supported GPUs; QAT (quantization-aware training).
- [ ] **6.8 Vision fine-tuning**: VLMs such as Qwen-VL, Gemma vision and Llama vision.
- [ ] **6.9 TTS and audio fine-tuning**: Orpheus, Sesame CSM, Whisper (STT).
- [ ] **6.10 Embedding model fine-tuning**: sentence-transformers / EmbeddingGemma.
- [ ] **6.11 Diffusion fine-tuning**: image and video LoRA (Flux, Qwen-Image, Z-Image and others).
- [ ] **6.12 Decision/classifier models**: train a classifier head from a language model.
- [ ] **6.13 Long-context training**: memory-saving gradient checkpointing, tiled MLP, context parallel.
- [ ] **6.14 Multi-GPU training**: DDP and FSDP2 (including QLoRA on FSDP2).
- [x] **6.15 Apple Silicon training** through MLX-LM LoRA: rank, alpha and dropout via an mlx-lm config, gradient accumulation and checkpointing, live loss/validation/tokens-per-second in the UI. Verified 2026-10-24 on an M1 Pro (16 GB): Qwen3-4B-Instruct 4-bit, 500 FineTome rows, 60 steps in about 12 minutes; validation loss 1.62 → 0.81; the 29 MB adapter loads with `mlx_lm generate`.
- [~] **6.16 Chat templates**: automatic template detection, `train_on_responses_only` masking, template override.
  Done: training renders rows with the model's own chat template, in the form it has at inference (so Qwen3's history-only think block isn't learnt), and refuses conversations for a model with no template. Loss is on the replies only on both trainers: MLX masks the prompt, and GPU runs now train on prompt-completion rows with completion-only loss (not yet run on GPU hardware). Still to do: overriding the template.
- [x] **6.17 Token preview**: show exactly what the model will see for a sample, with masked tokens dimmed.
  On the Train page, *See a row as the model sees it* shows any of the first 100 rows token by token with the base model's tokenizer: the prompt dimmed (no loss on it), token boundaries shaded, and tokens past the context length struck through with a warning.
- [x] **6.18 Evaluation**: held-out eval set, eval loss, early stopping, sample generations during training.
  A prepared dataset's held-out file is used for validation loss (MLX holds out 5% when there's none), measured about ten times a run. At each evaluation the model answers two held-out prompts, shown beside the reply in the data, with a slider back through earlier steps. Early stopping ends the run after three evaluations without improvement and keeps the best adapter. Tested on MLX: a run stopped at step 12 and kept step 3. The GPU path uses the trainer's early stopping and generates on evaluate; it hasn't been run on GPU hardware yet.
- [x] **6.19 Checkpoints and resume**: list checkpoints, resume from one, compare runs.
  Runs save a checkpoint every N steps (five over the run by default). Past runs on the Train page list each one; "Carry on from here" starts a new run from it for the steps left, its curve lined up with the original's, and ticked runs share one loss chart. Tested on MLX: a resumed run picks up the checkpoint's weights. GPU runs use the trainer's own checkpoints and `resume_from_checkpoint`.
- [x] **6.20 Training history**: every run stored with config, metrics, dataset hash and provenance; re-run with one click.
  Each run keeps its events (metrics, evaluations, dataset summary, errors) in `events.jsonl` and its outcome in `status.json`, with the base model's revision, the dataset's SHA-256 and library versions. "Run again" starts it with the same settings.
- [x] **6.21 VRAM-aware defaults**: choose batch size, rank and context automatically from the hardware probe.
  The Train page fits context length, then LoRA rank, then batch size to 80% of the memory the trainer can use (from the same estimate as the Memory section), with gradient accumulation keeping eight rows per step. It refits when the model or method changes, until one of those values is edited by hand; *Fit to this computer* turns it back on. On a 16 GB Mac, Qwen3-4B gets 2,048 tokens, rank 16, batch 2 × 4; Qwen3-0.6B gets 4,096.
- [ ] **6.22 Remote and cloud training**: launch the same job on a remote machine, Colab, Kaggle or a cloud GPU and stream its metrics back.
- [ ] **6.23 Text-diffusion language models** *(audit)*: train and run diffusion LMs such as DiffusionGemma and Dream.
- [ ] **6.24 Memory-saving optimizers** *(audit)*: 8-bit AdamW, Q-GaLore, FlashAdamW and optimizer-state CPU offload.
- [ ] **6.25 Drafter and MTP-head fine-tuning** *(audit)*: fine-tune a model's multi-token-prediction head or a draft model so speculative decoding still works after fine-tuning.
- [x] **6.26 Checkpoint preview** *(audit)*: chat with any checkpoint during or after training through a per-checkpoint `/v1` endpoint.
  "Chat with it" on any checkpoint loads the base model with that checkpoint's adapter in chat, and so on Temper's `/v1` API, while the run goes on or after it ends. One checkpoint is served at a time, through the usual endpoint rather than one per checkpoint.
- [x] **6.27 Raw-text continued pretraining** *(audit)*: chunk plain documents into training text, with no chat format needed. Datasets → *Or train on your own documents*: pick folders; Temper reads them with the knowledge-base readers (PDF, Word, PowerPoint, Excel, HTML, Markdown, text, code; scanned pages by OCR), cuts them into pieces of about 500, 1,000 or 2,000 tokens, drops duplicates and writes `{"text"}` rows with 5% held out, ready to train with no chat format. Tried live: README and PLAN became 42 rows, and a 6-step LoRA on Qwen3-0.6B (MLX) trained on them.

## Phase 7: Datasets and Data Recipes

- [x] **7.1 Dataset preview**: open a Hugging Face dataset or a local CSV, JSON or JSONL and see rows and columns.
- [x] **7.2 Column mapping**: map columns to chat roles (prompt/response, ShareGPT, OpenAI messages, plain text) with a live "as the model sees it" preview, suggested mappings for common column names (question/answer, prompt/completion…), an optional system prompt, and subset/split pickers for Hugging Face datasets.
- [x] **7.3 Format conversion and cleaning**: deduplicate (case- and whitespace-insensitive), filter by length, cap rows, shuffle, split train/eval. Writes standard `messages` JSONL plus `.eval.jsonl`, which training uses for validation loss automatically.
- [x] **7.4 Data Recipes**: build datasets from PDFs, DOCX, text files, folders, web pages and GitHub repos using a pipeline of steps (Recipes page). Built on NVIDIA data-designer (Apache-2.0) in its own uv environment; six templates (document Q&A, topic Q&A, graded answers, text to SQL, Python tasks, instructions for existing text); save, preview 3 rows, generate, then train on the result in one click.
- [~] **7.5 Synthetic data generation**: use a local or cloud model to generate Q&A, conversations or reasoning traces from source documents, with validators. Generated text, fields, code and grades, random-choice and template columns, and code checks all work with the local chat model; OpenAI-compatible cloud providers are wired up but not yet tested against a live service.
- [x] **7.6 Push to Hub**: upload finished datasets to Hugging Face.
  The upload icon on a prepared dataset in the Library publishes it (private unless unticked) as `data/train.jsonl` plus its held-out `data/test.jsonl`, with a dataset card naming the splits so the Hub's viewer and `load_dataset` read it. Tested against a stand-in for the Hub; nothing was uploaded to a real account.
- [~] **7.7 S3 and remote sources.**
  Done: anywhere a dataset is given (Datasets, Train, the token preview), it can be an https:// link to a .jsonl, .json, .csv or .parquet file, or a public `s3://` or `gs://` one. It's downloaded once into Temper's cache and then read like a local file; previews of JSONL and CSV stream only their first rows. Tested with GSM8K's Parquet test split from the Hub: previewed, mapped and prepared. Still to do: private buckets (credentials and request signing).

## Phase 8: Export and deploy

- [x] **8.1 Export UI**: choose a checkpoint and one or more formats.
- [~] **8.2 LoRA adapter, merged 16-bit and merged 4-bit** (`save_pretrained_merged`).
- [~] **8.3 GGUF export**: q4_k_m, q5_k_m, q8_0, f16 and other quant types. On Apple Silicon it fuses with mlx-lm, converts with llama.cpp's `convert_hf_to_gguf.py` (in an isolated uv environment), then runs `llama-quantize`; checked end to end. On NVIDIA/AMD it uses the training library's GGUF saver (not yet run).
- [x] **8.4 Dynamic quants**: dynamic per-layer quants with an importance matrix.
  Done: Export has "Calibrate on the training data (importance matrix)". The run's own training rows, rendered with the model's chat template, go through `llama-imatrix`, and every quant is made with that matrix (named `…-imatrix-Q4_K_M.gguf`). It also unlocks the IQ quants (IQ4_XS, IQ3_M, IQ2_M), and the smallest ones keep their output layer at 8 bits. Tested: Q4_K_M and IQ3_M made from a Qwen3-0.6B run, and both generate.
- [~] **8.5 FP8, NVFP4, TorchAO, OpenVINO and MLX exports.**
  Done: MLX. Runs trained on Apple Silicon export a fused MLX model that keeps the base model's quantization, ready for mlx-lm and LM Studio's MLX engine. Tested: it loads and chats. Still to do: FP8, NVFP4, TorchAO and OpenVINO, which need NVIDIA or Intel hardware to make and test.
- [x] **8.6 Ollama export**: write a Modelfile with the right template and `ollama create`.
  *Add to Ollama* on an export (Export page and Library) writes a Modelfile beside the GGUF and runs `ollama create` as a job, under a name taken from the run and quant (e.g. `gsm8k-math-tutor:q4_k_m`). Ollama picks the matching chat template and stop tokens from the template the GGUF carries, which is more reliable than translating Jinja to Go templates. Says so when Ollama isn't installed or isn't running. Tested with Ollama 0.40: it recognised a ChatML GGUF's template.
- [x] **8.7 Push to Hugging Face**: model card generated from the training run.
  The upload icon on a fine-tune (its adapter) or an export (its GGUF or merged files) in the Library publishes it, private unless unticked, leaving Temper's bookkeeping and intermediate checkpoints behind. The card is written from the run: base model and revision, training data and its SHA-256, objective, method and settings, final losses and library versions, with how to run it. Tested against a stand-in for the Hub; nothing was uploaded to a real account.
- [x] **8.8 Load the exported model in Chat** in one click, to try it right away.
- [ ] **8.9 MoE expert pruning (REAP)** *(audit)*: drop rarely used experts from mixture-of-experts models to shrink them before export.

## Phase 9: Image, video and audio

- [ ] **9.1 Image generation**: text-to-image and image editing with diffusion models (stable-diffusion.cpp for GGUF, diffusers for PyTorch), LoRA loading, ControlNet, gallery.
- [ ] **9.2 Video generation**: text/image-to-video models.
- [ ] **9.3 Speech-to-text**: Whisper (whisper.cpp) dictation in the composer and file transcription.
- [ ] **9.4 Text-to-speech**: voice library and audio gallery.
- [ ] **9.5 Music and audio editing models.**

## Phase 10: Serving, API and remote access

- [~] **10.1 OpenAI-compatible API**: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` for other apps to use.
- [x] **10.2 Anthropic-compatible API**: `/v1/messages`, so Claude Code can use local models directly.
  Done: `/v1/messages` and `/v1/messages/count_tokens` (an estimate) over the loaded model, translated to and from the servers' OpenAI shape: system prompts, images, tool use and tool results, tool choice, stop sequences, stop reasons and token counts, whole or streamed (with pings while a long prompt is read). A thinking model's reasoning is left out. Unknown model names, like Claude Code's, go to chat's model. `temper start claude` and the Connect page set Claude Code up; the Connect page also has an Anthropic SDK example. Tested with the official Anthropic SDK on a Qwen3-4B GGUF: a reply, a tool call, a streamed answer after a tool result, and a streamed tool call. Claude Code reached it and got a streamed reply started, but its prompt is long enough that a full session needs a faster or bigger model than this test machine could run.
- [ ] **10.3 API keys**: create, name, revoke; usage tracking and an API monitor page.
- [ ] **10.4 LAN access**: serve to other devices on the home network with a QR code.
- [ ] **10.5 Secure remote access**: Cloudflare Tunnel HTTPS link, with admin password and an option to turn tools off.
- [ ] **10.6 Accounts**: admin password, multiple users with separate chats and jobs, reset password from the CLI.
- [ ] **10.7 Cloud providers**: connect OpenAI, Anthropic, Google, OpenRouter or a ChatGPT/Codex subscription and use them next to local models; keys stored in the OS keychain.
- [ ] **10.8 Headless server mode**: `temper serve` with systemd unit, Docker image and a browser UI.
- [ ] **10.9 Decisions API** *(audit)*: typed answers (yes/no, choice, score) from decision models or a saved connection.
- [ ] **10.10 Engine discovery** *(audit)*: answer the probes Ollama- and llama.cpp-aware tools send, so they detect Temper automatically.

## Phase 11: CLI

- [ ] **11.1 `temper`** with subcommands mirroring the app: `chat`, `train`, `export`, `eval`, `serve`, `start <agent>`, `mcp`, `doctor`, `update`, `uninstall`.
- [ ] **11.2 YAML training configs**: `temper train config.yaml`, with `--dry-run` and CLI overrides.

## Phase 12: Polish, quality and launch

- [ ] **12.1 Guided tour** on first launch.
- [ ] **12.2 Profile and statistics**: tokens generated, models trained, time saved.
- [ ] **12.3 Accessibility pass**: keyboard-only use, screen-reader labels, contrast checks in both themes.
- [ ] **12.4 Telemetry: off by default**, opt-in crash reports only.
- [ ] **12.5 Tests**: engine unit tests (pytest), UI tests (Vitest + Playwright), end-to-end smoke test of a tiny model on CI.
- [ ] **12.6 Docs site and notebooks**: getting-started guides plus Colab/Kaggle notebooks for every training method.
- [ ] **12.7 Launch**: website, download page, release notes.
- [ ] **12.8 Desktop integration** *(audit)*: menu-bar/tray icon, "Open with Temper" and drag-and-drop for models and datasets, `temper://` deep links, and native clipboard and file dialogs everywhere.
- [x] **12.9 Stay awake during jobs**: training, export and recipe workers stop the computer idle-sleeping until they finish (caffeinate on macOS, SetThreadExecutionState on Windows, systemd-inhibit on Linux).

---

## Running it today (developers)

```bash
# prerequisites: Node 20+, Rust stable, uv
cd temper
npm install
npm run icons             # regenerate platform icons from assets/icon.svg
npm run tauri dev         # starts the UI + Rust shell; the shell starts the engine with uv

# engine only
cd engine && uv run temper-engine --port 8765 --token dev
```

Training needs the GPU extras: `cd engine && uv sync --extra train` (NVIDIA/AMD/Intel)
or `uv sync --extra mlx` (Apple Silicon). Local GGUF chat needs `llama-server` on PATH
(`brew install llama.cpp`, `winget install llama.cpp`, or a release from GitHub).

## Release

Push a tag `v*` and `.github/workflows/release.yml` builds and drafts a GitHub release with:
`Temper_<ver>_aarch64.dmg`, `Temper_<ver>_x64.dmg`, `Temper_<ver>_x64-setup.exe`,
`Temper_<ver>_arm64-setup.exe`, `Temper_<ver>_amd64.AppImage`, `.deb` and `.rpm`.
