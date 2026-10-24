# Temper: build plan

> **Temper**: run, fine-tune and ship local AI models from one desktop app.
>
> Tempering is the step after hardening steel. It tunes the metal for a specific job,
> which is what fine-tuning does to a base model. The app's palette comes from steel's
> temper colours: straw, bronze, violet and blue.

This plan lists every feature Unsloth offers, across Unsloth Core (the training library),
Unsloth Studio (the web UI), Unsloth Desktop and the `unsloth` CLI. Each feature is one item,
grouped into phases and ordered by dependency. Tick an item when it ships.

**Coverage audit (2026-10-24):** checked against the Unsloth source in `../unsloth`: every Studio
frontend feature folder, backend route and core module, every CLI command, the Core `models/`,
`optimizers/` and `dataprep/` packages, the desktop shell and the Studio docs. Items added by that
audit are marked *(audit)*. Re-run the audit when Unsloth ships a release.

Legend: `[x]` built and checked · `[~]` built in part, or built but not yet run end to end · `[ ]` not started

> **Status (v0.1.0):** the shell, UI, engine API, hardware probe, hub search/download and dataset preview
> were tested on macOS (Apple M1 Pro). On Apple Silicon the whole loop has run end to end: MLX LoRA
> training (Qwen3-4B, 60 steps), GGUF export with llama.cpp's converter (Q4_K_M), then chat with the
> result. The Unsloth (NVIDIA/AMD) training and export paths have **not** yet run on a GPU, so those
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

- **Unsloth Core** (`unsloth` on PyPI) is **mixed-licence**: its package metadata says Apache-2.0, but 74 of its 171 source files (kernels, FSDP2, diffusion, speculative decoding and others) are headed `AGPL-3.0-only`, and the repository ships an AGPL `COPYING` file. Temper never copies or ships Unsloth. First-run setup installs it from PyPI on the user's machine, and it runs in a separate worker process. Get legal advice before distributing Unsloth with Temper or offering Temper as a hosted service.
- **Unsloth Studio** (`unsloth/studio/**`) is **AGPL-3.0**. **Do not copy code from it** unless Temper is also released under AGPL-3.0. Use it only as a feature reference; write Temper's implementation from scratch.
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
│   • training workers  → unsloth (CUDA, ROCm, Intel XPU), MLX-LM (Apple)        │
│   • export workers    → merged 16-bit, LoRA, GGUF, FP8/NVFP4, Ollama           │
│   • OpenAI- and Anthropic-compatible API on /v1                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

Why this stack:
- **Tauri 2**: installers around 10 MB, native on macOS (DMG, universal), Windows (NSIS/MSI, x64 and ARM64) and Linux (AppImage, deb, rpm). Unsloth Desktop uses the same stack.
- **Python engine**: the whole training ecosystem (PyTorch, unsloth, TRL, PEFT) is Python. The shell installs it on first run with `uv`, so the installer stays small.

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
- [~] **0.7 CI**: typecheck, build, `cargo check`, and an engine smoke test on macOS, Windows and Linux.
- [~] **0.8 Release pipeline**: `tauri-action` matrix producing DMG (arm64 and x64), NSIS `.exe` (x64 and ARM64), AppImage, deb and rpm.
- [~] **0.9 First-run bootstrap**: on first launch the app downloads a pinned, checksum-verified `uv`, a uv-managed Python 3.12 and the engine's locked packages into app data, with a step-by-step progress screen and plain-language errors. Two profiles: **Chat only** (GGUF, no PyTorch, like Unsloth's `UNSLOTH_NO_TORCH`) and **Chat and training** (MLX on Apple Silicon; Unsloth + `--torch-backend=auto` on NVIDIA/AMD). App updates that change the engine re-sync automatically; Settings can add training or reinstall. `temper --setup chat|train [--clean]` installs headlessly, and CI runs it on every OS.
  Tested on macOS (Apple Silicon): both profiles, a read-only app bundle, a network failure and retry, and the update re-sync. Still to do: test the NVIDIA/AMD training profile on real hardware, add an Intel XPU profile, and check free disk space before installing.
- [ ] **0.10 Code signing**: Apple Developer ID plus notarization, Windows Authenticode (Azure Trusted Signing), Linux GPG-signed repos.
- [ ] **0.11 Auto-update**: Tauri updater with signed manifests and stable/nightly channels. Roll back if a staged update fails.
- [ ] **0.12 Distribution channels**: Homebrew cask, winget, Flathub, `.deb`/`.rpm` repos, `curl | sh` and `irm | iex` installers, Docker image (`temper/temper`, plus a ROCm variant).

## Phase 1: Hardware and system

- [x] **1.1 Hardware probe**: OS, CPU, RAM, NVIDIA GPUs (`nvidia-smi`), Apple Silicon unified memory and chip name, AMD (`rocm-smi`), Intel XPU.
- [x] **1.2 Status bar meter**: GPU name, VRAM/RAM in use, engine state, polled while the window is visible.
- [ ] **1.3 Multi-GPU**: list every device, choose which GPUs to use per model and per job, NVLink/P2P detection.
- [ ] **1.4 Backends**: NVIDIA CUDA, AMD ROCm (including RDNA2 APUs), Intel XPU, Apple Metal/MLX, Vulkan, CPU, and NPUs (Intel/AMD/Qualcomm).
- [ ] **1.5 iGPU memory carve-out helper**: explain and adjust shared-memory limits on APUs.
- [ ] **1.6 Doctor**: `temper doctor` and a Settings → Diagnostics page that checks drivers, CUDA/ROCm versions, disk space, ports and engine packages, with fix suggestions.
- [ ] **1.7 VRAM estimator**: before load or training, estimate memory for a given model, quantization, context, LoRA rank and batch size, and warn before running out of memory.
- [ ] **1.8 In-app runtime updates** *(audit)*: update the bundled llama.cpp build and upgrade `transformers` in place when a newly released model needs it, with rollback.

## Phase 2: Model hub and library

- [x] **2.1 Hugging Face search**: search by text, filter by author (default `unsloth`), sort by downloads, likes or recency.
- [x] **2.2 Local library**: list models in the HF cache with size and last-used date.
- [~] **2.3 Download manager**: background downloads with per-file progress and cancel. Still to do: byte-level progress, cancelling mid-file, an explicit resume button.
- [~] **2.4 GGUF quant picker**: list every quant file in a repo (Q2_K … Q8_0, UD-dynamic quants, BF16) with size and an estimate of whether it fits in memory.
- [ ] **2.5 HF token and gated models**: store the token in the OS keychain and guide the user through gated-model access.
- [ ] **2.6 ModelScope mirror**: alternative hub for users in regions where Hugging Face is slow or blocked.
- [ ] **2.7 Model catalog**: curated cards for Qwen, Llama, Gemma, Mistral, gpt-oss, GLM, Kimi, DeepSeek, Phi and others, with recommended settings and chat templates.
- [ ] **2.8 Import local files**: drag in a `.gguf`, a safetensors folder or an MLX folder.
- [ ] **2.9 Disk management**: show cache usage and delete or move models; choose a custom models folder.
- [ ] **2.10 Import from other tools**: chats and settings from Claude Code, Codex, Cursor and Ollama.
- [ ] **2.11 Pre-quantized and unusual checkpoints** *(audit)*: load FP8 (ModelOpt), MXFP4, NVFP4 and compressed-tensors checkpoints, and Mistral-format folders (`params.json` + `consolidated*.safetensors`).
- [ ] **2.12 Library** *(audit)*: one place for uploads, chat attachments, generated images and audio, fine-tuned models and sandbox files, with folders, favourites and rename.

## Phase 3: Inference and chat

- [x] **3.1 Chat UI**: streaming responses, Markdown rendering, stop, regenerate, copy, clear.
- [x] **3.2 llama.cpp runtime**: start and supervise `llama-server` for GGUF models; OpenAI-compatible streaming. Temper downloads a pinned llama.cpp release (checksummed binaries for each OS and GPU backend) with one click on the Models page, so nobody needs Homebrew or winget.
- [x] **3.3 Generation settings**: temperature, top-p, top-k, min-p, repeat penalty, max tokens, system prompt.
- [ ] **3.4 Chat history**: threads stored in SQLite, search, rename, pin, folders/projects, export as Markdown or JSON.
- [ ] **3.5 Reasoning view**: collapsible thinking blocks and a per-model reasoning-effort control.
- [ ] **3.6 Presets**: saved generation presets per model; import recommended settings from the model card.
- [ ] **3.7 Compare mode**: send one prompt to two or more models side by side (for example base vs fine-tuned).
- [ ] **3.8 Prompt queue**: queue follow-up prompts while one is generating; edit and reorder the queue.
- [ ] **3.9 Multimodal input**: images (vision models), PDFs, DOCX, CSV and audio attachments.
- [ ] **3.10 Auto-compaction**: rolling context window that summarises old turns when context fills.
- [ ] **3.11 Speculative decoding**: draft-model pairing for faster generation.
- [ ] **3.12 KV-cache controls**: context length, KV quantization, flash attention, layer offload.
- [ ] **3.13 Extra engines**: MLX (Apple), transformers, and opt-in vLLM and SGLang (Linux + NVIDIA/AMD) with tensor and pipeline parallel.
- [ ] **3.14 Multiple loaded models**: load several models and route between them; unload when idle.
- [ ] **3.15 Find in chat, keyboard shortcuts, interface zoom.**
- [ ] **3.16 Localisation (i18n)**: string catalog with parity checks; start with English, Hindi, Spanish, Chinese and Japanese.
- [ ] **3.17 Resumable generations** *(audit)*: replies keep generating in the engine if the window closes; reopening the chat shows the finished answer.
- [ ] **3.18 Prompt library** *(audit)*: save, name and reuse prompts.
- [ ] **3.19 AGENTS.md instructions** *(audit)*: a global instructions file plus each project folder's `AGENTS.md`, read the way Codex reads them.
- [ ] **3.20 Keep chat context across GPU swaps** *(audit)*: save and restore the KV cache when image or video generation temporarily takes the GPU.
- [ ] **3.21 Context-fit explanations** *(audit)*: when a prompt doesn't fit the context window, say why and what to change.

## Phase 4: Agents, tools and MCP

- [ ] **4.1 Tool calling**: native function calling for models that support it, with repair of malformed tool calls.
- [ ] **4.2 Code execution sandbox**: run Python/JS in an isolated sandbox with per-tool approval (Ask / Auto / Off).
- [ ] **4.3 MCP client**: add MCP servers (stdio, HTTP, SSE), browse tools, approve them; search over large tool catalogs (BM25) so they fit in context.
- [ ] **4.4 Temper as an MCP server**: expose local models and jobs to other agents.
- [ ] **4.5 Agent skills**: `SKILL.md` folders and `@skill` mentions in chat.
- [ ] **4.6 Built-in browser agent**: fetch, read and act on web pages.
- [ ] **4.7 Ready-made integrations**: Blender MCP, file system, GitHub.
- [~] **4.8 Connect coding agents**: copy-paste config for Codex, OpenCode, VS Code (Continue), Aider, Python and curl, pointing at the local `/v1` endpoint. Claude Code waits on 10.2; one-click setup waits on 4.9.
- [ ] **4.9 `temper start <agent>` CLI** that does the same from a terminal.
- [ ] **4.10 Terminal tool and OS sandbox** *(audit)*: a terminal tool alongside Python, and a page showing which OS sandbox (Seatbelt, bubblewrap, AppContainer) protects tool runs on this machine.
- [ ] **4.11 `temper mcp add <agent>`** *(audit)*: register Temper's MCP server with Claude Code, Codex and other agents, and check it with `temper doctor --mcp`.

## Phase 5: Search, research and RAG

- [ ] **5.1 Web search tool**: private web search with configurable providers.
- [ ] **5.2 Deep research**: multi-step research runs with citations, a review screen and a report export.
- [ ] **5.3 Knowledge base (RAG)**: add folders and files (PDF, DOCX, PPTX, XLSX, Markdown, code); chunk, embed (local embedding GGUF) and index in SQLite + FTS5 + vectors; hybrid retrieval with citations.
- [ ] **5.4 Folder sync**: watch folders and re-index on change, honouring `.gitignore`.
- [ ] **5.5 OCR for scanned PDFs**: vision-model transcription first, Tesseract as a fallback.
- [ ] **5.6 YouTube transcripts and URL ingestion.**

## Phase 6: Fine-tuning (the core of Unsloth)

- [x] **6.1 Training wizard UI**: choose base model, dataset, method and hyperparameters, then start, stop and watch.
- [~] **6.2 SFT with LoRA / QLoRA**: unsloth `FastLanguageModel` with 4-bit loading, `get_peft_model`, TRL `SFTTrainer`.
- [x] **6.3 Live metrics**: loss chart, learning rate, step/epoch, tokens/s, ETA and a log console over SSE.
- [ ] **6.4 Full fine-tuning and continued pretraining** (including embedding and `lm_head` training).
- [ ] **6.5 Preference training**: DPO, ORPO, KTO, SimPO.
- [ ] **6.6 Reinforcement learning**: GRPO, GSPO, Dr. GRPO, DAPO, with reward-function editor and vLLM fast generation.
- [ ] **6.7 FP8 and NVFP4 training** on supported GPUs; QAT (quantization-aware training).
- [ ] **6.8 Vision fine-tuning**: VLMs such as Qwen-VL, Gemma vision and Llama vision.
- [ ] **6.9 TTS and audio fine-tuning**: Orpheus, Sesame CSM, Whisper (STT).
- [ ] **6.10 Embedding model fine-tuning**: sentence-transformers / EmbeddingGemma.
- [ ] **6.11 Diffusion fine-tuning**: image and video LoRA (Flux, Qwen-Image, Z-Image and others).
- [ ] **6.12 Decision/classifier models**: train a classifier head from a language model.
- [ ] **6.13 Long-context training**: gradient checkpointing ("unsloth" mode), tiled MLP, context parallel.
- [ ] **6.14 Multi-GPU training**: DDP and FSDP2 (including QLoRA on FSDP2).
- [x] **6.15 Apple Silicon training** through MLX-LM LoRA: rank, alpha and dropout via an mlx-lm config, gradient accumulation and checkpointing, live loss/validation/tokens-per-second in the UI. Verified 2026-10-24 on an M1 Pro (16 GB): Qwen3-4B-Instruct 4-bit, 500 FineTome rows, 60 steps in about 12 minutes; validation loss 1.62 → 0.81; the 29 MB adapter loads with `mlx_lm generate`.
- [ ] **6.16 Chat templates**: automatic template detection, `train_on_responses_only` masking, template override.
- [ ] **6.17 Token preview**: show exactly what the model will see for a sample, with masked tokens dimmed.
- [ ] **6.18 Evaluation**: held-out eval set, eval loss, early stopping, sample generations during training.
- [ ] **6.19 Checkpoints and resume**: list checkpoints, resume from one, compare runs.
- [ ] **6.20 Training history**: every run stored with config, metrics, dataset hash and provenance; re-run with one click.
- [ ] **6.21 VRAM-aware defaults**: choose batch size, rank and context automatically from the hardware probe.
- [ ] **6.22 Remote and cloud training**: launch the same job on a remote machine, Colab, Kaggle or a cloud GPU and stream its metrics back.
- [ ] **6.23 Text-diffusion language models** *(audit)*: train and run diffusion LMs such as DiffusionGemma and Dream.
- [ ] **6.24 Memory-saving optimizers** *(audit)*: 8-bit AdamW, Q-GaLore, FlashAdamW and optimizer-state CPU offload.
- [ ] **6.25 Drafter and MTP-head fine-tuning** *(audit)*: fine-tune a model's multi-token-prediction head or a draft model so speculative decoding still works after fine-tuning.
- [ ] **6.26 Checkpoint preview** *(audit)*: chat with any checkpoint during or after training through a per-checkpoint `/v1` endpoint.
- [ ] **6.27 Raw-text continued pretraining** *(audit)*: chunk plain documents into training text, with no chat format needed.

## Phase 7: Datasets and Data Recipes

- [x] **7.1 Dataset preview**: open a Hugging Face dataset or a local CSV, JSON or JSONL and see rows and columns.
- [ ] **7.2 Column mapping**: map columns to chat roles (Alpaca, ShareGPT, ChatML, OpenAI messages) with live preview.
- [ ] **7.3 Format conversion and cleaning**: deduplicate, filter by length, split train/eval, shuffle.
- [ ] **7.4 Data Recipes**: build datasets from PDFs, DOCX, CSV, web pages and GitHub repos using a visual pipeline of steps.
- [ ] **7.5 Synthetic data generation**: use a local or cloud model to generate Q&A, conversations or reasoning traces from source documents, with validators.
- [ ] **7.6 Push to Hub**: upload finished datasets to Hugging Face.
- [ ] **7.7 S3 and remote sources.**

## Phase 8: Export and deploy

- [x] **8.1 Export UI**: choose a checkpoint and one or more formats.
- [~] **8.2 LoRA adapter, merged 16-bit and merged 4-bit** (`save_pretrained_merged`).
- [~] **8.3 GGUF export**: q4_k_m, q5_k_m, q8_0, f16 and other quant types. On Apple Silicon it fuses with mlx-lm, converts with llama.cpp's `convert_hf_to_gguf.py` (in an isolated uv environment), then runs `llama-quantize`; checked end to end. On NVIDIA/AMD it uses Unsloth's `save_pretrained_gguf` (not yet run).
- [ ] **8.4 Dynamic quants**: Unsloth Dynamic (UD) style quants with an importance matrix.
- [ ] **8.5 FP8, NVFP4, TorchAO, OpenVINO and MLX exports.**
- [ ] **8.6 Ollama export**: write a Modelfile with the right template and `ollama create`.
- [ ] **8.7 Push to Hugging Face**: model card generated from the training run.
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
- [ ] **10.2 Anthropic-compatible API**: `/v1/messages`, so Claude Code can use local models directly.
- [ ] **10.3 API keys**: create, name, revoke; usage tracking and an API monitor page.
- [ ] **10.4 LAN access**: serve to other devices on the home network with a QR code.
- [ ] **10.5 Secure remote access**: Cloudflare Tunnel HTTPS link, with admin password and an option to turn tools off.
- [ ] **10.6 Accounts**: admin password, multiple users with separate chats and jobs, reset password from the CLI.
- [ ] **10.7 Cloud providers**: connect OpenAI, Anthropic, Google, OpenRouter or a ChatGPT/Codex subscription and use them next to local models; keys stored in the OS keychain.
- [ ] **10.8 Headless server mode**: `temper serve` with systemd unit, Docker image and a browser UI.
- [ ] **10.9 Decisions API** *(audit)*: typed answers (yes/no, choice, score) from decision models or a saved connection, like Unsloth's System One API.
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
