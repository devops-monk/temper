<p align="center"><img src="assets/icon.svg" width="112" alt="Temper icon"></p>

<h1 align="center">Temper</h1>
<p align="center">Run, fine-tune and ship open-weight AI models on your own computer.</p>

Temper is a desktop app for macOS, Windows and Linux. Download a model from Hugging Face, chat with it,
fine-tune it on your own data (on NVIDIA and AMD GPUs, or with MLX on Apple
Silicon), export it to GGUF, and serve it to coding agents through an OpenAI-compatible API.

Tempering is the step after hardening steel. It tunes the metal for a specific job, which is what
fine-tuning does to a base model.

## What works today

- **Chat** with any GGUF model through llama.cpp, with streaming output and generation settings
- **Models**: search Hugging Face, pick a GGUF quant with a memory-fit check, download in the background, load
- **Train**: LoRA, QLoRA and full fine-tuning on NVIDIA and AMD GPUs; LoRA with MLX on Apple Silicon; live loss chart and logs
- **Datasets**: preview Hugging Face or local CSV/JSON/JSONL datasets and detect their chat format
- **Recipes**: generate training data with a model from PDFs, Word files, web pages, GitHub repos or topics you choose, then train on it
- **Export**: LoRA adapter, merged 16-bit/4-bit and GGUF (q4_k_m, q5_k_m, q8_0 and more)
- **Connect**: ready-made config for Codex, OpenCode, Continue, Aider, Python and curl
- **Hardware**: NVIDIA, AMD and Apple Silicon detection with a live memory meter

New to Temper? The [guides](docs/README.md) go from installing it to serving a model you trained.
Everything else is planned feature by feature in [PLAN.md](PLAN.md).

## From a terminal

The app's own binary takes these commands while Temper is open, or while `temper serve` runs (on macOS
it's `/Applications/Temper.app/Contents/MacOS/temper`; alias it as `temper`):

```bash
temper models               # the models on this computer
temper load Qwen3-4B-GGUF   # load one (a name, a GGUF file name or a path; --quant, --context)
temper ps                   # what's loaded; temper unload <model> or --all
temper chat "Hello"         # ask the loaded model once; with no prompt, chat until /bye
temper train tutor.yaml     # fine-tune from a YAML config (--set key=value, --dry-run), following the run
temper runs                 # past training runs
temper export <run>         # export a run (--format gguf|mlx|lora|..., --quant q4_k_m, --imatrix)
temper serve                # run the engine without the app, for a server or CI (--host, --port)
temper start claude         # or codex, opencode, aider: starts the agent on the model loaded in Temper
temper mcp add claude       # or codex, cursor, vscode: gives the agent Temper's MCP tools
temper doctor               # checks drivers, disk, ports and engine packages (--mcp checks the MCP server)
```

A training config uses the Train page's setting names:

```yaml
name: math-tutor
base_model: mlx-community/Qwen3-4B-Instruct-2507-4bit
dataset: ~/data/gsm8k.jsonl       # a file, a Hugging Face dataset or an https:// / s3:// link
method: lora
max_steps: 200
learning_rate: 2e-4
```

## Develop

You need Node 20+, Rust (stable) and [uv](https://docs.astral.sh/uv/).

```bash
npm install
npm run tauri dev        # UI + desktop shell; the shell starts the Python engine with uv
```

Optional pieces:

```bash
brew install llama.cpp                 # chat with GGUF models (or: winget install llama.cpp)
cd engine && uv sync --extra train     # training on NVIDIA / AMD / Intel GPUs
cd engine && uv sync --extra mlx       # training on Apple Silicon
```

Development builds run the engine straight from `engine/` with `uv run`. To try the first-run setup
that installed builds use, run `TEMPER_BOOTSTRAP=1 npm run tauri dev`, or install headlessly with
`src-tauri/target/debug/temper --setup chat` (or `train`; add `--clean` to rebuild from scratch).
To preview the setup screens in a browser, open `http://localhost:1420/#setup=choose` (also
`installing` and `failed`) while `npm run dev` is running.

To work on the UI in a browser, start the engine yourself with `cd engine && uv run temper-engine --token dev`,
then run `npm run dev` and open http://localhost:1420.

Tests: `cd engine && uv run --extra dev pytest` (engine), `npm test` (interface logic, Vitest) and
`npm run test:e2e` (the interface in Chrome against a fresh engine, Playwright; it starts both itself).
Typecheck and build the UI: `npm run build`.


## Icons

`assets/icon.svg` is the master. `npm run icons` renders every platform size into `src-tauri/icons/`.

## Release

Bump `version` in `package.json`, `src-tauri/tauri.conf.json` and `src-tauri/Cargo.toml`, commit, then tag and push:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The Release workflow checks that the tag matches the version, builds installers for macOS, Windows and Linux
(x64 and ARM64), and publishes the GitHub release once every platform has built. If a platform fails, the release
stays a draft; re-run the failed job to finish it. Running the workflow by hand from the Actions tab builds the
installers as artifacts without releasing anything.

Releases also publish signed update bundles and a `latest.json` that installed copies of Temper check for updates.
The workflow signs them with the repository secrets `TAURI_SIGNING_PRIVATE_KEY` and
`TAURI_SIGNING_PRIVATE_KEY_PASSWORD`; the matching public key is `plugins.updater.pubkey` in
`src-tauri/tauri.conf.json`. Keep the private key safe: without it, existing installs can't receive updates.

## License

Apache-2.0. The licences of everything compiled into the installers are in the app under
Settings → About → Open-source licences, generated by `npm run licenses`.

GPU training libraries are installed from PyPI on the user's machine by first-run setup and run in a
separate process; they are never bundled. Some are partly AGPL-3.0, so get legal advice before
bundling them or running Temper as a hosted service.
