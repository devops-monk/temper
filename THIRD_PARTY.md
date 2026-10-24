# Third-party software

Temper is licensed under Apache-2.0 (see `LICENSE`). It uses, but does not include source code from:

| Project | License | How Temper uses it |
|---|---|---|
| [Unsloth](https://github.com/unslothai/unsloth) (core library) | Apache-2.0 and AGPL-3.0 (mixed; see below) | Fine-tuning and export on NVIDIA and AMD GPUs; installed from PyPI by first-run setup, never bundled |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | MIT | Fine-tuning on Apple Silicon; installed with the `mlx` extra |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Local GGUF chat (`llama-server`), GGUF conversion (`convert_hf_to_gguf.py`, downloaded at runtime) and quantization |
| [Tauri](https://tauri.app) | Apache-2.0 / MIT | Desktop shell and installers |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model search and downloads |
| [Bricolage Grotesque](https://github.com/ateliertriay/bricolage) | OFL-1.1 | Interface typeface |
| [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) | OFL-1.1 | Code and log typeface |
| [Lucide](https://lucide.dev) | ISC | Interface icons |

Unsloth's package metadata says Apache-2.0, but many files in its core library (74 of 171 when
checked on 2026-10-24) and all of Unsloth Studio, the web UI, are AGPL-3.0. Temper does not copy
code from either. It installs Unsloth on the user's machine and runs it in a separate process.
Keep it that way, and get legal advice before bundling Unsloth or running Temper as a hosted service.
