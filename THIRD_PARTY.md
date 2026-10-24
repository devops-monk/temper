# Third-party software

Temper is licensed under Apache-2.0 (see `LICENSE`). It uses, but does not include source code from:

| Project | License | How Temper uses it |
|---|---|---|
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | MIT | Fine-tuning on Apple Silicon; installed with the `mlx` extra |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Local GGUF chat (`llama-server`), GGUF conversion (`convert_hf_to_gguf.py`, downloaded at runtime) and quantization |
| [Tauri](https://tauri.app) | Apache-2.0 / MIT | Desktop shell and installers |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model search and downloads |
| [Bricolage Grotesque](https://github.com/ateliertriay/bricolage) | OFL-1.1 | Interface typeface |
| [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) | OFL-1.1 | Code and log typeface |
| [Lucide](https://lucide.dev) | ISC | Interface icons |

GPU training libraries are installed from PyPI on the user's machine by first-run setup and run in a
separate process; they are never bundled. Some are partly AGPL-3.0, so get legal advice before
bundling them or running Temper as a hosted service.
