import type { EngineInfo, SetupState } from "./types";

/**
 * Development only: `#setup=choose|installing|failed` in a browser shows the first-run screens
 * with sample data, so they can be designed without wiping a real install.
 */
export function mockSetupInfo(): EngineInfo | null {
  const mode = location.hash.match(/setup=(\w+)/)?.[1];
  if (!mode) return null;
  const base: SetupState = {
    hardware: { kind: "apple", name: "Apple M1 Pro", can_train: true },
    options: [
      { id: "train", label: "Chat and training", description: "Adds MLX for fine-tuning models on Apple Silicon.", size: "about 600 MB", available: true, recommended: true },
      { id: "chat", label: "Chat only", description: "Run GGUF models with llama.cpp. You can add training later in Settings.", size: "about 150 MB", available: true, recommended: false },
    ],
    installed: null,
    profile: "train",
    steps: [
      { id: "uv", label: "Download the package manager", state: "done" },
      { id: "python", label: "Install Python 3.12", state: "done" },
      { id: "engine", label: "Install the engine and MLX", state: mode === "failed" ? "failed" : "running" },
    ],
    progress: null,
    log: [
      "== Install the engine and MLX",
      "$ uv --color never --no-progress sync --frozen --no-dev --no-editable --python 3.12 --project …/engine --extra mlx",
      "Using CPython 3.12.13",
      "Creating virtual environment at: …/engine-venv",
      " Downloaded numpy",
      " Downloaded transformers",
      ...(mode === "failed" ? ["error: Failed to fetch: `https://pypi.org/simple/mlx-metal/`", "  Caused by: dns error: failed to lookup address information"] : [" Downloading mlx-metal (31.2MiB)"]),
    ],
    error: mode === "failed" ? "Install the engine and MLX failed. Temper couldn't download what it needs. Check your internet connection or proxy settings and try again." : null,
  };
  const status = mode === "choose" ? "needs_setup" : mode === "failed" ? "setup_failed" : "installing";
  return { url: "", token: "", status, detail: null, setup: base };
}
