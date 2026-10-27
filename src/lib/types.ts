export type EngineStatus = "starting" | "running" | "crashed" | "missing_uv" | "stopped" | "needs_setup" | "installing" | "setup_failed";

export type SetupProfile = "chat" | "train";

export interface SetupOption {
  id: SetupProfile;
  label: string;
  description: string;
  size: string;
  available: boolean;
  recommended: boolean;
}

export interface SetupStep {
  id: string;
  label: string;
  state: "pending" | "running" | "done" | "failed";
}

export interface SetupState {
  hardware: { kind: "apple" | "nvidia" | "amd" | "none"; name: string | null; can_train: boolean } | null;
  options: SetupOption[];
  installed: SetupProfile | null;
  profile: SetupProfile | null;
  steps: SetupStep[];
  progress: number | null;
  log: string[];
  error: string | null;
}

export interface EngineInfo {
  url: string;
  token: string;
  status: EngineStatus;
  detail: string | null;
  /** Present in installed builds, which manage their own engine environment. */
  setup: SetupState | null;
}

export interface Gpu {
  index: number;
  vendor: "nvidia" | "amd" | "apple" | "intel";
  name: string;
  memory_total: number;
  memory_used: number;
  utilization: number | null;
  backend: string;
  unified_memory?: boolean;
}

export interface SystemInfo {
  os: { name: string; release: string; machine: string };
  cpu: { name: string; cores: number; threads: number };
  memory: { total: number; used: number };
  disk: { total: number; free: number };
  gpus: Gpu[];
  backend: "cuda" | "rocm" | "metal" | "xpu" | "cpu";
  capabilities: {
    gguf_inference: boolean;
    trainer: "pytorch" | "mlx" | null;
    torch: boolean;
    mlx: boolean;
  };
}

export interface RuntimeStatus {
  state: "idle" | "loading" | "ready" | "failed";
  model: string | null;
  path?: string | null;
  error: string | null;
  /** What runs it: llama.cpp, mlx, transformers, vllm or sglang. */
  engine: string;
  available: boolean;
  /** What the loaded model's chat template allows; null until it's ready. */
  reasoning?: { thinks: boolean; toggle: boolean; effort: boolean } | null;
  /** The vision or audio projector loaded with the model, if any. */
  projector?: string | null;
  /** What the model takes besides text; null until it's ready. */
  modalities?: { vision?: boolean; audio?: boolean } | null;
  /** Context length in tokens, as llama-server set it up. */
  context?: number | null;
  /** What it was loaded with (engine model_settings.py); the draft is a file name. */
  settings?: { context: number; gpu_layers: number; flash_attention: boolean; cache_type: string; speculation: string; draft: string | null; draft_max: number } | null;
  /** The fine-tuning adapter loaded on top, for a run chatted with before exporting. */
  adapter?: string | null;
  /** Seconds since its last request. */
  idle?: number | null;
  /** Models loaded alongside chat's (engine loaded.py); each answers on /v1 by its name. */
  alongside?: { model: string | null; path: string; state: RuntimeStatus["state"]; engine: string; error: string | null; idle: number | null }[];
}

export type JobStatus = "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: "download" | "train" | "export" | "recipe" | "prepare" | "convert" | "move" | "engine-install";
  title: string;
  status: JobStatus;
  progress: number | null;
  error: string | null;
  result: Record<string, unknown>;
  created: number;
}

export interface JobEvent {
  seq: number;
  event: string;
  t: number;
  [key: string]: unknown;
}

export interface HubModel {
  id: string;
  downloads: number;
  likes: number;
  updated: string | null;
  pipeline: string | null;
  tags: string[];
  gated: boolean;
}

export interface RepoFile {
  /** The file, or a split GGUF's first part (llama.cpp loads the others from it). */
  name: string;
  /** All parts together. */
  size: number;
  path?: string;
  /** Every part's name; one entry for an unsplit file. */
  parts: string[];
  /** False while some parts of a split GGUF are missing. */
  complete: boolean;
}

export interface LocalModel {
  id: string;
  size: number;
  last_used: number | null;
  path: string | null;
  gguf: RepoFile[];
  files: number;
  safetensors: boolean;
  /** Added from the user's own files rather than downloaded. */
  imported?: boolean;
  kind?: "gguf" | "safetensors" | "mlx" | "mistral";
  /** Why an imported model can't be found any more. */
  missing?: string;
  warnings?: string[];
  /** How the checkpoint was quantized before it was published (FP8, NVFP4, …); null for 16-bit weights. */
  quantization?: { label: string; convertible: boolean; note?: string } | null;
  /** llama.cpp's converter can turn it into a GGUF for chat. */
  convertible?: boolean;
  /** A model folder that generates text and has a chat template, so MLX or another engine can chat with it. */
  chat?: boolean;
}

export interface Run {
  id: string;
  base_model: string;
  dataset: string;
  backend: string;
  adapter: string | null;
  path: string;
}

export type Page = "chat" | "models" | "library" | "train" | "datasets" | "recipes" | "export" | "connect" | "settings";

/** A recipe column in data-designer's format; the editor knows the common fields of each type. */
export interface RecipeColumn {
  column_type: "sampler" | "llm-text" | "llm-structured" | "llm-code" | "llm-judge" | "expression" | "validation";
  name: string;
  drop?: boolean;
  [key: string]: unknown;
}

export interface RecipeSeed {
  type: "documents" | "file" | "hf";
  sources?: string[];
  chunk_size?: number;
  chunk_overlap?: number;
  path?: string;
  sampling?: "ordered" | "shuffle";
}

export interface RecipeModel {
  alias: string;
  provider: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
}

export interface RecipeProvider {
  name: string;
  endpoint: string;
  api_key?: string;
}

export interface Recipe {
  id: string | null;
  name: string;
  description: string;
  seed: RecipeSeed | null;
  models: RecipeModel[];
  providers: RecipeProvider[];
  columns: RecipeColumn[];
  processors: Record<string, unknown>[];
  template?: string | null;
  updated?: number;
}

export interface GeneratedDataset {
  name: string;
  path: string;
  rows: number;
  size: number;
  modified: number;
}
