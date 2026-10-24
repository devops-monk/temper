export type EngineStatus = "starting" | "running" | "crashed" | "missing_uv" | "stopped";

export interface EngineInfo {
  url: string;
  token: string;
  status: EngineStatus;
  detail: string | null;
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
    trainer: "unsloth" | "mlx" | null;
    torch: boolean;
    unsloth: boolean;
    mlx: boolean;
  };
}

export interface RuntimeStatus {
  state: "idle" | "loading" | "ready" | "failed";
  model: string | null;
  error: string | null;
  engine: string;
  available: boolean;
}

export type JobStatus = "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: "download" | "train" | "export";
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
  name: string;
  size: number;
  path?: string;
}

export interface LocalModel {
  id: string;
  size: number;
  last_used: number | null;
  path: string | null;
  gguf: RepoFile[];
  files: number;
  safetensors: boolean;
}

export interface Run {
  id: string;
  base_model: string;
  dataset: string;
  backend: string;
  adapter: string | null;
  path: string;
}

export type Page = "chat" | "models" | "train" | "datasets" | "export" | "connect" | "settings";
