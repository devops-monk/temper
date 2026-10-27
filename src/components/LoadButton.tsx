import { Layers, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import type { Budget, Verdict } from "./MemoryEstimate";
import { Button } from "./ui";

/** The context chat loads a model with unless its settings say otherwise (engine model_settings.DEFAULTS). */
export const CHAT_CONTEXT = 8192;

export interface ChatEstimate {
  weights: number;
  kv_cache: number;
  overhead: number;
  total: number;
  /** A draft model and its cache, for speculative decoding. */
  draft?: number;
  context: number;
  budget: Budget;
  verdict: Verdict;
  advice: string[];
}

/** An inference engine (engine engines.py): llama.cpp for GGUFs; MLX, Transformers, vLLM or SGLang for model folders. */
export interface Engine {
  id: "llama.cpp" | "mlx" | "transformers" | "vllm" | "sglang";
  name: string;
  available: boolean;
  /** Why it can't run on this computer. */
  unsupported: string | null;
  installable: boolean;
  formats: ("gguf" | "mlx" | "hf")[];
}

let enginesCache: Promise<Engine[]> | null = null;

/** The engines, fetched once per page load; `refresh` after installing or removing one. */
export function useEngines(): { engines: Engine[] | null; refresh: () => void } {
  const [engines, setEngines] = useState<Engine[] | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    enginesCache ??= api<{ engines: Engine[] }>("/api/runtime/engines").then((r) => r.engines, () => []);
    let stale = false;
    enginesCache.then((e) => !stale && setEngines(e));
    return () => {
      stale = true;
    };
  }, [tick]);
  return {
    engines,
    refresh: () => {
      enginesCache = null;
      setTick((t) => t + 1);
    },
  };
}

/** The engine that would run a model folder of this format, if one is installed. */
export function folderEngine(engines: Engine[] | null, format: "mlx" | "hf"): Engine | null {
  return engines?.find((e) => e.id !== "llama.cpp" && e.available && e.formats.includes(format)) ?? null;
}

/** Loads a model in chat, asking first when the memory estimate says it won't fit. A GGUF runs on llama.cpp; a folder (or a
 * cached Hugging Face id) on `engine`, with a fine-tuning run's `adapter` on top. */
export function LoadButton({ path, name, engine, adapter, label = "Load" }: { path: string; name: string; engine?: string; adapter?: string; label?: string }) {
  const { refreshRuntime, go, runtime } = useApp();
  const [error, setError] = useState<string | null>(null);
  const loaded = runtime?.state === "ready" && runtime.model === name;
  const beside = runtime?.alongside?.some((m) => m.path === path);
  // Another model is in chat: this one can load next to it rather than replacing it.
  const canAdd = !loaded && !beside && (runtime?.state === "ready" || runtime?.state === "loading");

  async function load(alongside: boolean) {
    setError(null);
    // At the model's saved settings (context, KV cache, draft model).
    const est = await api<ChatEstimate>("/api/memory/chat", { json: { path } }).catch(() => null);
    if (est?.verdict === "too_large") {
      const tips = est.advice.length ? `\n\n${est.advice.join("\n")}` : "";
      if (!confirm(`${name} needs about ${bytes(est.total)}, and this computer can give it ${bytes(est.budget.total)}. Loading may fail or make your computer very slow.${tips}\n\nLoad anyway?`)) return;
    }
    try {
      await api("/api/runtime/load", { json: { path, name, engine, adapter, alongside } });
      await refreshRuntime();
      if (!alongside) go("chat");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <span className="flex items-center gap-2">
      {error && <span className="max-w-[260px] truncate text-[12px] text-danger" title={error}>{error}</span>}
      {canAdd && (
        <Button size="sm" variant="ghost" onClick={() => void load(true)} title={`Load next to ${runtime?.model}; both answer on Temper's API, and you can switch chat between them`}>
          <Layers className="size-3.5" /> Alongside
        </Button>
      )}
      <Button size="sm" variant={loaded ? "secondary" : "primary"} onClick={() => (loaded ? go("chat") : void load(false))}>
        <Play className="size-3.5" /> {loaded ? "Open chat" : beside ? "Use in chat" : label}
      </Button>
    </span>
  );
}
