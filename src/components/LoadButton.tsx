import { Play } from "lucide-react";
import { useState } from "react";
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

/** Loads a GGUF in chat, asking first when the memory estimate says it won't fit. */
export function LoadButton({ path, name }: { path: string; name: string }) {
  const { refreshRuntime, go, runtime } = useApp();
  const [error, setError] = useState<string | null>(null);
  const loaded = runtime?.state === "ready" && runtime.model === name;
  return (
    <span className="flex items-center gap-2">
      {error && <span className="max-w-[260px] truncate text-[12px] text-danger" title={error}>{error}</span>}
      <Button
        size="sm"
        variant={loaded ? "secondary" : "primary"}
        onClick={async () => {
          if (loaded) return go("chat");
          setError(null);
          // At the model's saved settings (context, KV cache, draft model).
          const est = await api<ChatEstimate>("/api/memory/chat", { json: { path } }).catch(() => null);
          if (est?.verdict === "too_large") {
            const tips = est.advice.length ? `\n\n${est.advice.join("\n")}` : "";
            if (!confirm(`${name} needs about ${bytes(est.total)}, and this computer can give it ${bytes(est.budget.total)}. Loading may fail or make your computer very slow.${tips}\n\nLoad anyway?`)) return;
          }
          try {
            await api("/api/runtime/load", { json: { path, name } });
            await refreshRuntime();
            go("chat");
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      >
        <Play className="size-3.5" /> {loaded ? "Open chat" : "Load"}
      </Button>
    </span>
  );
}
