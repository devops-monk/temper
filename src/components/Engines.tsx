import { Download, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { useEngines } from "./LoadButton";
import { Badge, Button, ProgressBar, Section } from "./ui";

const ABOUT: Record<string, string> = {
  "llama.cpp": "Runs GGUF models on any computer. Chat's default.",
  mlx: "Runs MLX and Hugging Face model folders on Apple Silicon, and fine-tunes from this Mac before exporting.",
  transformers: "Runs Hugging Face model folders with PyTorch. Slower, but takes almost any model.",
  vllm: "Fast serving on NVIDIA and AMD GPUs; splits big models across several GPUs.",
  sglang: "Fast serving on NVIDIA and AMD GPUs; splits big models across several GPUs.",
};

/** The inference engines, with installs for the opt-in ones (engine engines.py). Each installs into its own environment. */
export function Engines() {
  const { engines, refresh } = useEngines();
  const { jobs, refreshJobs } = useApp();
  const [started, setStarted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const existing = jobs.find((j) => j.kind === "engine-install" && j.status === "running");
  const jobId = started ?? existing?.id ?? null;
  const { status, events } = useJob(jobId);
  const installing = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const lastLine = [...events].reverse().find((e) => e.event === "log")?.line as string | undefined;
  const failure = events.find((e) => e.event === "status" && e.status === "failed")?.error as string | undefined;

  useEffect(() => {
    if (status && status !== "running") {
      refresh();
      void refreshJobs();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function install(id: string) {
    setError(null);
    try {
      const job = await api<Job>(`/api/runtime/engines/${id}/install`, { method: "POST" });
      setStarted(job.id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(id: string, name: string) {
    if (!confirm(`Remove ${name}? Its environment is deleted; you can install it again later.`)) return;
    setError(null);
    try {
      await api(`/api/runtime/engines/${id}`, { method: "DELETE" });
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!engines?.length) return null;
  return (
    <Section title="Inference engines">
      <ul className="divide-y divide-line rounded-lg border border-line bg-surface">
        {engines.map((e) => (
          <li key={e.id} className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">{e.name}</span>
                {e.available ? <Badge tone="ok">Ready</Badge> : e.unsupported ? <Badge>Not for this computer</Badge> : null}
              </div>
              <p className="text-[12px] text-muted">{e.unsupported ?? ABOUT[e.id]}</p>
            </div>
            {e.installable && !e.available && (
              <Button size="sm" onClick={() => void install(e.id)} disabled={installing}>
                <Download className="size-3.5" /> Install
              </Button>
            )}
            {e.installable && e.available && (
              <button className="text-muted hover:text-danger" onClick={() => void remove(e.id, e.name)} title={`Remove ${e.name}`}>
                <Trash2 className="size-3.5" aria-label={`Remove ${e.name}`} />
              </button>
            )}
          </li>
        ))}
      </ul>
      {installing && (
        <div className="mt-3 flex flex-col gap-1.5">
          <ProgressBar value={progress ?? null} />
          <p className="truncate font-mono text-[11.5px] text-muted" title={lastLine}>
            {lastLine ?? "Starting…"}
          </p>
          <span>
            <Button size="sm" variant="ghost" onClick={() => void api(`/api/jobs/${jobId}/cancel`, { method: "POST" })}>
              Cancel
            </Button>
          </span>
        </div>
      )}
      {(error || (status === "failed" && failure)) && <p className="mt-2 text-[12px] text-danger">{error ?? failure}</p>}
      <p className="mt-2 text-[12px] text-muted">Hugging Face and MLX model folders get a Load button on the Models page once an engine that runs them is ready.</p>
    </Section>
  );
}
