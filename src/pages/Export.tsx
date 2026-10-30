import { FolderOpen, MessageSquare, PackageOpen } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { AddToOllama } from "../components/AddToOllama";
import { Badge, Button, EmptyState, Notice, PageHeader, ProgressBar, Section, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { useApp } from "../lib/store";
import { useJob } from "../lib/useJob";
import type { Job, Run } from "../lib/types";

const FORMATS = [
  { id: "gguf", label: "GGUF", note: "For llama.cpp, Ollama, LM Studio and Temper's chat." },
  { id: "lora", label: "LoRA adapter", note: "Just the trained adapter. Small; needs the base model to run." },
  { id: "merged_16bit", label: "Merged 16-bit", note: "Full safetensors model for vLLM, SGLang or transformers." },
  { id: "merged_4bit", label: "Merged 4-bit", note: "Bitsandbytes 4-bit weights. NVIDIA only." },
];
const QUANTS = [
  { id: "q4_k_m", note: "Recommended balance" },
  { id: "q5_k_m", note: "Higher quality" },
  { id: "q8_0", note: "Near lossless" },
  { id: "f16", note: "Unquantized" },
  { id: "q3_k_m", note: "Smaller" },
  { id: "q2_k", note: "Smallest" },
];

async function reveal(path: string) {
  if (!inTauri) return;
  const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
  await revealItemInDir(path);
}

/** Loads an exported GGUF into the chat runtime and opens Chat. */
function ChatButton({ path }: { path: string }) {
  const { refreshRuntime, go } = useApp();
  const [error, setError] = useState<string | null>(null);
  const quant = path.match(/-([A-Z0-9_]+)\.gguf$/)?.[1];
  return (
    <>
      <Button
        size="sm"
        variant="primary"
        onClick={async () => {
          setError(null);
          try {
            await api("/api/runtime/load", { json: { path } });
            await refreshRuntime();
            go("chat");
          } catch (e) {
            setError(String(e instanceof Error ? e.message : e));
          }
        }}
      >
        <MessageSquare className="size-3.5" /> Chat with {quant ?? "it"}
      </Button>
      {error && <span className="text-[12px] text-danger">{error}</span>}
    </>
  );
}

function ExportJob({ job }: { job: Job }) {
  const { events, status } = useJob(job.id);
  const phase = [...events].reverse().find((e) => e.event === "phase")?.name as string | undefined;
  const logs = events.filter((e) => e.event === "log").map((e) => e.line as string);
  const state = status ?? job.status;
  const out = (job.result.output_dir as string) ?? "";
  const result = events.find((e) => e.event === "result");
  const ggufs = ((result?.gguf ?? job.result.gguf) as string[] | undefined) ?? [];
  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="truncate font-medium">{job.title}</span>
        <Badge tone={state === "done" ? "ok" : state === "failed" ? "warn" : "neutral"}>{state === "running" ? (phase ?? "Starting") : state}</Badge>
      </div>
      {state === "running" && <ProgressBar value={job.progress} />}
      {state === "failed" && job.error && <Notice tone="error">{job.error}</Notice>}
      {state === "done" && (
        <div className="mt-1 flex flex-wrap items-center gap-2">
          {ggufs.map((path) => (
            <span key={path} className="inline-flex flex-wrap items-center gap-2">
              <ChatButton path={path} />
              <AddToOllama path={path} />
            </span>
          ))}
          {inTauri && (
            <Button size="sm" onClick={() => reveal(out)}>
              <FolderOpen className="size-3.5" /> Show in folder
            </Button>
          )}
        </div>
      )}
      {logs.length > 0 && (
        <details className="mt-2 text-[12.5px]">
          <summary className="cursor-pointer text-muted">Output</summary>
          <pre className="mt-2 max-h-48 overflow-auto font-mono text-[11.5px] whitespace-pre-wrap text-muted">{logs.slice(-200).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}

export function ExportPage() {
  const { jobs, refreshJobs, go } = useApp();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [formats, setFormats] = useState<string[]>(["gguf"]);
  const [quants, setQuants] = useState<string[]>(["q4_k_m"]);
  const [error, setError] = useState<string | null>(null);

  const finishedTrains = jobs.filter((j) => j.kind === "train" && j.status === "done").length;
  useEffect(() => {
    api<Run[]>("/api/train/runs")
      .then((r) => {
        setRuns(r);
        setRunId((cur) => cur ?? r.find((x) => x.adapter)?.id ?? null);
      })
      .catch(() => setRuns([]));
  }, [finishedTrains]);

  const run = runs?.find((r) => r.id === runId);
  const mlx = run?.backend === "mlx";
  const exportJobs = useMemo(() => jobs.filter((j) => j.kind === "export"), [jobs]);
  const toggle = (list: string[], set: (v: string[]) => void, id: string) => set(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);

  async function start() {
    setError(null);
    try {
      await api("/api/export", { json: { run_id: runId, formats, gguf_quants: quants } });
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (runs && !runs.some((r) => r.adapter)) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title="Export" description="Package a trained model for other tools." />
        <EmptyState icon={<PackageOpen className="size-8" />} title="Nothing to export yet">
          <p>Finish a training run first; its adapter shows up here.</p>
          <Button variant="primary" className="mt-4" onClick={() => go("train")}>
            Go to Train
          </Button>
        </EmptyState>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Export"
        description="Package a trained model for other tools."
        actions={
          <Button variant="primary" onClick={start} disabled={!runId || !formats.length || (formats.includes("gguf") && !quants.length)}>
            <PackageOpen className="size-4" /> Export
          </Button>
        }
      />
      <div className="flex min-h-0 flex-1">
        <div className="w-[420px] shrink-0 overflow-y-auto border-r border-line px-7 pb-8">
          <Section title="Run">
            <div className="flex flex-col gap-1.5">
              {runs?.filter((r) => r.adapter).map((r) => (
                <button
                  key={r.id}
                  onClick={() => setRunId(r.id)}
                  className={cx("rounded-lg border px-3 py-2 text-left", r.id === runId ? "border-accent bg-accent/6" : "border-line bg-surface hover:bg-surface-2")}
                >
                  <span className="block truncate font-medium">{r.id}</span>
                  <span className="block truncate text-[12.5px] text-muted">
                    {r.base_model} on {r.dataset}
                  </span>
                </button>
              ))}
            </div>
          </Section>
          <Section title="Formats">
            <div className="flex flex-col gap-1.5">
              {FORMATS.map((f) => {
                const disabled = mlx && f.id === "merged_4bit";
                return (
                  <label key={f.id} className={cx("flex cursor-pointer gap-3 rounded-lg border border-line bg-surface px-3 py-2", disabled && "opacity-40")}>
                    <input type="checkbox" disabled={disabled} checked={formats.includes(f.id)} onChange={() => toggle(formats, setFormats, f.id)} className="mt-1 accent-[var(--accent)]" />
                    <span>
                      <span className="font-medium">{f.label}</span>
                      <span className="block text-[12.5px] text-muted">{f.note}</span>
                    </span>
                  </label>
                );
              })}
            </div>
          </Section>
          {formats.includes("gguf") && (
            <Section title="GGUF quantization">
              <div className="grid grid-cols-2 gap-1.5">
                {QUANTS.map((q) => (
                  <label key={q.id} className="flex cursor-pointer items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-1.5 text-[13px]">
                    <input type="checkbox" checked={quants.includes(q.id)} onChange={() => toggle(quants, setQuants, q.id)} className="accent-[var(--accent)]" />
                    <span className="font-mono text-[12px]">{q.id.toUpperCase()}</span>
                    <span className="truncate text-muted">{q.note}</span>
                  </label>
                ))}
              </div>
              <p className="mt-2 text-[12.5px] text-muted">The first GGUF export also downloads llama.cpp's converter and PyTorch, which takes a few minutes.</p>
            </Section>
          )}
          {error && <Notice tone="error">{error}</Notice>}
        </div>
        <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
          {exportJobs.length ? (
            <div className="flex flex-col gap-3">
              {exportJobs.map((j) => (
                <ExportJob key={j.id} job={j} />
              ))}
            </div>
          ) : (
            <EmptyState icon={<PackageOpen className="size-8" />} title="Exports you start appear here" />
          )}
        </div>
      </div>
    </div>
  );
}
