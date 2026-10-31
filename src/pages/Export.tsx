import { FolderOpen, MessageSquare, PackageOpen } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { AddToOllama } from "../components/AddToOllama";
import { Badge, Button, EmptyState, Notice, PageHeader, ProgressBar, Section, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { useJob } from "../lib/useJob";
import type { Job, Run } from "../lib/types";

/** Formats without a label go by their id: GGUF, MLX. */
const FORMATS: { id: string; label?: Key; note: Key }[] = [
  { id: "gguf", note: "export.gguf_note" },
  { id: "lora", label: "export.lora", note: "export.lora_note" },
  { id: "merged_16bit", label: "export.merged16", note: "export.merged16_note" },
  { id: "merged_4bit", label: "export.merged4", note: "export.merged4_note" },
  { id: "mlx", note: "export.mlx_note" },
];
const QUANTS: { id: string; note: Key }[] = [
  { id: "q4_k_m", note: "export.q_recommended" },
  { id: "q5_k_m", note: "export.q_higher" },
  { id: "q8_0", note: "export.q_lossless" },
  { id: "f16", note: "export.q_unquantized" },
  { id: "q3_k_m", note: "export.q_smaller" },
  { id: "q2_k", note: "export.q_smallest" },
];
/** I-quants: better than K quants at the same size, but only with an importance matrix. */
const IQUANTS: { id: string; note: Key }[] = [
  { id: "iq4_xs", note: "export.iq4" },
  { id: "iq3_m", note: "export.iq3" },
  { id: "iq2_m", note: "export.iq2" },
];
const STATE: Record<string, Key> = { done: "job.done", failed: "job.failed", cancelled: "job.cancelled" };

async function reveal(path: string) {
  if (!inTauri) return;
  const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
  await revealItemInDir(path);
}

/** Loads an exported GGUF into the chat runtime and opens Chat. */
function ChatButton({ path }: { path: string }) {
  const { refreshRuntime, go } = useApp();
  const t = useT();
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
        <MessageSquare className="size-3.5" /> {quant ? t("export.chat_with", { quant }) : t("export.chat_with_it")}
      </Button>
      {error && <span className="text-[12px] text-danger">{error}</span>}
    </>
  );
}

function ExportJob({ job }: { job: Job }) {
  const { events, status } = useJob(job.id);
  const t = useT();
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
        <Badge tone={state === "done" ? "ok" : state === "failed" ? "warn" : "neutral"}>{state === "running" ? (phase ?? t("export.starting")) : STATE[state] ? t(STATE[state]) : state}</Badge>
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
              <FolderOpen className="size-3.5" /> {t("export.show_in_folder")}
            </Button>
          )}
        </div>
      )}
      {logs.length > 0 && (
        <details className="mt-2 text-[12.5px]">
          <summary className="cursor-pointer text-muted">{t("export.output")}</summary>
          <pre className="mt-2 max-h-48 overflow-auto font-mono text-[11.5px] whitespace-pre-wrap text-muted">{logs.slice(-200).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}

export function ExportPage() {
  const { jobs, refreshJobs, go } = useApp();
  const t = useT();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [formats, setFormats] = useState<string[]>(["gguf"]);
  const [quants, setQuants] = useState<string[]>(["q4_k_m"]);
  const [imatrix, setImatrix] = useState(false);
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
      await api("/api/export", { json: { run_id: runId, formats, gguf_quants: imatrix ? quants : quants.filter((q) => !q.startsWith("iq")), imatrix } });
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (runs && !runs.some((r) => r.adapter)) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title={t("nav.export")} description={t("export.description")} />
        <EmptyState icon={<PackageOpen className="size-8" />} title={t("export.nothing")}>
          <p>{t("export.nothing_hint")}</p>
          <Button variant="primary" className="mt-4" onClick={() => go("train")}>
            {t("export.go_train")}
          </Button>
        </EmptyState>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title={t("nav.export")}
        description={t("export.description")}
        actions={
          <Button variant="primary" onClick={start} disabled={!runId || !formats.length || (formats.includes("gguf") && !quants.length)}>
            <PackageOpen className="size-4" /> {t("export.export")}
          </Button>
        }
      />
      <div className="flex min-h-0 flex-1">
        <div className="w-[420px] shrink-0 overflow-y-auto border-r border-line px-7 pb-8">
          <Section title={t("export.run")}>
            <div className="flex flex-col gap-1.5">
              {runs?.filter((r) => r.adapter).map((r) => (
                <button
                  key={r.id}
                  onClick={() => setRunId(r.id)}
                  className={cx("rounded-lg border px-3 py-2 text-left", r.id === runId ? "border-accent bg-accent/6" : "border-line bg-surface hover:bg-surface-2")}
                >
                  <span className="block truncate font-medium">{r.id}</span>
                  <span className="block truncate text-[12.5px] text-muted">
                    {t("export.run_on", { model: r.base_model, dataset: r.dataset })}
                  </span>
                </button>
              ))}
            </div>
          </Section>
          <Section title={t("export.formats")}>
            <div className="flex flex-col gap-1.5">
              {FORMATS.map((f) => {
                const disabled = (mlx && f.id === "merged_4bit") || (!mlx && f.id === "mlx");
                return (
                  <label key={f.id} className={cx("flex cursor-pointer gap-3 rounded-lg border border-line bg-surface px-3 py-2", disabled && "opacity-40")}>
                    <input type="checkbox" disabled={disabled} checked={formats.includes(f.id)} onChange={() => toggle(formats, setFormats, f.id)} className="mt-1 accent-[var(--accent)]" />
                    <span>
                      <span className="font-medium">{f.label ? t(f.label) : f.id.toUpperCase()}</span>
                      <span className="block text-[12.5px] text-muted">{t(f.note)}</span>
                    </span>
                  </label>
                );
              })}
            </div>
          </Section>
          {formats.includes("gguf") && (
            <Section title={t("export.quantization")}>
              <div className="grid grid-cols-2 gap-1.5">
                {[...QUANTS, ...(imatrix ? IQUANTS : [])].map((q) => (
                  <label key={q.id} className="flex cursor-pointer items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-1.5 text-[13px]">
                    <input type="checkbox" checked={quants.includes(q.id)} onChange={() => toggle(quants, setQuants, q.id)} className="accent-[var(--accent)]" />
                    <span className="font-mono text-[12px]">{q.id.toUpperCase()}</span>
                    <span className="truncate text-muted">{t(q.note)}</span>
                  </label>
                ))}
              </div>
              <div className="mt-3">
                <Toggle label={t("export.imatrix")} checked={imatrix} onChange={setImatrix} />
                <p className="mt-1 text-[12.5px] text-muted">
                  {t("export.imatrix_hint")}
                </p>
              </div>
              <p className="mt-2 text-[12.5px] text-muted">{t("export.first_gguf")}</p>
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
            <EmptyState icon={<PackageOpen className="size-8" />} title={t("export.empty")} />
          )}
        </div>
      </div>
    </div>
  );
}
