import { ChevronDown, Download, Heart, Lock, Pause, Play, Search, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { RepoAccess } from "../components/HfAccount";
import { Badge, Button, EmptyState, Input, Notice, PageHeader, ProgressBar, Toggle, cx } from "../components/ui";
import { api } from "../lib/engine";
import { bytes, compact, duration, quantLabel, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { Budget, Verdict } from "../components/MemoryEstimate";
import type { HubModel, Job, LocalModel, RepoFile } from "../lib/types";
import { useJob } from "../lib/useJob";

/** Temper loads chat models with this context (runtime.LoadRequest), so estimates use it too. */
const CHAT_CONTEXT = 8192;

interface ChatEstimate {
  weights: number;
  kv_cache: number;
  overhead: number;
  total: number;
  context: number;
  budget: Budget;
  verdict: Verdict;
  advice: string[];
}

/** Whether a GGUF of `size` bytes fits: the file, plus the KV cache and compute buffers for the model's
 * shape (same for every quantization in a repo), against what the GPU can use. Without an estimate,
 * falls back to the file size plus 20%. */
function fit(size: number, memory: number, est: ChatEstimate | null): { label: string; tone: "ok" | "warn" | "neutral"; title?: string } {
  if (est) {
    const need = size + est.kv_cache + est.overhead;
    const title = `About ${bytes(need)} with a ${est.context.toLocaleString()}-token context; ${bytes(est.budget.free)} of ${bytes(est.budget.total)} free now.`;
    if (need <= est.budget.free * 0.9) return { label: "Fits", tone: "ok", title };
    if (need <= est.budget.total) return { label: "Tight fit", tone: "warn", title };
    return { label: "Too large", tone: "neutral", title };
  }
  if (!memory) return { label: "", tone: "neutral" };
  const need = size * 1.2;
  if (need < memory * 0.7) return { label: "Fits", tone: "ok" };
  if (need < memory * 0.95) return { label: "Tight fit", tone: "warn" };
  return { label: "Too large", tone: "neutral" };
}

function useMemoryBudget() {
  const system = useApp((s) => s.system);
  const gpu = system?.gpus[0];
  // Discrete GPUs can spill to system RAM, but slowly; use VRAM as the budget.
  return gpu ? gpu.memory_total : (system?.memory.total ?? 0);
}

/** Downloads Temper's pinned llama.cpp build (checksummed release binaries) into the app's data folder. */
function InstallRuntime() {
  const { refreshRuntime, refreshSystem } = useApp();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status } = useJob(jobId);
  // Until the stream's first event arrives, a job that was just started is running.
  const running = jobId !== null && (status === null || status === "running");

  useEffect(() => {
    if (status === "done") void Promise.all([refreshRuntime(), refreshSystem()]);
    if (status === "failed") setError("The download failed. Check your connection and try again.");
  }, [status, refreshRuntime, refreshSystem]);

  return (
    <Notice>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span>{error ?? "Chat runs models with llama.cpp, which isn't installed yet. It's a one-time download."}</span>
        <Button
          size="sm"
          disabled={running}
          onClick={async () => {
            setError(null);
            try {
              setJobId((await api<Job>("/api/llamacpp/install", { json: {} })).id);
            } catch (e) {
              setError(String(e instanceof Error ? e.message : e));
            }
          }}
        >
          <Download className="size-3.5" />
          {running ? "Installing llama.cpp…" : "Install llama.cpp"}
        </Button>
      </div>
    </Notice>
  );
}

function LoadButton({ path, name }: { path: string; name: string }) {
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
          const est = await api<ChatEstimate>("/api/memory/chat", { json: { path, context: CHAT_CONTEXT } }).catch(() => null);
          if (est?.verdict === "too_large") {
            const tips = est.advice.length ? `\n\n${est.advice.join("\n")}` : "";
            if (!confirm(`${name} needs about ${bytes(est.total)}, and this computer can give it ${bytes(est.budget.total)}. Loading may fail or make your computer very slow.${tips}\n\nLoad anyway?`)) return;
          }
          try {
            await api("/api/runtime/load", { json: { path } });
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

function RepoDetail({ id }: { id: string }) {
  return (
    <div className="space-y-3">
      <RepoAccess repoId={id} />
      <RepoFiles id={id} />
    </div>
  );
}

function RepoFiles({ id }: { id: string }) {
  const memory = useMemoryBudget();
  const refreshJobs = useApp((s) => s.refreshJobs);
  const [files, setFiles] = useState<RepoFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<Set<string>>(new Set());
  const [estimate, setEstimate] = useState<ChatEstimate | null>(null);

  useEffect(() => {
    api<{ gguf: RepoFile[] }>(`/api/models/repo?id=${encodeURIComponent(id)}`)
      .then((r) => {
        setFiles(r.gguf);
        // Every quantization shares the model's shape, so one header gives the KV cache for all of them.
        if (r.gguf.length)
          void api<ChatEstimate>("/api/memory/chat", { json: { repo_id: id, filename: r.gguf[0].name, context: CHAT_CONTEXT } })
            .then(setEstimate)
            .catch(() => setEstimate(null));
      })
      .catch((e) => setError(e.message));
  }, [id]);

  async function download(patterns: string[], key: string) {
    await api("/api/models/download", { json: { repo_id: id, patterns } });
    setStarted((s) => new Set(s).add(key));
    void refreshJobs();
  }

  if (error) return <Notice tone="error">{error}</Notice>;
  if (!files) return <p className="py-2 text-[13px] text-muted">Reading repository…</p>;
  if (!files.length) {
    return (
      <div className="flex items-center justify-between gap-3 py-1 text-[13px]">
        <span className="text-muted">No GGUF files. Download the full repository to fine-tune it, or pick its -GGUF sibling to chat.</span>
        <Button size="sm" onClick={() => download([], "all")} disabled={started.has("all")}>
          <Download className="size-3.5" /> {started.has("all") ? "Downloading" : "Download all"}
        </Button>
      </div>
    );
  }
  return (
    <table className="w-full text-[13px]">
      <tbody>
        {files.map((f) => {
          const f2 = fit(f.size, memory, estimate);
          return (
            <tr key={f.name} className="border-t border-line/70">
              <td className="py-1.5 pr-3 font-mono text-[12px]">{quantLabel(f.name)}</td>
              <td className="py-1.5 pr-3 text-muted tabular-nums">
                {bytes(f.size)}
                {f.parts.length > 1 && <span className="text-muted/70"> in {f.parts.length} parts</span>}
              </td>
              <td className="py-1.5 pr-3" title={f2.title}>
                {f2.label && <Badge tone={f2.tone}>{f2.label}</Badge>}
              </td>
              <td className="py-1.5 text-right">
                <Button size="sm" onClick={() => download(f.parts, f.name)} disabled={started.has(f.name)}>
                  <Download className="size-3.5" /> {started.has(f.name) ? "Downloading" : "Download"}
                </Button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function FindModels() {
  const [query, setQuery] = useState("");
  const [author, setAuthor] = useState("");
  const [ggufOnly, setGgufOnly] = useState(true);
  const [results, setResults] = useState<HubModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const search = useCallback(async () => {
    setError(null);
    try {
      const params = new URLSearchParams({ q: query, author, gguf: String(ggufOnly), limit: "40" });
      setResults(await api<HubModel[]>(`/api/models/search?${params}`));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [query, author, ggufOnly]);

  useEffect(() => {
    const t = setTimeout(search, 300);
    return () => clearTimeout(t);
  }, [search]);

  return (
    <div className="px-7 py-5">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative min-w-[260px] flex-1">
          <Search className="pointer-events-none absolute top-2.5 left-3 size-4 text-muted" />
          <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search models, e.g. qwen, gemma, llama" className="pl-9" autoFocus />
        </div>
        <Input value={author} onChange={(e) => setAuthor(e.target.value)} placeholder="Any author" className="w-40" aria-label="Author" />
        <div className="w-36">
          <Toggle checked={ggufOnly} onChange={setGgufOnly} label="GGUF only" />
        </div>
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      {results && results.length === 0 && <EmptyState icon={<Search className="size-7" />} title="No models match">Try a shorter search or clear the author filter.</EmptyState>}
      <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
        {results?.map((m) => (
          <li key={m.id}>
            <button onClick={() => setOpen(open === m.id ? null : m.id)} className="flex w-full items-center gap-4 px-4 py-2.5 text-left hover:bg-surface-2" aria-expanded={open === m.id}>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 truncate font-medium">
                  <span className="text-muted">{m.id.split("/")[0]}/</span>
                  {m.id.split("/")[1]}
                  {m.gated && <Lock className="size-3.5 text-straw" aria-label="Gated" />}
                </div>
                <div className="mt-0.5 flex gap-3 text-[12px] text-muted">
                  {m.pipeline && <span>{m.pipeline}</span>}
                  <span>Updated {relativeDate(m.updated)}</span>
                </div>
              </div>
              <span className="flex items-center gap-1 text-[12.5px] text-muted tabular-nums">
                <Download className="size-3.5" /> {compact(m.downloads)}
              </span>
              <span className="flex w-14 items-center gap-1 text-[12.5px] text-muted tabular-nums">
                <Heart className="size-3.5" /> {compact(m.likes)}
              </span>
              <ChevronDown className={cx("size-4 text-muted transition-transform", open === m.id && "rotate-180")} />
            </button>
            {open === m.id && (
              <div className="bg-surface-2/60 px-4 pb-3">
                <RepoDetail id={m.id} />
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function LocalModels() {
  const [models, setModels] = useState<LocalModel[] | null>(null);
  const jobs = useApp((s) => s.jobs);
  const finished = jobs.filter((j) => j.kind === "download" && j.status === "done").length;

  useEffect(() => {
    api<LocalModel[]>("/api/models/local").then(setModels).catch(() => setModels([]));
  }, [finished]);

  if (!models) return <p className="px-7 py-5 text-muted">Scanning the model cache…</p>;
  if (!models.length) return <EmptyState icon={<Download className="size-7" />} title="No models on this computer yet">Find one under Find models and download a GGUF file.</EmptyState>;

  return (
    <ul className="mx-7 my-5 divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
      {models.map((m) => (
        <li key={m.id} className="px-4 py-3">
          <div className="flex items-baseline justify-between gap-4">
            <span className="truncate font-medium">{m.id}</span>
            <span className="shrink-0 text-[12.5px] text-muted tabular-nums">
              {bytes(m.size)} · used {relativeDate(m.last_used)}
            </span>
          </div>
          {m.gguf.length > 0 ? (
            <div className="mt-2 flex flex-col gap-1.5">
              {m.gguf.map((f) => (
                <div key={f.name} className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="font-mono text-[12px]">
                    {quantLabel(f.name)} <span className="font-sans text-muted">{bytes(f.size)}</span>
                  </span>
                  {f.complete ? (
                    <LoadButton path={f.path!} name={f.name.split("/").pop()!} />
                  ) : (
                    <span className="text-[12px] text-muted">{f.parts.length} parts downloaded; download the rest to load it</span>
                  )}
                </div>
              ))}
            </div>
          ) : m.safetensors ? (
            <p className="mt-1 text-[12.5px] text-muted">Safetensors weights. Use this as a base model on the Train page.</p>
          ) : (
            <p className="mt-1 text-[12.5px] text-muted">No model weights downloaded, only metadata files.</p>
          )}
        </li>
      ))}
    </ul>
  );
}

interface DownloadStats {
  received?: number;
  total?: number;
  speed?: number;
  request?: { repo_id: string; patterns: string[] };
}

/** Active, paused and failed downloads. Pausing keeps the partial file, so Resume continues from it. */
function Downloads() {
  const { jobs, refreshJobs } = useApp();
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const downloads = jobs.filter((j) => j.kind === "download");
  const running = downloads.some((j) => j.status === "running");
  // A paused or failed download stays until resumed (a newer job with the same title) or dismissed.
  const shown = downloads.filter((j) => {
    if (dismissed.has(j.id)) return false;
    if (j.status === "running") return true;
    if (j.status === "done") return Date.now() / 1000 - j.created < 600;
    return !downloads.some((o) => o.title === j.title && o.created > j.created);
  });

  // Byte counts move fast; refresh every second while something is downloading.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => void refreshJobs(), 1000);
    return () => clearInterval(id);
  }, [running, refreshJobs]);

  if (!shown.length) return null;

  async function pause(id: string) {
    await api(`/api/jobs/${id}/cancel`, { method: "POST" });
    void refreshJobs();
  }
  async function resume(stats: DownloadStats) {
    if (!stats.request) return;
    await api("/api/models/download", { json: stats.request });
    void refreshJobs();
  }

  return (
    <div className="mx-7 mt-5 rounded-lg border border-line bg-surface px-4 py-3">
      <h2 className="mb-2 text-[13px] font-semibold">Downloads</h2>
      <ul className="flex flex-col gap-3">
        {shown.map((j) => {
          const st = j.result as DownloadStats;
          const left = st.speed && st.total && st.received != null ? (st.total - st.received) / st.speed : null;
          return (
            <li key={j.id} className="text-[13px]">
              <div className="mb-1 flex items-center justify-between gap-3">
                <span className="truncate">{j.title}</span>
                <span className="flex shrink-0 items-center gap-2">
                  {j.status === "running" && (
                    <Button size="sm" onClick={() => void pause(j.id)}>
                      <Pause className="size-3.5" /> Pause
                    </Button>
                  )}
                  {(j.status === "cancelled" || j.status === "failed") && st.request && (
                    <Button size="sm" onClick={() => void resume(st)}>
                      <Play className="size-3.5" /> Resume
                    </Button>
                  )}
                  {j.status !== "running" && (
                    <button aria-label="Dismiss" className="text-muted hover:text-text" onClick={() => setDismissed((d) => new Set(d).add(j.id))}>
                      <X className="size-3.5" />
                    </button>
                  )}
                </span>
              </div>
              {j.status !== "done" && <ProgressBar value={j.progress} />}
              <div className="mt-1 flex justify-between gap-3 text-[12px] text-muted tabular-nums">
                <span>
                  {st.total ? `${bytes(st.received ?? 0)} of ${bytes(st.total)}` : ""}
                  {j.status === "running" && st.speed ? ` · ${bytes(st.speed)}/s` : ""}
                </span>
                <span className={cx(j.status === "failed" && "text-danger")} title={j.error ?? undefined}>
                  {j.status === "running" ? (left != null ? `${duration(left)} left` : "Starting…") : j.status === "done" ? "Done" : j.status === "failed" ? (j.error ?? "Failed") : "Paused"}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function ModelsPage() {
  const [tab, setTab] = useState<"find" | "local">("local");
  const runtime = useApp((s) => s.runtime);
  const refreshRuntime = useApp((s) => s.refreshRuntime);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Models"
        description="Download open-weight models from Hugging Face and load them for chat."
        actions={
          runtime?.state === "ready" && (
            <Button size="sm" variant="ghost" onClick={async () => { await api("/api/runtime/unload", { method: "POST" }); void refreshRuntime(); }}>
              Unload {runtime.model}
            </Button>
          )
        }
      />
      <div className="flex gap-1 border-b border-line px-7" role="tablist">
        {(
          [
            ["local", "On this computer"],
            ["find", "Find models"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            className={cx("-mb-px border-b-2 px-3 py-2.5 font-medium", tab === key ? "border-accent text-text" : "border-transparent text-muted hover:text-text")}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {runtime && !runtime.available && (
          <div className="mx-7 mt-5">
            <InstallRuntime />
          </div>
        )}
        {runtime?.state === "failed" && runtime.error && (
          <div className="mx-7 mt-5">
            <Notice tone="error">
              <pre className="font-mono text-[12px] whitespace-pre-wrap">{runtime.error}</pre>
            </Notice>
          </div>
        )}
        <Downloads />
        {tab === "find" ? <FindModels /> : <LocalModels />}
      </div>
    </div>
  );
}
