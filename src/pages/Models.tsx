import { ChevronDown, Download, FileUp, FlaskConical, FolderUp, Heart, Lock, Pause, Play, Search, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { CatalogView } from "../components/Catalog";
import { useDevices } from "../components/Devices";
import { listed, useProviders } from "../components/Providers";
import { CHAT_CONTEXT, type ChatEstimate, LoadButton, folderEngine, useEngines } from "../components/LoadButton";
import { RepoAccess } from "../components/HfAccount";
import { sourceName, useHubSource } from "../components/HubSource";
import { Badge, Button, CopyButton, EmptyState, Input, Notice, PageHeader, ProgressBar, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, compact, duration, quantLabel, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { HubModel, Job, LocalModel, RepoFile } from "../lib/types";
import { useOpened } from "../lib/desktop";
import { useFileDrop } from "../lib/useFileDrop";
import { t as tr, useT } from "../lib/i18n";
import { useJob } from "../lib/useJob";

/** Whether a GGUF of `size` bytes fits: the file, plus the KV cache and compute buffers for the model's
 * shape (same for every quantization in a repo), against what the GPU can use. Without an estimate,
 * falls back to the file size plus 20%. */
function fit(size: number, memory: number, est: ChatEstimate | null): { label: string; tone: "ok" | "warn" | "neutral"; title?: string } {
  if (est) {
    const need = size + est.kv_cache + est.overhead;
    const title = tr("models.fit_title", { need: bytes(need), context: est.context.toLocaleString(), free: bytes(est.budget.free), total: bytes(est.budget.total) });
    if (need <= est.budget.free * 0.9) return { label: tr("models.fits"), tone: "ok", title };
    if (need <= est.budget.total) return { label: tr("models.tight"), tone: "warn", title };
    return { label: tr("models.too_large"), tone: "neutral", title };
  }
  if (!memory) return { label: "", tone: "neutral" };
  const need = size * 1.2;
  if (need < memory * 0.7) return { label: tr("models.fits"), tone: "ok" };
  if (need < memory * 0.95) return { label: tr("models.tight"), tone: "warn" };
  return { label: tr("models.too_large"), tone: "neutral" };
}

function useMemoryBudget() {
  const system = useApp((s) => s.system);
  const gpu = system?.gpus[0];
  // Discrete GPUs can spill to system RAM, but slowly; use VRAM as the budget.
  return gpu ? gpu.memory_total : (system?.memory.total ?? 0);
}

/** Downloads Temper's pinned llama.cpp build (checksummed release binaries) into the app's data folder. */
function InstallRuntime() {
  const t = useT();
  const { refreshRuntime, refreshSystem } = useApp();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status } = useJob(jobId);
  // Until the stream's first event arrives, a job that was just started is running.
  const running = jobId !== null && (status === null || status === "running");

  useEffect(() => {
    if (status === "done") void Promise.all([refreshRuntime(), refreshSystem()]);
    if (status === "failed") setError(t("models.runtime_failed"));
  }, [status, refreshRuntime, refreshSystem, t]);

  return (
    <Notice>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span>{error ?? t("models.runtime_missing")}</span>
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
          {running ? t("models.runtime_installing") : t("models.runtime_install")}
        </Button>
      </div>
    </Notice>
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
  const t = useT();
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
  if (!files) return <p className="py-2 text-[13px] text-muted">{t("models.reading_repo")}</p>;
  if (!files.length) {
    return (
      <div className="flex items-center justify-between gap-3 py-1 text-[13px]">
        <span className="text-muted">{t("models.no_gguf")}</span>
        <Button size="sm" onClick={() => download([], "all")} disabled={started.has("all")}>
          <Download className="size-3.5" /> {started.has("all") ? t("models.downloading") : t("models.download_all")}
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
                {f.parts.length > 1 && <span className="text-muted"> {t("models.in_parts", { n: f.parts.length })}</span>}
              </td>
              <td className="py-1.5 pr-3" title={f2.title}>
                {f2.label && <Badge tone={f2.tone}>{f2.label}</Badge>}
              </td>
              <td className="py-1.5 text-right">
                <Button size="sm" onClick={() => download(f.parts, f.name)} disabled={started.has(f.name)}>
                  <Download className="size-3.5" /> {started.has(f.name) ? t("models.downloading") : t("models.download")}
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
  const t = useT();
  const [query, setQuery] = useState("");
  const [author, setAuthor] = useState("");
  const [ggufOnly, setGgufOnly] = useState(true);
  const [results, setResults] = useState<HubModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const linked = useOpened((o) => o.search);

  // A temper://model/<owner>/<name> link searches for that repository.
  useEffect(() => {
    if (!linked) return;
    setQuery(linked);
    setAuthor("");
    useOpened.getState().set({ search: null });
  }, [linked]);

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
    const timer = setTimeout(search, 300);
    return () => clearTimeout(timer);
  }, [search]);

  return (
    <div className="px-7 py-5">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative min-w-[260px] flex-1">
          <Search className="pointer-events-none absolute top-2.5 left-3 size-4 text-muted" />
          <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("models.search_placeholder")} className="pl-9" autoFocus />
        </div>
        <Input value={author} onChange={(e) => setAuthor(e.target.value)} placeholder={t("models.any_author")} className="w-40" aria-label={t("models.author")} />
        <div className="w-36">
          <Toggle checked={ggufOnly} onChange={setGgufOnly} label={t("models.gguf_only")} />
        </div>
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      {results && results.length === 0 && <EmptyState icon={<Search className="size-7" />} title={t("models.no_match")}>{t("models.no_match_hint")}</EmptyState>}
      <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
        {results?.map((m) => (
          <li key={m.id}>
            <button onClick={() => setOpen(open === m.id ? null : m.id)} className="flex w-full items-center gap-4 px-4 py-2.5 text-left hover:bg-surface-2" aria-expanded={open === m.id}>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 truncate font-medium">
                  <span className="text-muted">{m.id.split("/")[0]}/</span>
                  {m.id.split("/")[1]}
                  {m.gated && <Lock className="size-3.5 text-straw" aria-label={t("models.gated")} />}
                </div>
                <div className="mt-0.5 flex gap-3 text-[12px] text-muted">
                  {m.pipeline && <span>{m.pipeline}</span>}
                  <span>{t("models.updated", { when: relativeDate(m.updated) })}</span>
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

/** Adds files to the library (POST /api/models/import), one at a time so each error names its file. */
async function importPaths(paths: string[]): Promise<string[]> {
  const errors: string[] = [];
  for (const path of paths) {
    await api("/api/models/import", { json: { path } }).catch((e: Error) => errors.push(e.message));
  }
  return errors;
}

function ImportBar({ onImported }: { onImported: (errors: string[]) => void }) {
  const t = useT();
  const [path, setPath] = useState("");

  async function pick(directory: boolean) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const chosen = await open(directory ? { directory: true } : { multiple: true, filters: [{ name: t("models.gguf_files"), extensions: ["gguf"] }] });
    const paths = chosen === null ? [] : Array.isArray(chosen) ? chosen : [chosen];
    if (paths.length) onImported(await importPaths(paths));
  }

  if (inTauri) {
    return (
      <div className="mx-7 mt-5 flex flex-wrap items-center gap-2 text-[13px]">
        <Button size="sm" onClick={() => void pick(false)}>
          <FileUp className="size-3.5" /> {t("models.import_file")}
        </Button>
        <Button size="sm" onClick={() => void pick(true)}>
          <FolderUp className="size-3.5" /> {t("models.import_folder")}
        </Button>
        <span className="text-muted">{t("models.import_hint")}</span>
      </div>
    );
  }
  return (
    <form
      className="mx-7 mt-5 flex gap-2"
      onSubmit={async (e) => {
        e.preventDefault();
        const errors = await importPaths([path.trim()]);
        if (!errors.length) setPath("");
        onImported(errors);
      }}
    >
      <Input value={path} onChange={(e) => setPath(e.target.value)} placeholder={t("models.path_placeholder")} spellCheck={false} className="flex-1" />
      <Button type="submit" disabled={!path.trim()}>
        {t("models.import")}
      </Button>
    </form>
  );
}

interface OllamaModel {
  name: string;
  size: number;
  imported: boolean;
  vision: boolean;
}

/** Models downloaded with Ollama that aren't in the library yet, offered without copying them. */
function OllamaModels({ version, onImported }: { version: number; onImported: (errors: string[]) => void }) {
  const t = useT();
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api<{ models: OllamaModel[] }>("/api/ollama").then((r) => setModels(r.models.filter((m) => !m.imported)), () => setModels([]));
  }, [version]);
  if (!models.length) return null;

  async function add(names: string[]) {
    setBusy(true);
    const errors = await api("/api/ollama/import", { json: { names } }).then(
      () => [],
      (e: Error) => [e.message],
    );
    setBusy(false);
    onImported(errors);
  }

  return (
    <div className="mx-7 mt-4 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
      <div className="flex items-center justify-between gap-3">
        <p>
          <span className="font-medium">{t("models.ollama_found")}</span> <span className="text-muted">{t("models.ollama_hint")}</span>
        </p>
        {models.length > 1 && (
          <Button size="sm" disabled={busy} onClick={() => void add(models.map((m) => m.name))}>
            {t("models.add_all")}
          </Button>
        )}
      </div>
      <ul className="mt-2 flex flex-col gap-1.5">
        {models.map((m) => (
          <li key={m.name} className="flex items-center justify-between gap-3">
            <span>
              <span className="font-mono text-[12.5px]">{m.name}</span> <span className="text-muted">{bytes(m.size)}</span>
            </span>
            <Button size="sm" disabled={busy} onClick={() => void add([m.name])}>
              {t("models.add_to_library")}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The models of linked computers (Connect → Linked devices), which the API answers from there. */
function DeviceModels() {
  const { devices } = useDevices();
  const go = useApp((s) => s.go);
  const t = useT();
  if (!devices.length) return null;
  return (
    <div className="mx-7 mt-6 mb-6">
      <h2 className="text-[13px] font-semibold">{t("models.other_devices")}</h2>
      <p className="mt-0.5 mb-2 text-[12px] text-muted">
        {t("models.other_devices_hint")}{" "}
        <button className="underline hover:text-text" onClick={() => go("connect")}>
          {t("models.linked_devices")}
        </button>
      </p>
      <ul className="divide-y divide-line rounded-lg border border-line bg-surface text-[13px]">
        {devices.flatMap((d) =>
          d.online
            ? d.models.map((m) => (
                <li key={`${d.id}/${m.id}`} className="flex items-center gap-2 px-4 py-2">
                  <span className="font-medium">{m.id}</span>
                  <span className="text-muted">{t("models.on_device", { device: d.name })}</span>
                  {m.loaded ? <Badge tone="ok">{t("models.loaded")}</Badge> : <Badge>{t("models.loads_when_asked")}</Badge>}
                  <span className="flex-1" />
                  <span className="font-mono text-[11.5px] text-muted">{`${m.id}@${d.name}`}</span>
                  <CopyButton text={`${m.id}@${d.name}`} />
                </li>
              ))
            : [
                <li key={d.id} className="px-4 py-2 text-muted">
                  {d.name}: {d.error}
                </li>,
              ],
        )}
      </ul>
    </div>
  );
}

/** Cloud providers' listed models (Connect → Cloud providers), ready to use in chat. */
function CloudModels() {
  const { providers } = useProviders();
  const go = useApp((s) => s.go);
  const t = useT();
  if (!providers.length) return null;
  return (
    <div className="mx-7 mt-6 mb-6">
      <h2 className="text-[13px] font-semibold">{t("models.cloud")}</h2>
      <p className="mt-0.5 mb-2 text-[12px] text-muted">
        {t("models.cloud_hint")}{" "}
        <button className="underline hover:text-text" onClick={() => go("connect")}>
          {t("connect.providers")}
        </button>
      </p>
      <ul className="divide-y divide-line rounded-lg border border-line bg-surface text-[13px]">
        {providers.flatMap((p) =>
          p.online || p.models.length
            ? listed(p)
                .slice(0, 12)
                .map((m) => (
                  <li key={`${p.id}/${m}`} className="flex items-center gap-2 px-4 py-2">
                    <span className="font-medium">{m}</span>
                    <span className="text-muted">{t("models.from_provider", { provider: p.name })}</span>
                    <span className="flex-1" />
                    <span className="font-mono text-[11.5px] text-muted">{`${m}@${p.name}`}</span>
                    <CopyButton text={`${m}@${p.name}`} />
                    <LoadButton path={`${m}@${p.name}`} name={`${m}@${p.name}`} label={t("providers.use")} />
                  </li>
                ))
            : [
                <li key={p.id} className="px-4 py-2 text-muted">
                  {p.name}: {p.error}
                </li>,
              ],
        )}
      </ul>
    </div>
  );
}

const KIND_LABEL = { safetensors: "models.kind_safetensors", mlx: "models.kind_mlx", gguf: "models.kind_gguf", mistral: "models.kind_mistral" } as const;
const CONVERT_QUANTS = [
  ["q4_k_m", "Q4_K_M"],
  ["q8_0", "Q8_0"],
  ["f16", "F16"],
] as const;

/** Chat with a model folder on MLX (Apple Silicon) or an installed Transformers, vLLM or SGLang, without converting it. */
function FolderLoad({ model }: { model: LocalModel }) {
  const { engines } = useEngines();
  const t = useT();
  const engine = folderEngine(engines, model.kind === "mlx" ? "mlx" : "hf");
  if (!engine) return null;
  return <LoadButton path={model.imported ? model.path! : model.id} name={model.id.split("/").pop()!} engine={engine.id} label={t("models.load_with", { engine: engine.name })} />;
}

/** What a weights folder is for, in one line: training, export, or converting to chat with it. */
function weightsSummary(m: LocalModel, t: ReturnType<typeof useT>) {
  const label = t(KIND_LABEL[m.kind ?? "safetensors"]);
  const kind = m.quantization ? t("models.prequantized", { kind: label, quant: m.quantization.label }) : label;
  const uses = t(m.kind === "mlx" ? "models.uses_mlx" : m.kind === "mistral" || m.quantization ? "models.uses_convert" : "models.uses_both");
  return t("models.summary", { kind, uses });
}

/** Converts a safetensors, pre-quantized or Mistral-format checkpoint to a GGUF, which is what chat runs. */
function ConvertButton({ model, onDone }: { model: LocalModel; onDone: () => void }) {
  const { jobs, refreshJobs } = useApp();
  const t = useT();
  const [quant, setQuant] = useState("q4_k_m");
  const [started, setStarted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // A conversion started earlier (or before the page was reopened) is picked up from the job list.
  const existing = jobs.find((j) => j.kind === "convert" && j.status === "running" && j.result.source === model.path);
  const jobId = started ?? existing?.id ?? null;
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const phase = [...events].reverse().find((e) => e.event === "phase")?.name as string | undefined;
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const failure = events.find((e) => e.event === "status" && e.status === "failed")?.error as string | undefined;

  useEffect(() => {
    if (status === "done") onDone();
    if (status && status !== "running") void refreshJobs();
  }, [status, onDone, refreshJobs]);

  async function start() {
    setError(null);
    try {
      const job = await api<Job>("/api/models/convert", { json: { path: model.path, quant } });
      setStarted(job.id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (running)
    return (
      <span className="flex items-center gap-2 text-[12px] text-muted">
        {phase ?? t("models.starting")}…
        <span className="w-28">
          <ProgressBar value={progress ?? null} />
        </span>
        <Button size="sm" variant="ghost" onClick={() => void api(`/api/jobs/${jobId}/cancel`, { method: "POST" })}>
          {t("models.cancel")}
        </Button>
      </span>
    );
  const message = error ?? (status === "failed" ? t("models.convert_failed", { error: failure ?? t("models.unknown_error") }) : null);
  return (
    <span className="flex min-w-0 items-center gap-2">
      {message && (
        <span className="truncate text-[12px] text-danger" title={message}>
          {message}
        </span>
      )}
      <Select value={quant} onChange={(e) => setQuant(e.target.value)} className="h-7 w-auto text-[12.5px]" aria-label={t("models.quantization")}>
        {CONVERT_QUANTS.map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </Select>
      <Button size="sm" onClick={() => void start()} title={t("models.convert_title")}>
        {t("models.convert")}
      </Button>
    </span>
  );
}

function LocalModels({ version, onImported }: { version: number; onImported: (errors: string[]) => void }) {
  const [models, setModels] = useState<LocalModel[] | null>(null);
  const { jobs, go, setPendingBaseModel } = useApp();
  const t = useT();
  const finished = jobs.filter((j) => j.kind === "download" && j.status === "done").length;

  const load = useCallback(() => {
    api<LocalModel[]>("/api/models/local").then(setModels).catch(() => setModels([]));
  }, []);
  useEffect(load, [finished, version, load]);

  function fineTune(model: string) {
    setPendingBaseModel(model);
    go("train");
  }

  async function deleteModel(m: LocalModel) {
    if (!confirm(t("models.delete_confirm", { model: m.id, size: bytes(m.size) }))) return;
    await api(`/api/models/local?repo_id=${encodeURIComponent(m.id)}`, { method: "DELETE" }).then(load, (e: Error) => alert(e.message));
  }

  async function deleteFile(m: LocalModel, f: RepoFile) {
    if (!confirm(t("models.delete_file_confirm", { quant: quantLabel(f.name), model: m.id, size: bytes(f.size) }))) return;
    await api("/api/models/local/delete-files", { json: { repo_id: m.id, files: f.parts } }).then(load, (e: Error) => alert(e.message));
  }

  async function forget(path: string) {
    await api(`/api/models/import?path=${encodeURIComponent(path)}`, { method: "DELETE" });
    load();
  }

  if (!models) return <p className="px-7 py-5 text-muted">{t("models.scanning")}</p>;
  return (
    <>
      <ImportBar onImported={onImported} />
      <OllamaModels version={version} onImported={onImported} />
      {!models.length ? (
        <EmptyState icon={<Download className="size-7" />} title={t("models.empty")}>
          {t("models.empty_hint")}
        </EmptyState>
      ) : (
        <ul className="mx-7 my-5 divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
          {models.map((m) => (
            <li key={m.path ?? m.id} className="px-4 py-3">
              <div className="flex items-baseline justify-between gap-4">
                <span className="flex min-w-0 items-center gap-2">
                  <span className="truncate font-medium">{m.id}</span>
                  {m.imported && <Badge>{t("models.imported")}</Badge>}
                </span>
                <span className="flex shrink-0 items-center gap-3 text-[12.5px] text-muted tabular-nums">
                  {m.missing ? null : t(m.imported ? "models.added_when" : "models.used_when", { size: bytes(m.size), when: relativeDate(m.last_used) })}
                  {m.imported && m.path && (
                    <button className="hover:text-text" onClick={() => void forget(m.path!)} title={t("models.remove_title")}>
                      <X className="size-3.5" aria-label={t("models.remove")} />
                    </button>
                  )}
                  {!m.imported && (
                    <button className="hover:text-danger" onClick={() => void deleteModel(m)} title={t("models.delete_title")}>
                      <Trash2 className="size-3.5" aria-label={t("models.delete_named", { name: m.id })} />
                    </button>
                  )}
                </span>
              </div>
              {m.imported && <p className="mt-0.5 truncate font-mono text-[11.5px] text-muted" title={m.path ?? ""}>{m.path}</p>}
              {m.missing ? (
                <p className="mt-1 text-[12.5px] text-danger">{m.missing} {t("models.missing_hint")}</p>
              ) : m.gguf.length > 0 ? (
                <div className="mt-2 flex flex-col gap-1.5">
                  {m.gguf.map((f) => (
                    <div key={f.name} className="flex items-center justify-between gap-3 text-[13px]">
                      <span className="font-mono text-[12px]">
                        {quantLabel(f.name)} <span className="font-sans text-muted">{bytes(f.size)}</span>
                      </span>
                      <span className="flex items-center gap-2">
                      {f.complete ? (
                        <LoadButton path={f.path!} name={f.name.split("/").pop()!} />
                      ) : (
                        <span className="text-[12px] text-muted">{t(m.imported ? "models.parts_found" : "models.parts_downloaded", { n: f.parts.length })}</span>
                      )}
                      {!m.imported && m.gguf.length > 1 && (
                        <button className="text-muted hover:text-danger" onClick={() => void deleteFile(m, f)} title={t("models.delete_named", { name: quantLabel(f.name) })}>
                          <Trash2 className="size-3.5" aria-label={t("models.delete_named", { name: quantLabel(f.name) })} />
                        </button>
                      )}
                      </span>
                    </div>
                  ))}
                </div>
              ) : m.safetensors || m.kind === "mistral" ? (
                <div className="mt-1 flex items-center justify-between gap-3">
                  <p className="min-w-0 text-[12.5px] text-muted">
                    {weightsSummary(m, t)} {m.warnings?.join(" ")}
                  </p>
                  <span className="flex shrink-0 items-center gap-2">
                    {m.convertible && m.path && <ConvertButton model={m} onDone={load} />}
                    {m.chat && (m.kind === "mlx" || !m.quantization) && m.path && <FolderLoad model={m} />}
                    {m.safetensors && !m.quantization && (
                      <Button size="sm" onClick={() => fineTune(m.imported ? m.path! : m.id)}>
                        <FlaskConical className="size-3.5" /> {t("models.fine_tune")}
                      </Button>
                    )}
                  </span>
                </div>
              ) : (
                <p className="mt-1 text-[12.5px] text-muted">{t("models.no_weights")}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </>
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
  const t = useT();
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
      <h2 className="mb-2 text-[13px] font-semibold">{t("models.downloads")}</h2>
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
                      <Pause className="size-3.5" /> {t("models.pause")}
                    </Button>
                  )}
                  {(j.status === "cancelled" || j.status === "failed") && st.request && (
                    <Button size="sm" onClick={() => void resume(st)}>
                      <Play className="size-3.5" /> {t("models.resume")}
                    </Button>
                  )}
                  {j.status !== "running" && (
                    <button aria-label={t("models.dismiss")} className="text-muted hover:text-text" onClick={() => setDismissed((d) => new Set(d).add(j.id))}>
                      <X className="size-3.5" />
                    </button>
                  )}
                </span>
              </div>
              {j.status !== "done" && <ProgressBar value={j.progress} />}
              <div className="mt-1 flex justify-between gap-3 text-[12px] text-muted tabular-nums">
                <span>
                  {st.total ? t("status.of", { used: bytes(st.received ?? 0), total: bytes(st.total) }) : ""}
                  {j.status === "running" && st.speed ? ` · ${bytes(st.speed)}/s` : ""}
                </span>
                <span className={cx(j.status === "failed" && "text-danger")} title={j.error ?? undefined}>
                  {j.status === "running" ? (left != null ? t("models.left", { time: duration(left) }) : t("models.starting_dots")) : j.status === "done" ? t("models.done") : j.status === "failed" ? (j.error ?? t("models.failed")) : t("models.paused")}
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
  const t = useT();
  const [tab, setTab] = useState<"catalog" | "find" | "local">("local");
  const runtime = useApp((s) => s.runtime);
  const refreshRuntime = useApp((s) => s.refreshRuntime);
  const [hub] = useHubSource();
  const [version, setVersion] = useState(0);
  const [importErrors, setImportErrors] = useState<string[]>([]);
  const imported = (errors: string[]) => {
    setImportErrors(errors);
    setVersion((v) => v + 1);
  };
  const dropping = useFileDrop(async (paths) => {
    setTab("local");
    imported(await importPaths(paths));
  });
  // Files opened with Temper, and links to a model, from the desktop shell (lib/desktop.ts).
  const opened = useOpened();
  useEffect(() => {
    if (opened.search) setTab("find");
  }, [opened.search]);
  useEffect(() => {
    if (!opened.models.length) return;
    const paths = opened.models;
    opened.set({ models: [] });
    setTab("local");
    void importPaths(paths).then(imported);
  }, [opened]);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {dropping && (
        <div className="pointer-events-none absolute inset-3 z-20 flex items-center justify-center rounded-xl border-2 border-dashed border-accent bg-bg/85 text-[15px] font-medium">
          {t("models.drop")}
        </div>
      )}
      <PageHeader
        title={t("nav.models")}
        description={t("models.description", { source: sourceName(hub?.source ?? "huggingface") })}
        actions={
          runtime?.state === "ready" && (
            <Button size="sm" variant="ghost" onClick={async () => { await api("/api/runtime/unload", { method: "POST" }); void refreshRuntime(); }}>
              {t("models.unload", { model: runtime.model ?? "" })}
            </Button>
          )
        }
      />
      <div className="flex gap-1 border-b border-line px-7" role="tablist">
        {(
          [
            ["local", t("models.tab_local")],
            ["catalog", t("models.tab_catalog")],
            ["find", t("models.tab_find")],
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
        {importErrors.length > 0 && (
          <div className="mx-7 mt-5">
            <Notice tone="error">
              {importErrors.map((e) => (
                <p key={e}>{e}</p>
              ))}
            </Notice>
          </div>
        )}
        {tab === "find" ? <FindModels /> : tab === "catalog" ? <CatalogView renderRepo={(id) => <RepoDetail id={id} />} /> : (
          <>
            <LocalModels version={version} onImported={imported} />
            <DeviceModels />
            <CloudModels />
          </>
        )}
      </div>
    </div>
  );
}
