import { Database, FlaskConical, FolderOpen, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge, Button, EmptyState, Field, Input, Notice, PageHeader, ProgressBar, Section, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";

type Format = "messages" | "sharegpt" | "alpaca" | "text";
type Turn = { role: string; content: string };

interface Preview {
  source: string;
  columns: string[];
  rows: Record<string, unknown>[];
  format: Format | null;
  local: boolean;
  suggested: { format: Format; mapping: Record<string, string> } | null;
  /** Hugging Face only: the subset and split shown, and the ones available. */
  config?: string;
  configs?: string[];
  split?: string;
  splits?: string[];
}

interface Rendered {
  rows: ({ conversation: Turn[] | string } | { error: string })[];
  usable: number;
}

const FORMAT_LABEL: Record<Format, string> = { messages: "OpenAI messages", sharegpt: "ShareGPT", alpaca: "Prompt and response", text: "Plain text" };
// Roles each format reads, with the column it uses when nothing is mapped.
const ROLES: Record<Format, { role: string; label: string; required: boolean }[]> = {
  messages: [{ role: "messages", label: "Messages (list of role and content)", required: true }],
  sharegpt: [{ role: "conversations", label: "Conversations (list of from and value)", required: true }],
  alpaca: [
    { role: "instruction", label: "Prompt", required: true },
    { role: "input", label: "Extra context", required: false },
    { role: "output", label: "Response", required: true },
    { role: "system", label: "System message", required: false },
  ],
  text: [{ role: "text", label: "Text", required: true }],
};
const SUGGESTIONS = ["mlabonne/FineTome-100k", "yahma/alpaca-cleaned", "HuggingFaceH4/ultrachat_200k", "openai/gsm8k"];

function cell(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 280 ? text.slice(0, 280) + "…" : text;
}

/** Default mapping: each role reads the column of the same name, when the dataset has one. */
function defaultMapping(format: Format, columns: string[]): Record<string, string> {
  return Object.fromEntries(ROLES[format].map(({ role }) => [role, columns.includes(role) ? role : ""]));
}

interface Cleaning {
  dedupe: boolean;
  shuffle: boolean;
  min_chars: number;
  max_chars: number;
  eval_percent: number;
  max_rows: number;
}

function Conversation({ conv }: { conv: Turn[] | string }) {
  if (typeof conv === "string") return <p className="font-mono text-[12px] whitespace-pre-wrap">{cell(conv)}</p>;
  return (
    <div className="flex flex-col gap-1.5">
      {conv.map((t, i) => (
        <div key={i} className="grid grid-cols-[76px_1fr] gap-2 text-[13px]">
          <span className={cx("font-medium", t.role === "assistant" ? "text-accent" : "text-muted")}>{t.role}</span>
          <span className="break-words whitespace-pre-wrap">{t.content.length > 600 ? t.content.slice(0, 600) + "…" : t.content}</span>
        </div>
      ))}
    </div>
  );
}

function PrepareRun({ job }: { job: Job }) {
  const { go, setPendingDataset } = useApp();
  const { events, status } = useJob(job.id);
  const state = status ?? job.status;
  const phase = [...events].reverse().find((e) => e.event === "phase")?.name as string | undefined;
  const result = events.find((e) => e.event === "result") as { output?: string; eval_output?: string | null; stats?: Record<string, number> } | undefined;
  const error = (events.find((e) => e.event === "error")?.message as string | undefined) ?? job.error;
  const s = result?.stats;
  const dropped = s ? [s.duplicates && `${s.duplicates} duplicates`, s.malformed && `${s.malformed} unusable`, s.too_short && `${s.too_short} too short`, s.too_long && `${s.too_long} too long`].filter(Boolean) : [];

  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="truncate font-medium">{job.title}</span>
        <Badge tone={state === "done" ? "ok" : state === "failed" ? "warn" : "neutral"}>{state === "running" ? (phase ?? "Starting") : state}</Badge>
      </div>
      {state === "running" && <ProgressBar value={job.progress} />}
      {state === "failed" && error && <Notice tone="error">{error}</Notice>}
      {s && (
        <p className="text-[13px]">
          {s.train} training rows{s.eval ? ` and ${s.eval} for evaluation` : ""} from {s.rows_in}.{dropped.length > 0 && <span className="text-muted"> Removed {dropped.join(", ")}.</span>}
        </p>
      )}
      {state === "done" && result?.output && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button size="sm" variant="primary" onClick={() => { setPendingDataset({ source: result.output! }); go("train"); }}>
            <FlaskConical className="size-3.5" /> Train on this
          </Button>
          <span className="truncate font-mono text-[11.5px] text-muted" title={result.output}>
            {result.output}
          </span>
        </div>
      )}
    </div>
  );
}

export function DatasetsPage() {
  const { go, setPendingDataset, jobs, refreshJobs } = useApp();
  const [source, setSource] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [format, setFormat] = useState<Format>("alpaca");
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [systemPrompt, setSystemPrompt] = useState("");
  const [cleaning, setCleaning] = useState<Cleaning>({ dedupe: true, shuffle: true, min_chars: 0, max_chars: 0, eval_percent: 5, max_rows: 0 });
  const [view, setView] = useState<"model" | "rows">("model");
  const [rendered, setRendered] = useState<Rendered | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [preparing, setPreparing] = useState(false);

  async function open(src = source, pick: { config?: string; split?: string } = {}) {
    if (!src.trim()) return;
    setSource(src);
    setError(null);
    setLoading(true);
    try {
      const params = new URLSearchParams({ source: src.trim(), ...(pick.config ? { config: pick.config } : {}), ...(pick.split ? { split: pick.split } : {}) });
      const p = await api<Preview>(`/api/datasets/preview?${params}`);
      setPreview(p);
      const fmt = p.format ?? p.suggested?.format ?? "alpaca";
      setFormat(fmt);
      setMapping(p.format ? defaultMapping(fmt, p.columns) : { ...defaultMapping(fmt, p.columns), ...(p.suggested?.mapping ?? {}) });
      setSystemPrompt("");
    } catch (e) {
      setPreview(null);
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function browse() {
    const { open: pick } = await import("@tauri-apps/plugin-dialog");
    const file = await pick({ filters: [{ name: "Datasets", extensions: ["jsonl", "json", "csv", "parquet"] }] });
    if (typeof file === "string") void open(file);
  }

  const missing = ROLES[format].filter((r) => r.required && !mapping[r.role]).map((r) => r.label);
  const body = preview && { source: preview.source, config: preview.config, split: preview.split ?? "train", format, mapping, system_prompt: systemPrompt };

  // Re-render the model's view shortly after the mapping stops changing.
  const key = JSON.stringify(body);
  useEffect(() => {
    if (!body || missing.length) {
      setRendered(null);
      return;
    }
    const timer = setTimeout(() => {
      api<Rendered>("/api/datasets/render", { json: body })
        .then((r) => {
          setRendered(r);
          setRenderError(null);
        })
        .catch((e) => setRenderError((e as Error).message));
    }, 300);
    return () => clearTimeout(timer);
  }, [key, missing.length]);

  async function prepare() {
    if (!body) return;
    setPreparing(true);
    setError(null);
    try {
      const { eval_percent, ...rest } = cleaning;
      await api("/api/datasets/prepare", { json: { ...body, ...rest, eval_fraction: eval_percent / 100 } });
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPreparing(false);
    }
  }

  const runs = preview ? jobs.filter((j) => j.kind === "prepare" && j.title === `Prepare ${preview.source.split("/").pop()}`) : [];
  const setClean = <K extends keyof Cleaning>(k: K) => (v: Cleaning[K]) => setCleaning((c) => ({ ...c, [k]: v }));

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Datasets"
        description="Check a dataset, map its columns to chat turns, and clean it before you train."
        actions={
          preview?.format && (
            <Button onClick={() => { setPendingDataset({ source: preview.source, config: preview.config, split: preview.split }); go("train"); }}>
              <FlaskConical className="size-4" /> Train on it as is
            </Button>
          )
        }
      />
      <div className="border-b border-line px-7 py-4">
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void open(); }}>
          <Input value={source} onChange={(e) => setSource(e.target.value)} placeholder="Hugging Face dataset id or a local file path" spellCheck={false} />
          {inTauri && (
            <Button type="button" onClick={browse}>
              <FolderOpen className="size-4" /> Choose file
            </Button>
          )}
          <Button type="submit" variant="primary" disabled={loading || !source.trim()}>
            {loading ? "Opening…" : "Open"}
          </Button>
        </form>
      </div>
      {!preview ? (
        <div className="min-h-0 flex-1 overflow-auto px-7 py-5">
          {error && <Notice tone="error">{error}</Notice>}
          {!error && (
            <EmptyState icon={<Database className="size-8" />} title="Open a dataset to preview it">
              <p>Try one of these:</p>
              <div className="mt-3 flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <Button key={s} size="sm" onClick={() => void open(s)}>
                    {s}
                  </Button>
                ))}
              </div>
            </EmptyState>
          )}
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          <div className="w-[400px] shrink-0 overflow-y-auto border-r border-line px-7 pb-8">
            <Section title="Format" aside={preview.format ? <Badge tone="ok">Detected</Badge> : preview.suggested ? <Badge tone="accent">Suggested</Badge> : <Badge tone="warn">Pick columns</Badge>}>
              <div className="flex flex-col gap-3">
                {((preview.configs?.length ?? 0) > 1 || (preview.splits?.length ?? 0) > 1) && (
                  <div className="grid grid-cols-2 gap-3">
                    <Field label="Subset">
                      <Select value={preview.config} onChange={(e) => void open(preview.source, { config: e.target.value })} disabled={loading}>
                        {preview.configs?.map((c) => (
                          <option key={c}>{c}</option>
                        ))}
                      </Select>
                    </Field>
                    <Field label="Split">
                      <Select value={preview.split} onChange={(e) => void open(preview.source, { config: preview.config, split: e.target.value })} disabled={loading}>
                        {preview.splits?.map((s) => (
                          <option key={s}>{s}</option>
                        ))}
                      </Select>
                    </Field>
                  </div>
                )}
                <Field label="Shape">
                  <Select
                    value={format}
                    onChange={(e) => {
                      const f = e.target.value as Format;
                      setFormat(f);
                      setMapping(defaultMapping(f, preview.columns));
                    }}
                  >
                    {(Object.keys(FORMAT_LABEL) as Format[]).map((f) => (
                      <option key={f} value={f}>
                        {FORMAT_LABEL[f]}
                      </option>
                    ))}
                  </Select>
                </Field>
                {ROLES[format].map(({ role, label, required }) => (
                  <Field key={role} label={label}>
                    <Select value={mapping[role] ?? ""} onChange={(e) => setMapping((m) => ({ ...m, [role]: e.target.value }))}>
                      <option value="">{required ? "Choose a column…" : "None"}</option>
                      {preview.columns.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </Select>
                  </Field>
                ))}
                {format !== "text" && (
                  <Field label="System prompt" hint="Optional. Added to every conversation that doesn't have one.">
                    <textarea
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      rows={2}
                      className="w-full resize-y rounded-md border border-line bg-surface px-3 py-2 focus:border-accent focus:outline-none"
                    />
                  </Field>
                )}
              </div>
            </Section>
            <Section title="Cleaning">
              <div className="flex flex-col gap-3">
                <Toggle checked={cleaning.dedupe} onChange={setClean("dedupe")} label="Remove duplicate rows" />
                <Toggle checked={cleaning.shuffle} onChange={setClean("shuffle")} label="Shuffle" />
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Shortest" hint="Characters; 0 keeps all.">
                    <Input type="number" min={0} value={cleaning.min_chars} onChange={(e) => setClean("min_chars")(Number(e.target.value))} />
                  </Field>
                  <Field label="Longest" hint="Characters; 0 keeps all.">
                    <Input type="number" min={0} value={cleaning.max_chars} onChange={(e) => setClean("max_chars")(Number(e.target.value))} />
                  </Field>
                  <Field label="Evaluation split" hint="Percent held out.">
                    <Input type="number" min={0} max={50} value={cleaning.eval_percent} onChange={(e) => setClean("eval_percent")(Number(e.target.value))} />
                  </Field>
                  <Field label="Rows to keep" hint="0 keeps all.">
                    <Input type="number" min={0} value={cleaning.max_rows} onChange={(e) => setClean("max_rows")(Number(e.target.value))} />
                  </Field>
                </div>
                <Button variant="primary" onClick={() => void prepare()} disabled={preparing || missing.length > 0}>
                  <Sparkles className="size-4" /> {preparing ? "Starting…" : "Prepare dataset"}
                </Button>
                {missing.length > 0 && <span className="text-xs text-muted">Choose a column for: {missing.join(", ")}.</span>}
              </div>
            </Section>
          </div>
          <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
            {error && (
              <div className="mb-4">
                <Notice tone="error">{error}</Notice>
              </div>
            )}
            {runs.length > 0 && (
              <div className="mb-5 flex flex-col gap-3">
                {runs.map((j) => (
                  <PrepareRun key={j.id} job={j} />
                ))}
              </div>
            )}
            <div className="mb-3 flex items-center gap-3">
              <div className="flex rounded-md border border-line bg-surface p-0.5" role="tablist">
                {(
                  [
                    ["model", "As the model sees it"],
                    ["rows", "Raw rows"],
                  ] as const
                ).map(([k, label]) => (
                  <button key={k} role="tab" aria-selected={view === k} onClick={() => setView(k)} className={cx("rounded px-2.5 py-1 text-[13px] font-medium", view === k ? "bg-surface-2 text-text" : "text-muted hover:text-text")}>
                    {label}
                  </button>
                ))}
              </div>
              <span className="truncate text-[13px] text-muted">
                {view === "model" && rendered ? `${rendered.usable} of ${rendered.rows.length} sample rows usable` : `First ${preview.rows.length} rows of ${preview.source}`}
              </span>
            </div>
            {view === "model" ? (
              missing.length ? (
                <Notice tone="info">Map the {missing.join(" and ")} column{missing.length > 1 ? "s" : ""} to see the conversations.</Notice>
              ) : renderError ? (
                <Notice tone="error">{renderError}</Notice>
              ) : (
                <div className="flex flex-col gap-3">
                  {rendered?.rows.map((r, i) =>
                    "error" in r ? (
                      <div key={i} className="rounded-lg border border-dashed border-line px-4 py-2.5 text-[13px] text-muted">
                        Row {i + 1} skipped: {r.error}
                      </div>
                    ) : (
                      <div key={i} className="rounded-lg border border-line bg-surface px-4 py-3">
                        <Conversation conv={r.conversation} />
                      </div>
                    ),
                  )}
                </div>
              )
            ) : (
              <div className="overflow-x-auto rounded-lg border border-line bg-surface">
                <table className="w-full text-left text-[12.5px]">
                  <thead className="bg-surface-2">
                    <tr>
                      <th className="w-10 px-3 py-2 font-medium text-muted">#</th>
                      {preview.columns.map((c) => (
                        <th key={c} className="px-3 py-2 font-mono font-medium">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i} className="border-t border-line align-top">
                        <td className="px-3 py-2 text-muted tabular-nums">{i + 1}</td>
                        {preview.columns.map((c) => (
                          <td key={c} className="max-w-[420px] px-3 py-2 font-mono text-[11.5px] break-words">
                            {cell(row[c])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
