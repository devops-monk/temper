import { Download, ExternalLink, Microscope, Plus, RotateCw, Square, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge, Button, EmptyState, PageHeader, ProgressBar, Select, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { relativeDate } from "../lib/format";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { useJob } from "../lib/useJob";

type Status = "running" | "writing" | "done" | "failed" | "cancelled" | "stopped";

/** A research run as the engine keeps it (engine research.py). */
interface Source {
  n: number;
  url: string;
  title: string;
  snippet: string;
  status: "waiting" | "used" | "irrelevant" | "failed";
  notes: string;
  error?: string;
  included: boolean;
}

interface Run {
  id: string;
  question: string;
  depth: "quick" | "standard" | "thorough";
  created: number;
  status: Status;
  job?: string;
  queries: string[];
  sources: Source[];
  report: string | null;
  error: string | null;
}

interface RunSummary {
  id: string;
  question: string;
  depth: Run["depth"];
  status: Status;
  created: number;
  sources: number;
}

const going = (s: Status) => s === "running" || s === "writing";
const TONE: Record<Status, "neutral" | "ok" | "warn" | "accent"> = { running: "accent", writing: "accent", done: "ok", failed: "warn", cancelled: "neutral", stopped: "neutral" };

/** Saves the report as Markdown: a save dialog in the app, a download in a browser. */
async function exportRun(id: string) {
  const { text, filename } = await api<{ text: string; filename: string }>(`/api/research/${id}/export`);
  if (inTauri) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ defaultPath: filename, filters: [{ name: "Markdown", extensions: ["md"] }] });
    if (path) await api(`/api/research/${id}/export`, { json: { path } });
    return;
  }
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  Object.assign(document.createElement("a"), { href: url, download: filename }).click();
  URL.revokeObjectURL(url);
}

export function ResearchPage() {
  const t = useT();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const refresh = useCallback(() => void api<{ runs: RunSummary[] }>("/api/research").then((r) => setRuns(r.runs), () => setRuns([])), []);
  useEffect(refresh, [refresh]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title={t("research.title")}
        description={t("research.about")}
        actions={
          open && (
            <Button size="sm" onClick={() => setOpen(null)}>
              <Plus className="size-3.5" /> {t("research.new")}
            </Button>
          )
        }
      />
      <div className="flex min-h-0 flex-1">
        <aside className="w-[260px] shrink-0 overflow-y-auto border-r border-line p-2">
          {runs?.length === 0 && <p className="px-2 py-3 text-[12.5px] text-muted">{t("research.none")}</p>}
          {runs?.map((r) => (
            <button
              key={r.id}
              onClick={() => setOpen(r.id)}
              aria-current={open === r.id ? "true" : undefined}
              className={cx("mb-0.5 flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left", open === r.id ? "bg-surface-2" : "hover:bg-surface-2")}
            >
              <span className="line-clamp-2 text-[13px]">{r.question}</span>
              <span className="flex items-center gap-1.5 text-[11.5px] text-muted">
                {going(r.status) ? <span className="text-accent">{t("research.status.running")}</span> : relativeDate(r.created)}
                {r.status === "done" && <span>· {t("research.sources_count", { count: r.sources })}</span>}
              </span>
            </button>
          ))}
        </aside>
        <main className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
          {open ? (
            <RunView key={open} id={open} onChanged={refresh} onDeleted={() => (setOpen(null), refresh())} />
          ) : (
            <NewRun onStarted={(id) => (setOpen(id), refresh())} />
          )}
        </main>
      </div>
    </div>
  );
}

function NewRun({ onStarted }: { onStarted: (id: string) => void }) {
  const t = useT();
  const { runtime } = useApp();
  const [question, setQuestion] = useState("");
  const [depth, setDepth] = useState<Run["depth"]>("standard");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ready = runtime?.state === "ready";

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    await api<Run>("/api/research", { json: { question, depth } }).then((r) => onStarted(r.id), (err: Error) => setError(err.message));
    setBusy(false);
  }

  return (
    <form onSubmit={(e) => void start(e)} className="mx-auto flex max-w-[680px] flex-col gap-3 pt-6">
      <div className="mb-2 flex flex-col items-center text-center">
        <Microscope className="mb-2 size-7 text-muted" />
        <h2 className="text-[17px] font-semibold">{t("research.ask")}</h2>
        <p className="mt-1 text-[13px] text-muted">{t("research.how")}</p>
      </div>
      <textarea
        autoFocus
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && (e.metaKey || e.ctrlKey) && e.currentTarget.form?.requestSubmit()}
        rows={3}
        placeholder={t("research.placeholder")}
        className="w-full resize-y rounded-xl border border-line bg-surface px-4 py-3 text-[14px] placeholder:text-muted/70 focus:border-accent focus:outline-none"
      />
      <div className="flex items-center gap-2">
        <Select value={depth} onChange={(e) => setDepth(e.target.value as Run["depth"])} className="w-auto" aria-label={t("research.depth")}>
          {(["quick", "standard", "thorough"] as const).map((d) => (
            <option key={d} value={d}>
              {t(`research.depth.${d}` as Key)}
            </option>
          ))}
        </Select>
        <span className="flex-1 text-[12px] text-muted">{t(`research.depth.${depth}_about` as Key)}</span>
        <Button variant="primary" type="submit" disabled={!ready || busy || question.trim().length < 3}>
          {t("research.start")}
        </Button>
      </div>
      {!ready && <p className="text-[12.5px] text-straw">{t("research.needs_model")}</p>}
      {error && <p className="text-[12.5px] text-danger">{error}</p>}
    </form>
  );
}

/** The report's [n] citations as links to its sources. */
const linked = (report: string, count: number) => report.replace(/\[(\d+)\](?!\()/g, (m, n) => (+n >= 1 && +n <= count ? `[[${n}]](#source-${n})` : m));

function RunView({ id, onChanged, onDeleted }: { id: string; onChanged: () => void; onDeleted: () => void }) {
  const t = useT();
  const [run, setRun] = useState<Run | null>(null);
  const [excluded, setExcluded] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [lit, setLit] = useState<number | null>(null);
  const load = useCallback(() => void api<Run>(`/api/research/${id}`).then((r) => (setRun(r), setExcluded(new Set(r.sources.filter((s) => !s.included).map((s) => s.n)))), () => {}), [id]);
  useEffect(load, [load]);
  const live = run && going(run.status) ? (run.job ?? null) : null;
  const { status, events } = useJob(live);
  const line = [...events].reverse().find((e) => e.event === "log")?.line as string | undefined;
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;

  // Follow a run while it goes: reload as it reads each source, and once more when it ends.
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(load, 2500);
    return () => clearInterval(timer);
  }, [live, load]);
  useEffect(() => {
    if (status && status !== "running") (load(), onChanged());
  }, [status, load, onChanged]);

  if (!run) return null;
  const used = run.sources.filter((s) => s.status === "used");
  const changed = used.some((s) => excluded.has(s.n) === s.included);

  async function rewrite() {
    setError(null);
    await api<Run>(`/api/research/${id}/write`, { json: { excluded: [...excluded] } }).then(setRun, (e: Error) => setError(e.message));
  }

  async function remove() {
    if (!confirm(t("research.delete_confirm"))) return;
    await api(`/api/research/${id}`, { method: "DELETE" });
    onDeleted();
  }

  function jump(n: number) {
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setLit(n);
    setTimeout(() => setLit(null), 1600);
  }

  return (
    <div className="mx-auto flex max-w-[820px] flex-col gap-5">
      <header className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="text-[17px] leading-snug font-semibold">{run.question}</h2>
          <p className="mt-1 flex items-center gap-2 text-[12px] text-muted">
            <Badge tone={TONE[run.status]}>{t(`research.status.${run.status}` as Key)}</Badge>
            {t(`research.depth.${run.depth}` as Key)} · {relativeDate(run.created)} · {t("research.searches", { count: run.queries.length })}
          </p>
        </div>
        {going(run.status) && run.job ? (
          <Button size="sm" variant="ghost" onClick={() => void api(`/api/jobs/${run.job}/cancel`, { method: "POST" })}>
            <Square className="size-3.5" /> {t("research.stop")}
          </Button>
        ) : (
          <>
            {run.report && (
              <Button size="sm" onClick={() => void exportRun(id).catch((e: Error) => setError(e.message))}>
                <Download className="size-3.5" /> {t("research.export")}
              </Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => void remove()} aria-label={t("research.delete")} title={t("research.delete")}>
              <Trash2 className="size-3.5" />
            </Button>
          </>
        )}
      </header>

      {going(run.status) && (
        <div className="flex flex-col gap-1.5">
          <ProgressBar value={progress ?? null} />
          <p className="truncate font-mono text-[11.5px] text-muted" title={line}>
            {line ?? "…"}
          </p>
        </div>
      )}
      {run.error && <p className="text-[12.5px] text-danger">{run.error}</p>}
      {error && <p className="text-[12.5px] text-danger">{error}</p>}

      {run.report && (
        <article className="prose-chat rounded-xl border border-line bg-surface px-5 py-4 text-[14px] leading-relaxed">
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) =>
                href?.startsWith("#source-") ? (
                  <button className="mx-px align-super text-[10.5px] text-accent hover:underline" onClick={() => jump(Number(href.slice(8)))}>
                    {children}
                  </button>
                ) : (
                  <a href={href} target="_blank" rel="noreferrer">
                    {children}
                  </a>
                ),
            }}
          >
            {linked(run.report, run.sources.length)}
          </Markdown>
        </article>
      )}

      {run.sources.length > 0 && (
        <section>
          <div className="mb-2 flex items-center justify-between gap-3">
            <h3 className="text-[13px] font-semibold">{t("research.sources")}</h3>
            {!going(run.status) && used.length > 0 && (
              <Button size="sm" variant={changed ? "primary" : "ghost"} onClick={() => void rewrite()} disabled={!changed && !!run.report}>
                <RotateCw className="size-3.5" /> {t("research.rewrite")}
              </Button>
            )}
          </div>
          {!going(run.status) && used.length > 0 && <p className="mb-2 text-[12px] text-muted">{t("research.review_hint")}</p>}
          <ol className="flex flex-col gap-2">
            {run.sources.map((s) => (
              <li
                key={s.n}
                id={`source-${s.n}`}
                className={cx("rounded-lg border bg-surface px-3 py-2 text-[13px] transition-colors", lit === s.n ? "border-accent" : "border-line", s.status !== "used" && "opacity-70")}
              >
                <div className="flex items-start gap-2">
                  <span className="mt-px w-5 shrink-0 font-mono text-[12px] text-muted">{s.n}</span>
                  <span className="min-w-0 flex-1">
                    <a href={s.url} target="_blank" rel="noreferrer" className="inline-flex max-w-full items-center gap-1 font-medium hover:underline">
                      <span className="truncate">{s.title || s.url}</span> <ExternalLink className="size-3 shrink-0 text-muted" />
                    </a>
                    <span className="block truncate text-[11.5px] text-muted">{s.url}</span>
                  </span>
                  {s.status === "used" ? (
                    !going(run.status) && (
                      <label className="flex shrink-0 items-center gap-1.5 text-[12px] text-muted">
                        <input
                          type="checkbox"
                          checked={!excluded.has(s.n)}
                          onChange={(e) => setExcluded((x) => (e.target.checked ? new Set([...x].filter((n) => n !== s.n)) : new Set([...x, s.n])))}
                        />
                        {t("research.include")}
                      </label>
                    )
                  ) : (
                    <Badge>{t(`research.source.${s.status}` as Key)}</Badge>
                  )}
                </div>
                {s.notes && <div className="mt-1.5 ml-7 text-[12.5px] whitespace-pre-wrap text-muted">{s.notes}</div>}
                {s.error && <p className="mt-1 ml-7 text-[12px] text-muted">{s.error}</p>}
              </li>
            ))}
          </ol>
        </section>
      )}
      {!run.sources.length && !going(run.status) && <EmptyState icon={<Microscope className="size-6" />} title={t("research.no_sources")} />}
    </div>
  );
}
