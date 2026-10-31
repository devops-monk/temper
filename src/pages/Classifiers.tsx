import { ChevronDown, ChevronRight, FolderOpen, Loader2, Play, Tags, Trash2 } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useState } from "react";
import { Badge, Button, Field, Input, ProgressBar, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { Setup } from "./Audio";

interface Encoder {
  id: string;
  name: string;
  size: number;
  about: string;
  downloaded: boolean;
}

interface LabelMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

interface Classifier {
  name: string;
  labels: string[];
  dataset: string;
  encoder_id: string;
  created: number;
  fine_tuned: boolean;
  train_examples: number;
  eval_examples: number;
  metrics: { accuracy: number; macro_f1: number; labels: Record<string, LabelMetrics>; confusion: number[][]; mistakes: { text: string; label: string; guess: string }[] } | null;
}

interface Status {
  installed: boolean;
  encoders: Encoder[];
  classifiers: Classifier[];
}

interface Peek {
  columns: string[];
  rows: number;
  text_column?: string;
  label_column?: string;
  examples?: number;
  labels?: Record<string, number>;
  problem: string | null;
}

interface Result {
  label: string;
  confidence: number;
  scores: Record<string, number>;
}

const pct = (n: number) => `${Math.round(n * 100)}%`;

/** Train → Classifier: a small model that sorts text into your labels, trained on examples (engine classifiers.py). */
export function Classifiers() {
  const [st, setSt] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => void api<Status>("/api/classifiers").then(setSt, (e: Error) => setError(e.message)), []);
  useEffect(load, [load]);

  if (!st) return error ? <p className="p-7 text-[12px] text-danger">{error}</p> : null;
  const encoder = st.encoders[0];
  if (!st.installed || !st.encoders.some((e) => e.downloaded))
    return (
      <div className="px-7 py-6">
        <Setup
          installed={st.installed}
          size={encoder.size}
          repo={encoder.id}
          install="/api/classifiers/install"
          download={() => api<Job>("/api/classifiers/encoders/download", { json: { encoder: encoder.id } })}
          icon={<Tags className="size-8" />}
          title={st.installed ? `Download ${encoder.name}` : "Set up classifier training"}
          about={
            st.installed
              ? `The encoder that reads each text, ${bytes(encoder.size)}.`
              : "A classifier sorts text into your labels (billing or bug, spam or not) after learning from examples, and answers in milliseconds. Training needs PyTorch and sentence-transformers, which install into their own environment (about 1 GB)."
          }
          onDone={load}
        />
      </div>
    );

  return (
    <div className="flex min-h-0 flex-1">
      <div className="w-[400px] shrink-0 overflow-y-auto border-r border-line px-7 py-5">
        <TrainForm st={st} onTrained={load} />
      </div>
      <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
        <h2 className="mb-3 text-[13px] font-semibold">Your classifiers</h2>
        {st.classifiers.length === 0 ? (
          <p className="text-[13px] text-muted">None yet. A file of examples, each a text and its label, is all a classifier needs; a dozen of each label is a good start.</p>
        ) : (
          <ul className="flex flex-col gap-3">
            {st.classifiers.map((c) => (
              <Row key={c.name} c={c} onDelete={(s) => setSt(s)} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function TrainForm({ st, onTrained }: { st: Status; onTrained: () => void }) {
  const { refreshJobs } = useApp();
  const [name, setName] = useState("");
  const [dataset, setDataset] = useState("");
  const [textColumn, setTextColumn] = useState("");
  const [labelColumn, setLabelColumn] = useState("");
  const [encoder, setEncoder] = useState(st.encoders[0].id);
  const [tune, setTune] = useState(false);
  const [peek, setPeek] = useState<Peek | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const phase = [...events].reverse().find((e) => e.event === "phase")?.name as string | undefined;
  const chosen = st.encoders.find((e) => e.id === encoder);

  useEffect(() => {
    setPeek(null);
    if (!dataset.trim()) return;
    const t = setTimeout(
      () =>
        void api<Peek>("/api/classifiers/peek", { json: { dataset: dataset.trim(), text_column: textColumn || null, label_column: labelColumn || null } }).then(setPeek, (e: Error) =>
          setPeek({ columns: [], rows: 0, problem: e.message }),
        ),
      300,
    );
    return () => clearTimeout(t);
  }, [dataset, textColumn, labelColumn]);

  useEffect(() => {
    if (!status || status === "running") return;
    if (status === "failed") setError((events.find((e) => e.event === "status")?.error as string) ?? "It failed.");
    setJobId(null);
    onTrained();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function browse() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const file = await open({ filters: [{ name: "Examples", extensions: ["csv", "jsonl", "json"] }] });
    if (typeof file === "string") {
      setDataset(file);
      if (!name) setName(file.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, ""));
    }
  }

  async function train() {
    setError(null);
    try {
      const body = { name, dataset: dataset.trim(), text_column: textColumn || null, label_column: labelColumn || null, encoder, fine_tune: tune };
      if (chosen && !chosen.downloaded) {
        const dl = await api<Job>("/api/classifiers/encoders/download", { json: { encoder } });
        setError(`Downloading ${chosen.name} first (job ${dl.id}); train again when it's done.`);
        void refreshJobs();
        return;
      }
      setJobId((await api<Job>("/api/classifiers", { json: body })).id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const columns = peek?.columns ?? [];
  return (
    <div className="flex flex-col gap-3 text-[13px]">
      <h2 className="text-[13px] font-semibold">Train a classifier</h2>
      <Field label="Examples" hint="A CSV, JSON or JSONL file: a column of text and a column of labels.">
        <div className="flex gap-2">
          <Input value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="/path/to/examples.csv" spellCheck={false} />
          {inTauri && (
            <Button size="sm" onClick={() => void browse()}>
              <FolderOpen className="size-3.5" />
            </Button>
          )}
        </div>
      </Field>
      {columns.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          <Field label="Text column">
            <Select value={textColumn || peek?.text_column || ""} onChange={(e) => setTextColumn(e.target.value)} aria-label="Text column">
              {!peek?.text_column && <option value="">Choose…</option>}
              {columns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </Select>
          </Field>
          <Field label="Label column">
            <Select value={labelColumn || peek?.label_column || ""} onChange={(e) => setLabelColumn(e.target.value)} aria-label="Label column">
              {!peek?.label_column && <option value="">Choose…</option>}
              {columns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </Select>
          </Field>
        </div>
      )}
      {peek?.labels && (
        <div className="rounded-md border border-line bg-surface px-3 py-2">
          <p className="mb-1 text-[12px] text-muted">
            {peek.examples?.toLocaleString()} examples, {Object.keys(peek.labels).length} labels
          </p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(peek.labels)
              .slice(0, 24)
              .map(([label, n]) => (
                <Badge key={label}>
                  {label} · {n}
                </Badge>
              ))}
          </div>
        </div>
      )}
      {peek?.problem && <p className="text-[12px] text-straw">{peek.problem}</p>}
      <Field label="Name">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="support-tickets" />
      </Field>
      <Field label="Encoder" hint={chosen ? `${bytes(chosen.size)}. ${chosen.about}${chosen.downloaded ? "" : " Downloads first."}` : undefined}>
        <Select value={encoder} onChange={(e) => setEncoder(e.target.value)} aria-label="Encoder">
          {st.encoders.map((e) => (
            <option key={e.id} value={e.id}>
              {e.name}
              {e.downloaded ? "" : " (not downloaded)"}
            </option>
          ))}
        </Select>
      </Field>
      <Toggle label="Tune the encoder on pairs of examples (slower; best with few examples)" checked={tune} onChange={setTune} />
      <Button variant="primary" onClick={() => void train()} disabled={running || !name.trim() || !peek?.labels || !!peek.problem}>
        {running ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />} Train
      </Button>
      {running && (
        <div className="flex flex-col gap-1.5">
          <ProgressBar value={progress ?? null} />
          <p className="text-[12px] text-muted">{phase ?? "Starting…"}</p>
        </div>
      )}
      {error && <p className="text-[12px] text-danger">{error}</p>}
    </div>
  );
}

function Row({ c, onDelete }: { c: Classifier; onDelete: (s: Status) => void }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const m = c.metrics;

  async function tryIt() {
    setBusy(true);
    setError(null);
    try {
      const r = await api<Result & { classifier: string }>(`/v1/classifiers/${c.name}`, { json: { text } });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm(`Delete the classifier ${c.name}?`)) return;
    onDelete(await api<Status>(`/api/classifiers/${c.name}`, { method: "DELETE" }));
  }

  return (
    <li className="rounded-lg border border-line bg-surface text-[13px]">
      <div className="flex items-center gap-3 px-4 py-3">
        <button onClick={() => setOpen(!open)} className="text-muted hover:text-text" aria-expanded={open} aria-label={open ? "Hide details" : "Show details"}>
          {open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[12.5px] font-medium">{c.name}</span>
            {c.fine_tuned && <Badge tone="accent">tuned</Badge>}
          </div>
          <p className="truncate text-[12px] text-muted">
            {c.labels.join(", ")} · {c.train_examples.toLocaleString()} examples · {relativeDate(c.created)}
          </p>
        </div>
        {m && (
          <span className="text-right">
            <span className="block text-[15px] font-semibold tabular-nums">{pct(m.accuracy)}</span>
            <span className="block text-[11px] text-muted">on {c.eval_examples} held out</span>
          </span>
        )}
        <button className="text-muted hover:text-danger" onClick={() => void remove()} title="Delete" aria-label={`Delete ${c.name}`}>
          <Trash2 className="size-3.5" />
        </button>
      </div>
      {open && (
        <div className="flex flex-col gap-4 border-t border-line px-4 py-3">
          <form
            className="flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim()) void tryIt();
            }}
          >
            <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Try it on some text" aria-label={`Text for ${c.name}`} />
            <Button type="submit" disabled={!text.trim() || busy}>
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />} Classify
            </Button>
          </form>
          {result && (
            <div className="flex flex-col gap-1">
              {Object.entries(result.scores).map(([label, score]) => (
                <div key={label} className="flex items-center gap-2 text-[12px]">
                  <span className={cx("w-28 truncate", label === result.label && "font-semibold")}>{label}</span>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <span className="block h-full rounded-full bg-accent" style={{ width: pct(score) }} />
                  </span>
                  <span className="w-10 text-right tabular-nums text-muted">{pct(score)}</span>
                </div>
              ))}
            </div>
          )}
          {error && <p className="text-[12px] text-danger">{error}</p>}
          {m ? <Metrics c={c} /> : <p className="text-[12px] text-muted">Too few examples were held out to measure it.</p>}
          <p className="text-[12px] text-muted">
            Trained on {c.dataset.split(/[\\/]/).pop()} with {c.encoder_id.split("/").pop()}. Apps use it at <span className="font-mono">/v1/classifiers/{c.name}</span>, and a saved decision on
            Connect can answer with it.
          </p>
        </div>
      )}
    </li>
  );
}

function Cell({ children, strong, dim }: { children: ReactNode; strong?: boolean; dim?: boolean }) {
  return <td className={cx("px-2 py-1 text-right tabular-nums", strong && "font-semibold", dim && "text-muted")}>{children}</td>;
}

/** How it did on the held-out examples: per label, which labels it confused, and what it got wrong. */
function Metrics({ c }: { c: Classifier }) {
  const m = c.metrics!;
  return (
    <div className="flex flex-col gap-4">
      <table className="max-w-[520px] text-[12px]">
        <thead className="text-muted">
          <tr>
            <th className="py-1 text-left font-medium">Label</th>
            <th className="px-2 text-right font-medium">Precision</th>
            <th className="px-2 text-right font-medium">Recall</th>
            <th className="px-2 text-right font-medium">Held out</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(m.labels).map(([label, v]) => (
            <tr key={label} className="border-t border-line">
              <td className="py-1">{label}</td>
              <Cell>{pct(v.precision)}</Cell>
              <Cell>{pct(v.recall)}</Cell>
              <Cell dim>{v.support}</Cell>
            </tr>
          ))}
        </tbody>
      </table>
      {c.labels.length <= 12 && (
        <div>
          <p className="mb-1 text-[12px] text-muted">Which label it picked (columns) for each true label (rows)</p>
          <table className="text-[12px]">
            <thead className="text-muted">
              <tr>
                <th />
                {c.labels.map((l) => (
                  <th key={l} className="max-w-20 truncate px-2 text-right font-medium" title={l}>
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {m.confusion.map((row, i) => (
                <tr key={c.labels[i]} className="border-t border-line">
                  <td className="max-w-28 truncate py-1 pr-2">{c.labels[i]}</td>
                  {row.map((n, j) => (
                    <Cell key={j} strong={i === j && n > 0} dim={n === 0}>
                      {n}
                    </Cell>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {m.mistakes.length > 0 && (
        <div>
          <p className="mb-1 text-[12px] text-muted">What it got wrong</p>
          <ul className="flex flex-col gap-1 text-[12px]">
            {m.mistakes.map((x, i) => (
              <li key={i}>
                “{x.text}” <span className="text-muted">is {x.label}; it said</span> {x.guess}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
