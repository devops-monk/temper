import { FolderOpen, Loader2, Play, Sparkles, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Button, Field, Input, ProgressBar, Select } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { Setup } from "./Audio";

interface Base {
  id: string;
  name: string;
  size: number;
  about: string;
  downloaded: boolean;
}

interface Retrieval {
  recall_at_1: number;
  recall_at_5: number;
  mrr_at_10: number;
  candidates: number;
}

interface Tuned {
  name: string;
  base_id: string;
  dataset: string;
  created: number;
  train_pairs: number;
  eval_pairs: number;
  epochs: number;
  dimensions: number;
  before: Retrieval | null;
  after: Retrieval | null;
}

interface Status {
  installed: boolean;
  bases: Base[];
  tuned: Tuned[];
}

interface Peek {
  columns: string[];
  anchor_column?: string;
  positive_column?: string;
  pairs?: number;
  sample?: [string, string][];
  problem: string | null;
}

const pct = (n: number) => `${Math.round(n * 100)}%`;

/** Train → Embeddings: tune an embedding model on pairs, so search finds what you mean (engine embedders.py). */
export function Embedders() {
  const [st, setSt] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => void api<Status>("/api/embedders").then(setSt, (e: Error) => setError(e.message)), []);
  useEffect(load, [load]);

  if (!st) return error ? <p className="p-7 text-[12px] text-danger">{error}</p> : null;
  const base = st.bases[0];
  if (!st.installed || !st.bases.some((b) => b.downloaded))
    return (
      <div className="px-7 py-6">
        <Setup
          installed={st.installed}
          size={base.size}
          repo={base.id}
          install="/api/classifiers/install"
          download={() => api<Job>("/api/classifiers/encoders/download", { json: { encoder: base.id } })}
          icon={<Sparkles className="size-8" />}
          title={st.installed ? `Download ${base.name}` : "Set up embedding training"}
          about={
            st.installed
              ? `The model to start from, ${bytes(base.size)}.`
              : "Tuning teaches an embedding model which texts belong together in your data, so search finds the right passage. It needs PyTorch and sentence-transformers, which install into their own environment (about 1 GB), shared with classifiers."
          }
          onDone={load}
        />
      </div>
    );

  return (
    <div className="flex min-h-0 flex-1">
      <div className="w-[400px] shrink-0 overflow-y-auto border-r border-line px-7 py-5">
        <TuneForm st={st} onTuned={load} />
      </div>
      <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
        <h2 className="mb-3 text-[13px] font-semibold">Your embedding models</h2>
        {st.tuned.length === 0 ? (
          <p className="text-[13px] text-muted">
            None yet. Pairs are all it needs: a question and the passage that answers it, or a search and what it should find. A hundred pairs is a good start.
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {st.tuned.map((t) => (
              <Row key={t.name} t={t} onDelete={setSt} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function TuneForm({ st, onTuned }: { st: Status; onTuned: () => void }) {
  const { refreshJobs } = useApp();
  const [name, setName] = useState("");
  const [dataset, setDataset] = useState("");
  const [anchor, setAnchor] = useState("");
  const [positive, setPositive] = useState("");
  const [base, setBase] = useState(st.bases[0].id);
  const [epochs, setEpochs] = useState(1);
  const [peek, setPeek] = useState<Peek | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const phase = [...events].reverse().find((e) => e.event === "phase")?.name as string | undefined;
  const chosen = st.bases.find((b) => b.id === base);

  useEffect(() => {
    setPeek(null);
    if (!dataset.trim()) return;
    const t = setTimeout(
      () =>
        void api<Peek>("/api/embedders/peek", { json: { dataset: dataset.trim(), anchor_column: anchor || null, positive_column: positive || null } }).then(setPeek, (e: Error) =>
          setPeek({ columns: [], problem: e.message }),
        ),
      300,
    );
    return () => clearTimeout(t);
  }, [dataset, anchor, positive]);

  useEffect(() => {
    if (!status || status === "running") return;
    if (status === "failed") setError((events.find((e) => e.event === "status")?.error as string) ?? "It failed.");
    setJobId(null);
    onTuned();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function browse() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const file = await open({ filters: [{ name: "Pairs", extensions: ["csv", "jsonl", "json"] }] });
    if (typeof file === "string") {
      setDataset(file);
      if (!name) setName(file.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, ""));
    }
  }

  async function tune() {
    setError(null);
    try {
      if (chosen && !chosen.downloaded) {
        const dl = await api<Job>("/api/classifiers/encoders/download", { json: { encoder: base } });
        setError(`Downloading ${chosen.name} first (job ${dl.id}); tune again when it's done.`);
        void refreshJobs();
        return;
      }
      const body = { name, dataset: dataset.trim(), anchor_column: anchor || null, positive_column: positive || null, base, epochs };
      setJobId((await api<Job>("/api/embedders", { json: body })).id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const columns = peek?.columns ?? [];
  const column = (label: string, value: string, found: string | undefined, set: (v: string) => void) => (
    <Field label={label}>
      <Select value={value || found || ""} onChange={(e) => set(e.target.value)} aria-label={label}>
        {!found && <option value="">Choose…</option>}
        {columns.map((c) => (
          <option key={c}>{c}</option>
        ))}
      </Select>
    </Field>
  );
  return (
    <div className="flex flex-col gap-3 text-[13px]">
      <h2 className="text-[13px] font-semibold">Tune an embedding model</h2>
      <Field label="Pairs" hint="A CSV, JSON or JSONL file: a column of questions or searches, and a column of what each should find.">
        <div className="flex gap-2">
          <Input value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="/path/to/pairs.jsonl" spellCheck={false} />
          {inTauri && (
            <Button size="sm" onClick={() => void browse()}>
              <FolderOpen className="size-3.5" />
            </Button>
          )}
        </div>
      </Field>
      {columns.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {column("Question column", anchor, peek?.anchor_column, setAnchor)}
          {column("Answer column", positive, peek?.positive_column, setPositive)}
        </div>
      )}
      {peek?.sample && (
        <div className="rounded-md border border-line bg-surface px-3 py-2 text-[12px]">
          <p className="mb-1 text-muted">{peek.pairs?.toLocaleString()} pairs, such as</p>
          {peek.sample.slice(0, 2).map(([a, p], i) => (
            <p key={i} className="truncate">
              {a} <span className="text-muted">→</span> {p}
            </p>
          ))}
        </div>
      )}
      {peek?.problem && <p className="text-[12px] text-straw">{peek.problem}</p>}
      <Field label="Name">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="our-docs" />
      </Field>
      <Field label="Start from" hint={chosen ? `${bytes(chosen.size)}. ${chosen.about}${chosen.downloaded ? "" : " Downloads first."}` : undefined}>
        <Select value={base} onChange={(e) => setBase(e.target.value)} aria-label="Base model">
          {st.bases.map((b) => (
            <option key={b.id} value={b.id}>
              {b.name}
              {b.downloaded ? "" : " (not downloaded)"}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Passes over the pairs" hint="More helps small sets; too many and it forgets what it knew.">
        <Input type="number" min={1} max={20} value={epochs} onChange={(e) => setEpochs(Math.max(1, +e.target.value || 1))} />
      </Field>
      <Button variant="primary" onClick={() => void tune()} disabled={running || !name.trim() || !peek?.pairs || !!peek.problem}>
        {running ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />} Tune
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

function Row({ t, onDelete }: { t: Tuned; onDelete: (s: Status) => void }) {
  async function remove() {
    if (!confirm(`Delete the embedding model ${t.name}?`)) return;
    onDelete(await api<Status>(`/api/embedders/${t.name}`, { method: "DELETE" }));
  }
  const rows: [string, keyof Retrieval][] = [
    ["Right one first", "recall_at_1"],
    ["In the top 5", "recall_at_5"],
  ];
  return (
    <li className="rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <span className="font-mono text-[12.5px] font-medium">{t.name}</span>
          <p className="truncate text-[12px] text-muted">
            {t.base_id.split("/").pop()} · {t.train_pairs.toLocaleString()} pairs · {t.epochs} {t.epochs === 1 ? "pass" : "passes"} · {relativeDate(t.created)}
          </p>
        </div>
        <button className="text-muted hover:text-danger" onClick={() => void remove()} title="Delete" aria-label={`Delete ${t.name}`}>
          <Trash2 className="size-3.5" />
        </button>
      </div>
      {t.before && t.after ? (
        <table className="mt-3 max-w-[420px] text-[12px]">
          <thead className="text-muted">
            <tr>
              <th className="py-1 text-left font-medium">On {t.eval_pairs} held-out pairs</th>
              <th className="px-3 text-right font-medium">Before</th>
              <th className="px-3 text-right font-medium">After</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, key]) => (
              <tr key={key} className="border-t border-line">
                <td className="py-1">{label}</td>
                <td className="px-3 text-right tabular-nums text-muted">{pct(t.before![key])}</td>
                <td className="px-3 text-right font-semibold tabular-nums">{pct(t.after![key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="mt-2 text-[12px] text-muted">No pairs were held out, so it wasn't measured.</p>
      )}
      <p className="mt-2 text-[12px] text-muted">
        Apps use it as model <span className="font-mono">{t.name}</span> at <span className="font-mono">/v1/embeddings</span>.
      </p>
    </li>
  );
}
