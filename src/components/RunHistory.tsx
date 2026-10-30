import { ChevronDown, ChevronRight, MessageSquare, RotateCcw, StepForward } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import { useEngines } from "./LoadButton";
import { SampleReplies, type Sample } from "./SampleReplies";
import { Badge, Button, Notice, Section, cx } from "./ui";

/** A past fine-tuning run (engine training.summary). */
interface Run {
  id: string;
  name: string | null;
  base_model: string | null;
  dataset: string | null;
  method: string | null;
  backend: "mlx" | "gpu" | null;
  adapter: string | null;
  path: string;
  created: number;
  status: string;
  steps: number;
  total_steps: number | null;
  loss: number | null;
  val_loss: number | null;
  dataset_sha256: string | null;
  checkpoints: number;
}

interface Detail extends Run {
  metrics: { step: number; loss: number | null }[];
  evals: { step: number; val_loss: number | null }[];
  provenance: Record<string, unknown> | null;
  error: string | null;
  best: { step: number; val_loss: number; stopped_at: number | null } | null;
  samples: Sample[];
  checkpoint_list: { step: number; path: string; bytes: number }[];
}

// Line colours for compared runs, in the order they're ticked.
const COLORS = ["#E3BE63", "#4F86E8", "#C9864A", "#8D65CB", "#5BAE7E", "#D9667A"];

const short = (p: string | null | undefined) => (p ? p.split(/[\\/]/).pop()! : "");
const fixed = (n: number | null | undefined) => (n == null ? "—" : n.toFixed(3));

/** Every run Temper has trained, newest first: what went in, how it went, and its checkpoints to chat with or carry on from. */
export function RunHistory({ tick }: { tick: number }) {
  const { refreshJobs, go } = useApp();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [compare, setCompare] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api<Run[]>("/api/train/runs").then(setRuns, (e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load, tick]);

  async function act(call: Promise<unknown>, after?: () => void) {
    setError(null);
    try {
      await call;
      after?.();
    } catch (e) {
      setError((e as Error).message);
    }
    void refreshJobs();
    load();
  }

  if (!runs?.length) return error ? <Notice tone="error">{error}</Notice> : null;
  const toggle = (id: string) => setCompare((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id]));

  return (
    <Section title="Past runs" aside={compare.length === 1 ? <span className="text-[12px] text-muted">Tick another run to compare</span> : undefined}>
      {compare.length > 1 && <Compare ids={compare} runs={runs} />}
      {error && <Notice tone="error">{error}</Notice>}
      <ul className="divide-y divide-line rounded-lg border border-line">
        {runs.map((r) => (
          <li key={r.id}>
            <div className="flex items-center gap-3 px-3 py-2 text-[13px]">
              <input type="checkbox" checked={compare.includes(r.id)} onChange={() => toggle(r.id)} aria-label={`Compare ${r.name ?? r.id}`} title="Compare" />
              <button className="flex min-w-0 flex-1 items-center gap-1.5 text-left" onClick={() => setOpen(open === r.id ? null : r.id)}>
                {open === r.id ? <ChevronDown className="size-3.5 shrink-0" /> : <ChevronRight className="size-3.5 shrink-0" />}
                <span className="min-w-0">
                  <span className="block truncate font-medium">{r.name || r.id}</span>
                  <span className="block truncate text-[12px] text-muted">
                    {short(r.base_model)} · {short(r.dataset)}
                  </span>
                </span>
              </button>
              <Badge tone={r.status === "done" ? "ok" : r.status === "running" ? "accent" : r.status === "failed" ? "warn" : "neutral"}>{r.status}</Badge>
              <span className="w-20 text-right tabular-nums text-muted" title="Steps">
                {r.steps ? `${r.steps}${r.total_steps ? `/${r.total_steps}` : ""}` : "—"}
              </span>
              <span className="w-24 text-right tabular-nums" title="Training loss / validation loss">
                {fixed(r.loss)}
                {r.val_loss != null && <span className="text-muted"> / {fixed(r.val_loss)}</span>}
              </span>
              <span className="w-20 text-right text-muted">{relativeDate(r.created)}</span>
            </div>
            {open === r.id && <RunDetail run={r} onAct={act} onChat={() => go("chat")} />}
          </li>
        ))}
      </ul>
    </Section>
  );
}

function RunDetail({ run, onAct, onChat }: { run: Run; onAct: (call: Promise<unknown>, after?: () => void) => Promise<void>; onChat: () => void }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const { engines } = useEngines();
  const { refreshRuntime } = useApp();
  useEffect(() => {
    api<Detail>(`/api/train/runs/${run.id}`).then(setDetail, () => setDetail(null));
  }, [run.id, run.steps]);

  const engine = run.backend === "mlx" ? engines?.find((e) => e.id === "mlx" && e.available) : engines?.find((e) => (e.id === "vllm" || e.id === "sglang") && e.available);
  const prov = detail?.provenance ?? {};
  const versions = Object.entries((prov.libraries as Record<string, string>) ?? {});

  async function chat(step: number | null) {
    const adapter = step == null ? run.adapter : (await api<{ adapter: string }>(`/api/train/runs/${run.id}/checkpoints/${step}/adapter`, { method: "POST" })).adapter;
    await api("/api/runtime/load", { json: { path: run.base_model, engine: engine!.id, adapter } });
    await refreshRuntime();
    onChat();
  }

  return (
    <div className="space-y-3 border-t border-line bg-surface-2/40 px-9 py-3 text-[12.5px]">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
        <dt className="text-muted">Base model</dt>
        <dd className="break-all">
          {run.base_model}
          {typeof prov.base_revision === "string" && <span className="text-muted"> @ {prov.base_revision.slice(0, 10)}</span>}
        </dd>
        <dt className="text-muted">Dataset</dt>
        <dd className="break-all">
          {run.dataset}
          {run.dataset_sha256 && <span className="text-muted" title={run.dataset_sha256}> · sha256 {run.dataset_sha256.slice(0, 12)}</span>}
        </dd>
        <dt className="text-muted">Method</dt>
        <dd>
          {run.method} on {run.backend === "mlx" ? "MLX" : "GPU"}
        </dd>
        {versions.length > 0 && (
          <>
            <dt className="text-muted">Libraries</dt>
            <dd className="text-muted">{versions.map(([k, v]) => `${k} ${v}`).join(", ")}</dd>
          </>
        )}
      </dl>
      {detail?.best && (
        <p>
          {detail.best.stopped_at ? `Stopped early at step ${detail.best.stopped_at}. ` : ""}Kept the adapter from step {detail.best.step}, validation loss {detail.best.val_loss.toFixed(3)}.
        </p>
      )}
      {detail?.error && <Notice tone="error">{detail.error}</Notice>}
      {detail && detail.samples.length > 0 && <SampleReplies samples={detail.samples} />}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="ghost" onClick={() => void onAct(api(`/api/train/runs/${run.id}/rerun`, { method: "POST" }))} disabled={run.status === "running"}>
          <RotateCcw className="size-3.5" /> Run again
        </Button>
        {run.adapter && run.base_model && engine && (
          <Button size="sm" variant="ghost" onClick={() => void onAct(chat(null))}>
            <MessageSquare className="size-3.5" /> Chat with the final adapter
          </Button>
        )}
      </div>
      {detail && detail.checkpoint_list.length > 0 && (
        <div>
          <p className="mb-1 text-muted">Checkpoints</p>
          <ul className="space-y-1">
            {detail.checkpoint_list.map((c) => (
              <li key={c.step} className="flex items-center gap-3">
                <span className="w-20 tabular-nums">step {c.step}</span>
                <span className="w-16 text-muted">{bytes(c.bytes)}</span>
                {engine && run.base_model && (
                  <button className="flex items-center gap-1 text-muted hover:text-text" onClick={() => void onAct(chat(c.step))}>
                    <MessageSquare className="size-3" /> Chat with it
                  </button>
                )}
                {run.status !== "running" && (
                  <button className="flex items-center gap-1 text-muted hover:text-text" onClick={() => void onAct(api(`/api/train/runs/${run.id}/resume`, { json: { step: c.step } }))}>
                    <StepForward className="size-3" /> Carry on from here
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** The loss curves of the ticked runs, on one chart. */
function Compare({ ids, runs }: { ids: string[]; runs: Run[] }) {
  const [curves, setCurves] = useState<Record<string, { step: number; loss: number }[]>>({});
  useEffect(() => {
    for (const id of ids) {
      if (curves[id]) continue;
      api<Detail>(`/api/train/runs/${id}`).then(
        (d) => setCurves((c) => ({ ...c, [id]: d.metrics.filter((m): m is { step: number; loss: number } => m.loss != null) })),
        () => setCurves((c) => ({ ...c, [id]: [] })),
      );
    }
  }, [ids, curves]);

  const W = 560;
  const H = 200;
  const pad = { l: 40, r: 12, t: 12, b: 24 };
  const lines = ids.map((id, i) => ({ id, color: COLORS[i % COLORS.length], points: curves[id] ?? [], run: runs.find((r) => r.id === id) }));
  const all = lines.flatMap((l) => l.points);
  if (all.length < 2) return <div className="mb-3 flex h-[200px] items-center justify-center rounded-lg border border-dashed border-line text-[13px] text-muted">These runs have no loss recorded yet.</div>;
  const maxStep = Math.max(...all.map((p) => p.step)) || 1;
  const lo = Math.min(...all.map((p) => p.loss));
  const hi = Math.max(...all.map((p) => p.loss));
  const span = hi - lo || 1;
  const x = (s: number) => pad.l + (s / maxStep) * (W - pad.l - pad.r);
  const y = (l: number) => pad.t + (1 - (l - lo) / span) * (H - pad.t - pad.b);

  return (
    <div className="mb-3 rounded-lg border border-line p-3">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Loss curves of the compared runs">
        {[hi, lo + span / 2, lo].map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeDasharray="2 4" />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" fontSize="10.5" fill="var(--muted)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize="10.5" fill="var(--muted)">
          step {maxStep}
        </text>
        {lines.map((l) => l.points.length > 1 && <path key={l.id} d={l.points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join("")} fill="none" stroke={l.color} strokeWidth="2" strokeLinejoin="round" />)}
      </svg>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px]">
        {lines.map((l) => (
          <li key={l.id} className={cx("flex items-center gap-1.5", !l.points.length && "text-muted")}>
            <span className="inline-block h-0.5 w-4 rounded" style={{ background: l.color }} />
            {l.run?.name || l.id} <span className="text-muted">{fixed(l.points.at(-1)?.loss)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
