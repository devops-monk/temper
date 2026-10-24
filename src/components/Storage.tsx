import { FolderOpen } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, inTauri } from "../lib/engine";
import { bytes } from "../lib/format";
import { useJob } from "../lib/useJob";
import { Button, Input, Notice, ProgressBar, Section, cx } from "./ui";

interface StorageInfo {
  models_dir: string;
  default_dir: string;
  custom: boolean;
  disk: { total: number; free: number };
  usage: Record<"models" | "exports" | "runs" | "datasets" | "engine", number>;
}

const PARTS = [
  ["models", "Models", "bg-accent"],
  ["exports", "Exports", "bg-ok"],
  ["runs", "Training runs", "bg-straw"],
  ["datasets", "Datasets", "bg-text/40"],
  ["engine", "Engine and tools", "bg-muted/50"],
] as const;

/** Settings: what Temper keeps on disk, and where models are stored. */
export function Storage({ ready }: { ready: boolean }) {
  const [info, setInfo] = useState<StorageInfo | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [move, setMove] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const moving = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const jobError = events.find((e) => e.event === "status")?.error as string | undefined;

  const load = useCallback((fresh = false) => {
    api<StorageInfo>(`/api/storage${fresh ? "?fresh=true" : ""}`).then(setInfo, (e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (ready) load();
  }, [ready, load]);
  useEffect(() => {
    if (status === "done" || status === "failed") {
      if (status === "done") setTarget(null);
      load(true);
    }
  }, [status, load]);

  async function choose() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const folder = await open({ directory: true, title: "Choose where to keep models" });
    if (typeof folder === "string") setTarget(folder);
  }

  async function apply(path: string, withMove: boolean) {
    setError(null);
    try {
      const r = await api<{ job: { id: string } | null }>("/api/storage/models-dir", { method: "PUT", json: { path, move: withMove } });
      if (r.job) setJobId(r.job.id);
      else {
        setTarget(null);
        load(true);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!info) return null;
  const used = Object.values(info.usage).reduce((a, b) => a + b, 0);
  const scale = info.disk.total;
  return (
    <Section title="Storage">
      <div className="rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
        <div className="flex items-baseline justify-between gap-3">
          <span className="font-medium">Temper uses {bytes(used)}</span>
          <span className="text-muted tabular-nums">{bytes(info.disk.free)} free of {bytes(info.disk.total)}</span>
        </div>
        <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={`Temper uses ${bytes(used)}; ${bytes(info.disk.free)} of the disk is free`}>
          {PARTS.map(([key, label, color]) => (
            <div key={key} className={color} style={{ width: `${(info.usage[key] / scale) * 100}%` }} title={`${label}: ${bytes(info.usage[key])}`} />
          ))}
        </div>
        <dl className="mt-3 grid grid-cols-[auto_1fr_auto] items-center gap-x-2.5 gap-y-1 text-[12.5px]">
          {PARTS.map(([key, label, color]) => (
            <div key={key} className="contents">
              <span className={cx("size-2 rounded-full", color)} />
              <dt className="text-muted">{label}</dt>
              <dd className="text-right tabular-nums">{bytes(info.usage[key])}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="mt-3 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="font-medium">Models folder</p>
            <p className="truncate font-mono text-[11.5px] text-muted" title={info.models_dir}>
              {info.models_dir}
            </p>
          </div>
          {inTauri && (
            <Button size="sm" disabled={moving} onClick={() => void choose()}>
              <FolderOpen className="size-3.5" /> Change
            </Button>
          )}
        </div>
        {!inTauri && !target && (
          <form
            className="mt-2 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              const path = new FormData(e.currentTarget).get("path") as string;
              if (path.trim()) setTarget(path.trim());
            }}
          >
            <Input name="path" placeholder="/Volumes/External/models" spellCheck={false} className="flex-1" aria-label="New models folder" />
            <Button type="submit">Change</Button>
          </form>
        )}
        {target && !moving && (
          <div className="mt-3 border-t border-line pt-3">
            <p>
              Keep models in <span className="font-mono text-[12px]">{target}</span>
            </p>
            <label className="mt-2 flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={move} onChange={(e) => setMove(e.target.checked)} className="accent-[var(--accent)]" />
              Move my models there ({bytes(info.usage.models)})
            </label>
            {!move && <p className="mt-1 text-[12px] text-muted">Models already downloaded stay where they are and won't show in Temper until you switch back.</p>}
            <div className="mt-3 flex gap-2">
              <Button size="sm" variant="primary" onClick={() => void apply(target, move)}>
                {move ? "Move models" : "Use this folder"}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setTarget(null)}>
                Cancel
              </Button>
            </div>
          </div>
        )}
        {moving && (
          <div className="mt-3">
            <p className="mb-1.5 text-muted">Moving models… Chat and downloads can wait until this finishes.</p>
            <ProgressBar value={progress ?? null} />
          </div>
        )}
        {info.custom && !target && !moving && (
          <button className="mt-2 text-[12px] text-accent hover:underline" onClick={() => setTarget(info.default_dir)}>
            Go back to the standard folder
          </button>
        )}
        <p className="mt-2 text-[12px] text-muted">The standard folder is shared with other Hugging Face tools. Another folder is used only by Temper.</p>
      </div>
      {(error || status === "failed") && (
        <div className="mt-3">
          <Notice tone="error">{error ?? `Moving the models stopped: ${jobError ?? "unknown error"}. Models moved so far are in the new folder; the rest are where they were.`}</Notice>
        </div>
      )}
    </Section>
  );
}
