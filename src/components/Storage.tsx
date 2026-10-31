import { FolderOpen } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, inTauri } from "../lib/engine";
import { bytes } from "../lib/format";
import { useT, withNodes } from "../lib/i18n";
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
  ["models", "storage.models", "bg-accent"],
  ["exports", "storage.exports", "bg-ok"],
  ["runs", "storage.runs", "bg-straw"],
  ["datasets", "storage.datasets", "bg-text/40"],
  ["engine", "storage.engine", "bg-muted/50"],
] as const;

/** Settings: what Temper keeps on disk, and where models are stored. */
export function Storage({ ready }: { ready: boolean }) {
  const t = useT();
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
    const folder = await open({ directory: true, title: t("storage.choose") });
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
    <Section title={t("storage.title")}>
      <div className="rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
        <div className="flex items-baseline justify-between gap-3">
          <span className="font-medium">{t("storage.uses", { size: bytes(used) })}</span>
          <span className="text-muted tabular-nums">{t("storage.free", { free: bytes(info.disk.free), total: bytes(info.disk.total) })}</span>
        </div>
        <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={t("storage.bar", { size: bytes(used), free: bytes(info.disk.free) })}>
          {PARTS.map(([key, label, color]) => (
            <div key={key} className={color} style={{ width: `${(info.usage[key] / scale) * 100}%` }} title={`${t(label)}: ${bytes(info.usage[key])}`} />
          ))}
        </div>
        <dl className="mt-3 grid grid-cols-[auto_1fr_auto] items-center gap-x-2.5 gap-y-1 text-[12.5px]">
          {PARTS.map(([key, label, color]) => (
            <div key={key} className="contents">
              <span className={cx("size-2 rounded-full", color)} />
              <dt className="text-muted">{t(label)}</dt>
              <dd className="text-right tabular-nums">{bytes(info.usage[key])}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="mt-3 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="font-medium">{t("storage.folder")}</p>
            <p className="truncate font-mono text-[11.5px] text-muted" title={info.models_dir}>
              {info.models_dir}
            </p>
          </div>
          {inTauri && (
            <Button size="sm" disabled={moving} onClick={() => void choose()}>
              <FolderOpen className="size-3.5" /> {t("storage.change")}
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
            <Input name="path" placeholder="/Volumes/External/models" spellCheck={false} className="flex-1" aria-label={t("storage.new_folder")} />
            <Button type="submit">{t("storage.change")}</Button>
          </form>
        )}
        {target && !moving && (
          <div className="mt-3 border-t border-line pt-3">
            <p>
              {withNodes(t("storage.keep_in"), { path: <span className="font-mono text-[12px]">{target}</span> })}
            </p>
            <label className="mt-2 flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={move} onChange={(e) => setMove(e.target.checked)} className="accent-[var(--accent)]" />
              {t("storage.move_mine", { size: bytes(info.usage.models) })}
            </label>
            {!move && <p className="mt-1 text-[12px] text-muted">{t("storage.stay")}</p>}
            <div className="mt-3 flex gap-2">
              <Button size="sm" variant="primary" onClick={() => void apply(target, move)}>
                {t(move ? "storage.move" : "storage.use")}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setTarget(null)}>
                {t("storage.cancel")}
              </Button>
            </div>
          </div>
        )}
        {moving && (
          <div className="mt-3">
            <p className="mb-1.5 text-muted">{t("storage.moving")}</p>
            <ProgressBar value={progress ?? null} />
          </div>
        )}
        {info.custom && !target && !moving && (
          <button className="mt-2 text-[12px] text-accent hover:underline" onClick={() => setTarget(info.default_dir)}>
            {t("storage.back")}
          </button>
        )}
        <p className="mt-2 text-[12px] text-muted">{t("storage.shared")}</p>
      </div>
      {(error || status === "failed") && (
        <div className="mt-3">
          <Notice tone="error">{error ?? t("storage.move_failed", { error: jobError ?? t("storage.unknown") })}</Notice>
        </div>
      )}
    </Section>
  );
}
