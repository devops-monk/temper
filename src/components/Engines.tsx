import { Download, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { IdleSelect } from "./LoadedModels";
import { useEngines } from "./LoadButton";
import { Badge, Button, ProgressBar, Section } from "./ui";

const ABOUT: Record<string, Key> = {
  "llama.cpp": "eng.about_llamacpp",
  mlx: "eng.about_mlx",
  transformers: "eng.about_transformers",
  vllm: "eng.about_gpu",
  sglang: "eng.about_gpu",
};

/** The inference engines, with installs for the opt-in ones (engine engines.py). Each installs into its own environment. */
export function Engines() {
  const t = useT();
  const { engines, refresh } = useEngines();
  const { jobs, refreshJobs } = useApp();
  const [started, setStarted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const existing = jobs.find((j) => j.kind === "engine-install" && j.status === "running");
  const jobId = started ?? existing?.id ?? null;
  const { status, events } = useJob(jobId);
  const installing = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const lastLine = [...events].reverse().find((e) => e.event === "log")?.line as string | undefined;
  const failure = events.find((e) => e.event === "status" && e.status === "failed")?.error as string | undefined;

  useEffect(() => {
    if (status && status !== "running") {
      refresh();
      void refreshJobs();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function install(id: string) {
    setError(null);
    try {
      const job = await api<Job>(`/api/runtime/engines/${id}/install`, { method: "POST" });
      setStarted(job.id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(id: string, name: string) {
    if (!confirm(t("eng.remove_confirm", { name }))) return;
    setError(null);
    try {
      await api(`/api/runtime/engines/${id}`, { method: "DELETE" });
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function prefer(id: string | null) {
    setError(null);
    try {
      await api("/api/runtime/engines/preferred", { method: "PUT", json: { engine: id } });
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!engines?.length) return null;
  // A choice only matters once two engines can run model folders.
  const folderEngines = engines.filter((e) => e.id !== "llama.cpp" && e.available);
  return (
    <Section title={t("eng.title")}>
      <ul className="divide-y divide-line rounded-lg border border-line bg-surface">
        {engines.map((e) => (
          <li key={e.id} className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">{e.name}</span>
                {e.available ? <Badge tone="ok">{t("eng.ready")}</Badge> : e.unsupported ? <Badge>{t("eng.unsupported")}</Badge> : null}
                {e.version && <span className="font-mono text-[11.5px] text-muted">{e.version}</span>}
                {e.preferred && folderEngines.length > 1 && <Badge>{t("eng.folders")}</Badge>}
              </div>
              <p className="text-[12px] text-muted">{e.unsupported ?? (ABOUT[e.id] ? t(ABOUT[e.id]) : null)}</p>
            </div>
            {folderEngines.length > 1 && e.available && e.id !== "llama.cpp" && (
              <Button size="sm" variant="ghost" onClick={() => void prefer(e.preferred ? null : e.id)}>
                {t(e.preferred ? "eng.by_format" : "eng.use_for_folders")}
              </Button>
            )}
            {e.installable && !e.available && (
              <Button size="sm" onClick={() => void install(e.id)} disabled={installing}>
                <Download className="size-3.5" /> {t("eng.install")}
              </Button>
            )}
            {e.installable && e.available && (
              <button className="text-muted hover:text-danger" onClick={() => void remove(e.id, e.name)} title={t("eng.remove", { name: e.name })}>
                <Trash2 className="size-3.5" aria-label={t("eng.remove", { name: e.name })} />
              </button>
            )}
          </li>
        ))}
      </ul>
      {installing && (
        <div className="mt-3 flex flex-col gap-1.5">
          <ProgressBar value={progress ?? null} />
          <p className="truncate font-mono text-[11.5px] text-muted" title={lastLine}>
            {lastLine ?? t("eng.starting")}
          </p>
          <span>
            <Button size="sm" variant="ghost" onClick={() => void api(`/api/jobs/${jobId}/cancel`, { method: "POST" })}>
              {t("eng.cancel")}
            </Button>
          </span>
        </div>
      )}
      {(error || (status === "failed" && failure)) && <p className="mt-2 text-[12px] text-danger">{error ?? failure}</p>}
      <p className="mt-2 text-[12px] text-muted">{t("eng.load_hint")}</p>
      <div className="mt-4 flex items-center justify-between gap-3 text-[13px]">
        <span>
          <span className="font-medium">{t("eng.unload")}</span>
          <span className="block text-[12px] text-muted">{t("eng.unload_hint")}</span>
        </span>
        <IdleSelect />
      </div>
    </Section>
  );
}
