import { ExternalLink, UploadCloud, X } from "lucide-react";
import { useState } from "react";
import { api } from "../lib/engine";
import { useJob } from "../lib/useJob";
import { Button, Input } from "./ui";

/** What to publish (engine publish.py): a run's adapter, an export folder, or a prepared dataset. */
export type Publishable = { run_id: string } | { export: string } | { dataset: string };

/** A repository name from a run, export or dataset name: dates dropped, Hub-safe characters only. */
function repoName(name: string): string {
  return (
    name
      .replace(/^\d{8}-\d{6}-/, "")
      .replace(/-\d{6}$/, "")
      .replace(/\.jsonl$/, "")
      .replace(/[^A-Za-z0-9._-]+/g, "-")
      .replace(/^[-.]+|[-.]+$/g, "") || "my-model"
  );
}

/** Uploads to Hugging Face, private unless unticked, with a card written from the run. An icon until opened. */
export function PublishToHub({ what, name }: { what: Publishable; name: string }) {
  const [open, setOpen] = useState(false);
  const [repo, setRepo] = useState(() => repoName(name));
  const [isPrivate, setPrivate] = useState(true);
  const [job, setJob] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(job);
  const url = (events.find((e) => e.event === "status")?.result as { url?: string } | undefined)?.url;

  async function publish() {
    setError(null);
    try {
      const body = "dataset" in what ? { path: what.dataset, repo, private: isPrivate } : { ...what, repo, private: isPrivate };
      const started = await api<{ id: string }>(`/api/publish/${"dataset" in what ? "dataset" : "model"}`, { json: body });
      setJob(started.id);
      setOpen(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (job && status !== "failed") {
    if (status !== "done") return <span className="text-[12.5px] text-muted">Publishing…</span>;
    return (
      <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12.5px] text-muted hover:text-text">
        Published <ExternalLink className="size-3" />
      </a>
    );
  }
  if (!open) {
    return (
      <span className="inline-flex items-center gap-2">
        {(error || status === "failed") && <span className="text-[12px] text-danger">{error ?? "Publishing failed; see the job's output."}</span>}
        <button className="text-muted hover:text-text" onClick={() => setOpen(true)} title="Publish to Hugging Face">
          <UploadCloud className="size-3.5" aria-label={`Publish ${name} to Hugging Face`} />
        </button>
      </span>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2 text-[12.5px]">
      <span className="w-44">
        <Input value={repo} onChange={(e) => setRepo(e.target.value)} className="h-7 text-[12.5px]" aria-label="Repository name on Hugging Face" spellCheck={false} />
      </span>
      <label className="inline-flex items-center gap-1 text-muted">
        <input type="checkbox" checked={isPrivate} onChange={(e) => setPrivate(e.target.checked)} /> Private
      </label>
      <Button size="sm" onClick={() => void publish()} disabled={!repo.trim()}>
        Publish
      </Button>
      <button className="text-muted hover:text-text" onClick={() => setOpen(false)} aria-label="Cancel">
        <X className="size-3.5" />
      </button>
      {error && <span className="text-[12px] text-danger">{error}</span>}
    </span>
  );
}
