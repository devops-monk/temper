import { FolderPlus, Library, RotateCw, Search, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { useJob } from "../lib/useJob";
import { pickFolder } from "./Skills";
import { Badge, Button, Field, Input, ProgressBar, Section, Select, Toggle } from "./ui";

/** A knowledge base as the engine lists it (engine knowledge.py). */
interface Base {
  name: string;
  title: string;
  paths: string[];
  embedder: string;
  enabled: boolean;
  files: number;
  passages: number;
  skipped: number;
  indexed: number | null;
  job: string | null;
}

interface Embedder {
  id: string;
  name: string;
  megabytes: number;
  languages: "many" | "en";
  downloaded: boolean;
}

interface KnowledgeInfo {
  bases: Base[];
  embedders: Embedder[];
  embedder: string;
  llama_cpp: boolean;
}

interface Passage {
  id: number;
  heading: string;
  text: string;
  path: string;
}

/** Settings: folders of documents chat models can search, and the embedding model that indexes them. */
export function KnowledgeBases() {
  const t = useT();
  const { jobs, refreshJobs } = useApp();
  const [info, setInfo] = useState<KnowledgeInfo | null>(null);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(() => void api<KnowledgeInfo>("/api/knowledge").then(setInfo, () => {}), []);
  useEffect(refresh, [refresh]);
  const download = jobs.find((j) => j.kind === "download" && j.status === "running" && j.title.startsWith("Embedding model"))?.id ?? null;
  const { status, events } = useJob(download);
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  useEffect(() => {
    if (status && status !== "running") refresh();
  }, [status, refresh]);

  if (!info) return null;
  const current = info.embedders.find((e) => e.id === info.embedder);

  async function choose(id: string) {
    setError(null);
    await api<{ job: string | null }>("/api/knowledge/embedder", { json: { id } }).then(() => (refresh(), void refreshJobs()), (e: Error) => setError(e.message));
  }

  return (
    <Section
      title={t("kb.title")}
      aside={
        !adding && (
          <Button size="sm" onClick={() => setAdding(true)} disabled={!current?.downloaded || !info.llama_cpp}>
            <FolderPlus className="size-3.5" /> {t("kb.add")}
          </Button>
        )
      }
    >
      <p className="mb-3 text-[12.5px] text-muted">{t("kb.about")}</p>
      <div className="mb-3 flex flex-col gap-1.5 text-[13px]">
        <Field label={t("kb.embedder")} hint={t("kb.embedder_hint")}>
          <div className="flex items-center gap-2">
            <Select value={info.embedder} onChange={(e) => void choose(e.target.value)} disabled={!!download}>
              {info.embedders.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name} · {t(e.languages === "many" ? "kb.many_languages" : "kb.english")} · {e.megabytes} MB{e.downloaded ? "" : ` · ${t("kb.not_downloaded")}`}
                </option>
              ))}
            </Select>
            {current && !current.downloaded && !download && (
              <Button size="sm" onClick={() => void choose(current.id)}>
                {t("kb.download")}
              </Button>
            )}
          </div>
        </Field>
        {download && <ProgressBar value={progress ?? null} />}
        {!info.llama_cpp && <p className="text-[12px] text-straw">{t("kb.needs_llama")}</p>}
      </div>
      {adding && <NewBase onDone={(made) => (setAdding(false), made && (refresh(), void refreshJobs()))} />}
      {info.bases.length === 0 && !adding && <p className="text-[13px] text-muted">{t("kb.empty")}</p>}
      <ul className="flex flex-col gap-2">
        {info.bases.map((b) => (
          <BaseRow key={b.name} base={b} onChange={refresh} />
        ))}
      </ul>
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}
    </Section>
  );
}

function NewBase({ onDone }: { onDone: (made: boolean) => void }) {
  const t = useT();
  const [name, setName] = useState("");
  const [paths, setPaths] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  async function add() {
    const folder = await pickFolder(t("kb.pick"));
    if (folder) setPaths((p) => [...new Set([...p, folder])]);
  }
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    await api("/api/knowledge", { json: { name, paths } }).then(() => onDone(true), (err: Error) => setError(err.message));
  }
  return (
    <form onSubmit={(e) => void save(e)} className="mb-3 flex flex-col gap-3 rounded-lg border border-line bg-surface p-3 text-[13px]">
      <Field label={t("kb.name")}>
        <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder={t("kb.name_hint")} maxLength={48} />
      </Field>
      <div className="flex flex-col gap-1.5">
        <span className="text-[12.5px] font-medium">{t("kb.folders")}</span>
        {paths.map((p) => (
          <span key={p} className="flex items-center gap-2 font-mono text-[12px]">
            <span className="min-w-0 flex-1 truncate">{p}</span>
            <button type="button" className="text-muted hover:text-text" onClick={() => setPaths(paths.filter((x) => x !== p))} aria-label={t("kb.remove_folder")}>
              ×
            </button>
          </span>
        ))}
        <Button size="sm" variant="ghost" type="button" className="self-start" onClick={() => void add()}>
          <FolderPlus className="size-3.5" /> {t("kb.add_folder")}
        </Button>
      </div>
      {error && <p className="text-[12px] text-danger">{error}</p>}
      <div className="flex gap-2">
        <Button size="sm" variant="primary" type="submit" disabled={!name.trim() || !paths.length}>
          {t("kb.create")}
        </Button>
        <Button size="sm" variant="ghost" type="button" onClick={() => onDone(false)}>
          {t("prompts.cancel")}
        </Button>
      </div>
    </form>
  );
}

function BaseRow({ base: b, onChange }: { base: Base; onChange: () => void }) {
  const t = useT();
  const { refreshJobs } = useApp();
  const { status, events } = useJob(b.job);
  const [query, setQuery] = useState("");
  const [found, setFound] = useState<Passage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const indexing = !!b.job && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  useEffect(() => {
    if (status && status !== "running") onChange();
  }, [status, onChange]);
  const path = `/api/knowledge/${b.name}`;

  async function search(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    await api<{ results: Passage[] }>(`${path}/search?q=${encodeURIComponent(query)}`).then((r) => setFound(r.results), (err: Error) => setError(err.message));
  }

  return (
    <li className="rounded-lg border border-line bg-surface px-3 py-2 text-[13px]">
      <div className="flex items-center gap-2">
        <Library className="size-3.5 shrink-0 text-muted" />
        <span className="font-medium">{b.title}</span>
        {indexing ? <Badge tone="accent">{t("kb.indexing")}</Badge> : <Badge>{t("kb.stats", { files: b.files, passages: b.passages })}</Badge>}
        <span className="flex-1" />
        <Toggle label={t("kb.use")} checked={b.enabled} onChange={(enabled) => void api(path, { method: "PUT", json: { enabled } }).then(onChange)} />
        <Button size="sm" variant="ghost" disabled={indexing} onClick={() => void api(`${path}/index`, { method: "POST" }).then(() => (onChange(), void refreshJobs()), (e: Error) => setError(e.message))} title={t("kb.reindex")} aria-label={t("kb.reindex")}>
          <RotateCw className="size-3.5" />
        </Button>
        <Button size="sm" variant="ghost" onClick={() => confirm(t("kb.delete_confirm", { name: b.title })) && void api(path, { method: "DELETE" }).then(onChange)} title={t("kb.delete")} aria-label={t("kb.delete")}>
          <Trash2 className="size-3.5" />
        </Button>
      </div>
      <p className="mt-1 truncate font-mono text-[11.5px] text-muted" title={b.paths.join("\n")}>
        {b.paths.join(" · ")}
      </p>
      {b.skipped > 0 && !indexing && <p className="mt-0.5 text-[12px] text-muted">{t("kb.skipped", { count: b.skipped })}</p>}
      {indexing && (
        <div className="mt-2">
          <ProgressBar value={progress ?? null} />
        </div>
      )}
      {!indexing && b.passages > 0 && (
        <form onSubmit={(e) => void search(e)} className="mt-2 flex gap-2">
          <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("kb.try")} aria-label={t("kb.try")} />
          <Button size="sm" type="submit" disabled={!query.trim()} aria-label={t("kb.search")}>
            <Search className="size-3.5" />
          </Button>
        </form>
      )}
      {error && <p className="mt-1 text-[12px] text-danger">{error}</p>}
      {found && (
        <ol className="mt-2 flex flex-col gap-1.5">
          {found.length === 0 && <li className="text-[12px] text-muted">{t("kb.no_results")}</li>}
          {found.map((p) => (
            <li key={p.id} className="rounded-md bg-bg px-2.5 py-1.5">
              <span className="block truncate text-[11.5px] text-muted" title={p.path}>
                <span className="font-medium text-text">{p.path.split(/[\\/]/).pop()}</span>
                {p.heading && ` · ${p.heading}`}
              </span>
              <span className="line-clamp-3 text-[12.5px]">{p.text}</span>
            </li>
          ))}
        </ol>
      )}
    </li>
  );
}
