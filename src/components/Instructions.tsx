import { FolderOpen, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, inTauri } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Button, Field, Section } from "./ui";

interface Gathered {
  global: string;
  parts: { source: "global" | "project"; path: string; bytes: number }[];
  truncated: boolean;
  max_bytes: number;
}

async function pickFolder(title: string): Promise<string | null> {
  if (inTauri) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({ directory: true, title });
    return typeof picked === "string" ? picked : null;
  }
  return prompt(title)?.trim() || null;
}

/** The chat's project folder, and which instruction files (engine instructions.py) apply to it. */
export function ProjectFolder({ project, onChange }: { project: string | null; onChange: (project: string | null) => void }) {
  const t = useT();
  const [found, setFound] = useState<Gathered | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    setError(null);
    api<Gathered>(`/api/instructions${project ? `?project=${encodeURIComponent(project)}` : ""}`).then(
      (g) => !stale && setFound(g),
      (e: Error) => !stale && (setFound(null), setError(e.message)),
    );
    return () => {
      stale = true;
    };
  }, [project]);

  const files = found?.parts.filter((p) => p.source === "project") ?? [];
  const list = [found?.parts.some((p) => p.source === "global") && t("instr.global_short"), files.length > 0 && t("instr.files", { n: files.length })].filter(Boolean).join(" + ");

  return (
    <Field label={t("instr.project")} hint={t("instr.project_hint")}>
      <div className="flex items-center gap-1.5">
        <Button size="sm" variant="secondary" className="min-w-0 flex-1 justify-start" onClick={() => void pickFolder(t("instr.folder_prompt")).then((p) => p && onChange(p))} title={project ?? t("instr.choose")}>
          <FolderOpen className="size-3.5 shrink-0" />
          <span className="truncate">{project ? project.split(/[\\/]/).filter(Boolean).pop() : t("instr.choose")}</span>
        </Button>
        {project && (
          <button className="text-muted hover:text-text" onClick={() => onChange(null)} title={t("instr.clear")}>
            <X className="size-3.5" aria-label={t("instr.clear")} />
          </button>
        )}
      </div>
      {error ? (
        <p className="mt-1 text-[12px] text-danger">{error}</p>
      ) : (
        list && (
          <p className="mt-1 text-[12px] text-muted" title={found?.parts.map((p) => p.path).join("\n")}>
            {t("instr.using", { list })}
            {found?.truncated && ` ${t("instr.truncated", { kb: Math.round((found.max_bytes ?? 0) / 1024) })}`}
          </p>
        )
      )}
    </Field>
  );
}

/** Settings: the global AGENTS.md every chat gets. */
export function GlobalInstructions() {
  const t = useT();
  const [text, setText] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Gathered>("/api/instructions").then(
      (g) => {
        setText(g.global);
        setSaved(g.global);
      },
      () => setText(""),
    );
  }, []);

  if (text === null) return null;
  return (
    <Section title={t("instr.title")}>
      <p className="mb-2 text-[12.5px] text-muted">{t("instr.global_hint")}</p>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        placeholder={t("instr.placeholder")}
        aria-label={t("instr.title")}
        className="w-full resize-y rounded-md border border-line bg-surface px-3 py-2 font-mono text-[12.5px] placeholder:text-muted focus:border-accent focus:outline-none"
      />
      <div className="mt-2 flex items-center gap-3">
        <Button
          size="sm"
          variant="primary"
          disabled={text === saved}
          onClick={async () => {
            setError(null);
            try {
              await api("/api/instructions/global", { method: "PUT", json: { text } });
              setSaved(text);
            } catch (e) {
              setError((e as Error).message);
            }
          }}
        >
          {t("instr.save")}
        </Button>
        {text === saved && saved && <span className="text-[12px] text-muted">{t("instr.saved")}</span>}
        {error && <span className="text-[12px] text-danger">{error}</span>}
      </div>
    </Section>
  );
}
