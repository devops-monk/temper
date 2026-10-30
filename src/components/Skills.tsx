import { FolderInput, Plus, Sparkles, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api, inTauri } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Badge, Button, Field, Input, Section, cx } from "./ui";

/** A skill as the engine lists it (engine skills.py). */
export interface Skill {
  name: string;
  description: string;
  path: string;
  source: "global" | "project";
}

const fetchSkills = (project: string | null) => api<{ skills: Skill[]; folder: string }>(`/api/skills${project ? `?project=${encodeURIComponent(project)}` : ""}`);

/** The @mention being typed at the end of the message, if any. */
export const mentionAt = (draft: string) => /(?:^|\s)@([a-z0-9-]*)$/i.exec(draft)?.[1]?.toLowerCase() ?? null;

/** The skills matching what follows an @ at the end of the message; arrow keys and Enter or Tab pick one. */
export function SkillPicker({ query, project, onPick, onDismiss }: { query: string; project: string | null; onPick: (name: string) => void; onDismiss: () => void }) {
  const t = useT();
  const [skills, setSkills] = useState<Skill[] | null>(null);
  const [active, setActive] = useState(0);
  useEffect(() => void fetchSkills(project).then((r) => setSkills(r.skills), () => setSkills([])), [project]);
  const matches = (skills ?? []).filter((s) => s.name.includes(query) || s.description.toLowerCase().includes(query)).slice(0, 8);
  useEffect(() => setActive(0), [query]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") return (e.preventDefault(), onDismiss());
      if (!matches.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        setActive((a) => (a + (e.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length);
      } else if ((e.key === "Enter" || e.key === "Tab") && !e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        onPick(matches[active].name);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  if (!skills) return null;
  return (
    <div className="absolute bottom-full left-0 z-30 mb-2 w-[380px] rounded-xl border border-line bg-surface p-1 text-[13px] shadow-lg" role="listbox" aria-label={t("skills.title")}>
      {matches.map((s, i) => (
        <button
          key={s.name}
          role="option"
          aria-selected={i === active}
          onMouseDown={(e) => (e.preventDefault(), onPick(s.name))}
          className={cx("flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left", i === active ? "bg-surface-2" : "hover:bg-surface-2")}
        >
          <Sparkles className="mt-0.5 size-3.5 shrink-0 text-muted" />
          <span className="min-w-0">
            <span className="block font-medium">
              @{s.name} {s.source === "project" && <Badge>{t("skills.project")}</Badge>}
            </span>
            {s.description && <span className="block truncate text-[12px] text-muted">{s.description}</span>}
          </span>
        </button>
      ))}
      {!matches.length && <p className="px-2 py-2 text-[12.5px] text-muted">{skills.length ? t("skills.none_match") : t("skills.empty_picker")}</p>}
    </div>
  );
}

async function pickFolder(title: string): Promise<string | null> {
  if (inTauri) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({ directory: true, title });
    return typeof picked === "string" ? picked : null;
  }
  return prompt(title)?.trim() || null;
}

/** Settings: Temper's own skills, made here or copied in from a folder. */
export function SkillsSettings() {
  const t = useT();
  const [skills, setSkills] = useState<Skill[] | null>(null);
  const [folder, setFolder] = useState("");
  const [making, setMaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refresh = () => void fetchSkills(null).then((r) => (setSkills(r.skills), setFolder(r.folder)), () => {});
  useEffect(refresh, []);

  async function importFolder() {
    const path = await pickFolder(t("skills.import"));
    if (!path) return;
    setError(null);
    await api("/api/skills/import", { json: { path } }).then(refresh, (e: Error) => setError(e.message));
  }

  async function remove(name: string) {
    if (!confirm(t("skills.remove_confirm", { name }))) return;
    await api<{ skills: Skill[] }>(`/api/skills/${name}`, { method: "DELETE" }).then((r) => setSkills(r.skills), (e: Error) => setError(e.message));
  }

  return (
    <Section
      title={t("skills.title")}
      aside={
        !making && (
          <span className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => void importFolder()}>
              <FolderInput className="size-3.5" /> {t("skills.import")}
            </Button>
            <Button size="sm" onClick={() => setMaking(true)}>
              <Plus className="size-3.5" /> {t("skills.new")}
            </Button>
          </span>
        )
      }
    >
      <p className="mb-3 text-[12.5px] text-muted">{t("skills.about")}</p>
      {making && (
        <NewSkill
          onDone={(made) => {
            setMaking(false);
            if (made) refresh();
          }}
        />
      )}
      {skills?.length === 0 && !making && <p className="text-[13px] text-muted">{t("skills.empty")}</p>}
      <ul className="flex flex-col divide-y divide-line rounded-lg border border-line bg-surface empty:hidden">
        {skills?.map((s) => (
          <li key={s.name} className="flex items-start gap-3 px-3 py-2 text-[13px]">
            <span className="min-w-0 flex-1">
              <span className="block font-medium">@{s.name}</span>
              <span className="block text-[12px] text-muted">{s.description || t("skills.no_description")}</span>
            </span>
            <Button size="sm" variant="ghost" onClick={() => void remove(s.name)} aria-label={t("skills.remove")} title={t("skills.remove")}>
              <Trash2 className="size-3.5" />
            </Button>
          </li>
        ))}
      </ul>
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}
      {folder && <p className="mt-2 text-[12px] text-muted">{t("skills.folder", { folder })}</p>}
    </Section>
  );
}

function NewSkill({ onDone }: { onDone: (made: boolean) => void }) {
  const t = useT();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [instructions, setInstructions] = useState("");
  const [error, setError] = useState<string | null>(null);
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    await api("/api/skills", { json: { name, description, instructions } }).then(() => onDone(true), (err: Error) => setError(err.message));
  }
  return (
    <form onSubmit={(e) => void save(e)} className="mb-3 flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
      <Field label={t("skills.name")} hint={t("skills.name_hint")}>
        <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="write-tests" maxLength={64} />
      </Field>
      <Field label={t("skills.description")}>
        <Input value={description} onChange={(e) => setDescription(e.target.value)} placeholder={t("skills.description_hint")} maxLength={1024} />
      </Field>
      <Field label={t("skills.instructions")}>
        <textarea value={instructions} onChange={(e) => setInstructions(e.target.value)} rows={7} className="w-full resize-y rounded-md border border-line bg-bg px-3 py-2 text-[13px] focus:border-accent focus:outline-none" />
      </Field>
      {error && <p className="text-[12px] text-danger">{error}</p>}
      <div className="flex gap-2">
        <Button size="sm" variant="primary" type="submit" disabled={!name.trim() || !instructions.trim()}>
          {t("skills.save")}
        </Button>
        <Button size="sm" variant="ghost" type="button" onClick={() => onDone(false)}>
          {t("prompts.cancel")}
        </Button>
      </div>
    </form>
  );
}
