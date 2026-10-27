import { ArrowLeft, BookText, Pencil, Plus, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Button, Input, cx } from "./ui";

export interface Prompt {
  id: string;
  name: string;
  text: string;
  /** The {{names}} to fill in when it's used. */
  variables: string[];
}

type View = { kind: "list" } | { kind: "edit"; prompt: Prompt | null; text: string } | { kind: "fill"; prompt: Prompt; values: Record<string, string> };

const fill = (text: string, values: Record<string, string>) => text.replace(/\{\{\s*([^{}]+?)\s*\}\}/g, (hole, name: string) => values[name] ?? hole);

/** The prompt library (engine prompts.py), opened from the composer or by typing / in an empty message.
 * `query` is what follows the /; `onUse` gets the prompt's text with its variables filled in. */
export function PromptLibrary({ open, onOpenChange, query, draft, onUse }: { open: boolean; onOpenChange: (open: boolean) => void; query: string | null; draft: string; onUse: (text: string) => void }) {
  const t = useT();
  const [prompts, setPrompts] = useState<Prompt[] | null>(null);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<View>({ kind: "list" });
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const q = query ?? search;
  // The parent passes a new function each render; reading it through a ref keeps the effects below from rerunning.
  const onOpenChangeRef = useRef(onOpenChange);
  onOpenChangeRef.current = onOpenChange;

  useEffect(() => {
    if (!open) return;
    setView({ kind: "list" });
    setError(null);
    setActive(0);
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && onOpenChangeRef.current(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const timer = setTimeout(() => {
      api<{ prompts: Prompt[] }>(`/api/prompts?q=${encodeURIComponent(q)}`).then((r) => setPrompts(r.prompts), () => setPrompts([]));
    }, 120);
    return () => clearTimeout(timer);
  }, [open, q]);

  useEffect(() => setActive(0), [q]);

  // The composer's / search steers the list with the arrow keys and Enter.
  useEffect(() => {
    if (!open || query === null || view.kind !== "list") return;
    const onKey = (e: KeyboardEvent) => {
      const list = prompts ?? [];
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        setActive((a) => (list.length ? (a + (e.key === "ArrowDown" ? 1 : -1) + list.length) % list.length : 0));
      } else if (e.key === "Enter" && list[active]) {
        e.preventDefault();
        e.stopPropagation();
        choose(list[active]);
      } else if (e.key === "Escape") {
        e.preventDefault();
        onOpenChange(false);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  function choose(prompt: Prompt) {
    if (prompt.variables.length) return setView({ kind: "fill", prompt, values: {} });
    use(prompt, prompt.text);
  }

  function use(prompt: Prompt, text: string) {
    void api(`/api/prompts/${prompt.id}/used`, { method: "POST" }).catch(() => {});
    onUse(text);
    onOpenChange(false);
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (view.kind !== "edit") return;
    setError(null);
    try {
      const body = { json: { name, text: view.text } };
      await (view.prompt ? api(`/api/prompts/${view.prompt.id}`, { method: "PUT", ...body }) : api("/api/prompts", body));
      setView({ kind: "list" });
      setPrompts((await api<{ prompts: Prompt[] }>(`/api/prompts?q=${encodeURIComponent(q)}`)).prompts);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function remove(prompt: Prompt) {
    if (!confirm(t("prompts.delete_confirm", { name: prompt.name }))) return;
    await api(`/api/prompts/${prompt.id}`, { method: "DELETE" }).catch(() => {});
    setPrompts((ps) => ps?.filter((p) => p.id !== prompt.id) ?? null);
  }

  return (
    <div ref={ref} className="relative">
      <Button variant="ghost" onClick={() => onOpenChange(!open)} aria-label={t("prompts.button")} title={t("prompts.button")} aria-expanded={open} className="size-9 rounded-xl px-0 text-muted">
        <BookText className="size-[18px]" />
      </Button>
      {open && (
        <div className="absolute bottom-11 left-0 z-30 flex max-h-[420px] w-[380px] flex-col rounded-xl border border-line bg-surface text-[13px] shadow-lg">
          {view.kind === "list" && (
            <>
              <div className="border-b border-line p-2">
                {query === null ? (
                  <Input autoFocus value={search} onChange={(e) => setSearch(e.target.value)} placeholder={t("prompts.search")} aria-label={t("prompts.search")} className="h-8 text-[13px]" />
                ) : (
                  <p className="px-1 text-[12px] text-muted">{t("prompts.hint")}</p>
                )}
              </div>
              <ul className="min-h-0 flex-1 overflow-y-auto p-1" role="listbox" aria-label={t("prompts.button")}>
                {prompts?.map((p, i) => (
                  <li key={p.id} role="option" aria-selected={query !== null && i === active} className={cx("group flex items-start gap-2 rounded-lg px-2 py-1.5", query !== null && i === active ? "bg-surface-2" : "hover:bg-surface-2")}>
                    <button className="min-w-0 flex-1 text-left" onClick={() => choose(p)}>
                      <span className="block truncate font-medium">{p.name}</span>
                      <span className="block truncate text-[12px] text-muted">{p.text.split("\n")[0]}</span>
                    </button>
                    <span className="flex shrink-0 gap-1 pt-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100">
                      <button
                        className="text-muted hover:text-text"
                        onClick={() => {
                          setName(p.name);
                          setView({ kind: "edit", prompt: p, text: p.text });
                        }}
                        title={t("prompts.edit")}
                      >
                        <Pencil className="size-3.5" aria-label={t("prompts.edit")} />
                      </button>
                      <button className="text-muted hover:text-danger" onClick={() => void remove(p)} title={t("prompts.delete")}>
                        <Trash2 className="size-3.5" aria-label={t("prompts.delete")} />
                      </button>
                    </span>
                  </li>
                ))}
                {prompts?.length === 0 && <li className="px-2 py-3 text-[12.5px] text-muted">{q ? t("prompts.none_match") : t("prompts.empty")}</li>}
              </ul>
              <div className="border-t border-line p-2">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!draft.trim() || query !== null}
                  onClick={() => {
                    setName("");
                    setView({ kind: "edit", prompt: null, text: draft });
                  }}
                >
                  <Plus className="size-3.5" /> {t("prompts.save_current")}
                </Button>
              </div>
            </>
          )}
          {view.kind === "edit" && (
            <form onSubmit={(e) => void save(e)} className="flex flex-col gap-2 p-3">
              <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder={t("prompts.name")} aria-label={t("prompts.name")} className="h-8 text-[13px]" maxLength={120} />
              <textarea
                value={view.text}
                onChange={(e) => setView({ ...view, text: e.target.value })}
                aria-label={t("prompts.text")}
                rows={7}
                className="resize-y rounded-md border border-line bg-bg px-2.5 py-2 text-[13px] focus:border-accent focus:outline-none"
              />
              <p className="text-[12px] text-muted">{t("prompts.variables_hint")}</p>
              {error && <p className="text-[12px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <Button size="sm" variant="primary" type="submit" disabled={!name.trim() || !view.text.trim()}>
                  {t("prompts.save")}
                </Button>
                <Button size="sm" variant="ghost" type="button" onClick={() => setView({ kind: "list" })}>
                  {t("prompts.cancel")}
                </Button>
              </div>
            </form>
          )}
          {view.kind === "fill" && (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                use(view.prompt, fill(view.prompt.text, view.values));
              }}
              className="flex flex-col gap-2 p-3"
            >
              <button type="button" className="flex items-center gap-1 self-start text-[12px] text-muted hover:text-text" onClick={() => setView({ kind: "list" })}>
                <ArrowLeft className="size-3.5" /> {t("prompts.back")}
              </button>
              <p className="font-medium">{view.prompt.name}</p>
              {view.prompt.variables.map((v, i) => (
                <label key={v} className="flex flex-col gap-1">
                  <span className="text-[12px] text-muted">{t("prompts.fill", { name: v })}</span>
                  <textarea
                    autoFocus={i === 0}
                    rows={v.length > 12 || /code|text|content/i.test(v) ? 4 : 1}
                    value={view.values[v] ?? ""}
                    onChange={(e) => setView({ ...view, values: { ...view.values, [v]: e.target.value } })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) e.currentTarget.form?.requestSubmit();
                    }}
                    className="resize-y rounded-md border border-line bg-bg px-2.5 py-1.5 text-[13px] focus:border-accent focus:outline-none"
                  />
                </label>
              ))}
              <Button size="sm" variant="primary" type="submit" className="self-start">
                {t("prompts.use")}
              </Button>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
