import { ChevronRight, Download, Folder, FolderPlus, History, MoreHorizontal, Pencil, Pin, PinOff, Search, SquarePen, Trash2, X } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { useT } from "../lib/i18n";
import { api, inTauri } from "../lib/engine";
import { relativeDate } from "../lib/format";
import { useJob } from "../lib/useJob";
import { Button, Input, ProgressBar, cx } from "./ui";

export interface ChatSummary {
  id: string;
  title: string;
  model: string | null;
  folder: string | null;
  pinned: boolean;
  updated: number;
  source: string | null;
  project: string | null;
  messages: number | null;
  /** The matching text, with the match marked [[like this]]. */
  snippet?: string;
}

interface ChatListData {
  chats: ChatSummary[];
  folders: string[];
}

/** Saves a chat export: a save dialog in the app, a download in a browser. */
async function exportChat(chat: ChatSummary, format: "md" | "json") {
  const { text, filename } = await api<{ text: string; filename: string }>(`/api/chats/${chat.id}/export?format=${format}`);
  if (inTauri) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ defaultPath: filename, filters: [{ name: format === "md" ? "Markdown" : "JSON", extensions: [format] }] });
    if (path) await api(`/api/chats/${chat.id}/export`, { json: { format, path } });
    return;
  }
  const url = URL.createObjectURL(new Blob([text], { type: format === "md" ? "text/markdown" : "application/json" }));
  Object.assign(document.createElement("a"), { href: url, download: filename }).click();
  URL.revokeObjectURL(url);
}

function Snippet({ text }: { text: string }) {
  return (
    <span className="line-clamp-2 text-[11.5px] text-muted">
      {text.split(/(\[\[.*?\]\])/g).map((part, i) =>
        part.startsWith("[[") ? (
          <mark key={i} className="rounded-sm bg-straw/30 text-text">
            {part.slice(2, -2)}
          </mark>
        ) : (
          part
        ),
      )}
    </span>
  );
}

/** The chat page's list of saved threads: search, pinned, folders, recent, and chats imported from other tools. */
export function ChatList({ activeId, version, onOpen, onNew, onDeleted }: { activeId: string | null; version: number; onOpen: (id: string) => void; onNew: () => void; onDeleted: (id: string) => void }) {
  const [data, setData] = useState<ChatListData | null>(null);
  const t = useT();
  const [query, setQuery] = useState("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [newFolder, setNewFolder] = useState(false);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api<ChatListData>(`/api/chats${query.trim() ? `?q=${encodeURIComponent(query.trim())}` : ""}`).then(setData, (e: Error) => setError(e.message));
  }, [query]);
  useEffect(() => {
    const id = setTimeout(load, query ? 200 : 0);
    return () => clearTimeout(id);
  }, [load, version, query]);

  const run = async (call: Promise<unknown>) => {
    setError(null);
    try {
      await call;
    } catch (e) {
      setError((e as Error).message);
    }
    load();
  };
  const patch = (id: string, json: Record<string, unknown>) => run(api(`/api/chats/${id}`, { method: "PATCH", json }));

  const item = (c: ChatSummary) =>
    renaming === c.id ? (
      <InlineName
        key={c.id}
        initial={c.title}
        onDone={(title) => {
          setRenaming(null);
          if (title && title !== c.title) void patch(c.id, { title });
        }}
      />
    ) : (
      <ChatRow
        key={c.id}
        chat={c}
        active={c.id === activeId}
        folders={data?.folders ?? []}
        onOpen={() => onOpen(c.id)}
        onRename={() => setRenaming(c.id)}
        onPin={() => void patch(c.id, { pinned: !c.pinned })}
        onMove={(folder) => void patch(c.id, { folder })}
        onExport={(format) => void exportChat(c, format).catch((e: Error) => setError(e.message))}
        onDelete={() => {
          if (!confirm(`Delete “${c.title}”? This can't be undone.`)) return;
          void run(api(`/api/chats/${c.id}`, { method: "DELETE" })).then(() => onDeleted(c.id));
        }}
      />
    );

  const group = (key: string, label: string, chats: ChatSummary[], icon: ReactNode, actions?: ReactNode, startOpen = true) => {
    const isOpen = open[key] ?? startOpen;
    return (
      <div key={key} className="group/section mt-2">
        <div className="flex h-7 items-center gap-1 px-1.5 text-[12px] font-medium text-muted">
          <button className="flex min-w-0 flex-1 items-center gap-1.5 hover:text-text" onClick={() => setOpen((o) => ({ ...o, [key]: !isOpen }))} aria-expanded={isOpen}>
            <ChevronRight className={cx("size-3 shrink-0 transition-transform", isOpen && "rotate-90")} />
            {icon}
            <span className="truncate">{label}</span>
            <span className="tabular-nums opacity-70">{chats.length || ""}</span>
          </button>
          {actions}
        </div>
        {isOpen && <div className="flex flex-col">{chats.map(item)}</div>}
      </div>
    );
  };

  const chats = data?.chats ?? [];
  const searching = query.trim() !== "";
  const imported = [...new Set(chats.filter((c) => c.source).map((c) => c.source!))].sort();

  return (
    <aside className="flex w-[248px] shrink-0 flex-col border-r border-line bg-surface/40" aria-label={t("chat.list")}>
      <div className="flex flex-col gap-2 px-3 pt-3 pb-2">
        <Button variant="secondary" className="justify-start" onClick={onNew}>
          <SquarePen className="size-4" /> {t("chat.new")}
        </Button>
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted" />
          <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("chat.search")} className="h-8 pr-7 pl-8 text-[13px]" aria-label={t("chat.search")} />
          {query && (
            <button className="absolute top-1/2 right-2 -translate-y-1/2 text-muted hover:text-text" onClick={() => setQuery("")} aria-label={t("chat.clear_search")}>
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-3">
        {error && <p className="mx-1.5 mb-2 text-[12px] text-danger">{error}</p>}
        {!data ? null : searching ? (
          chats.length ? (
            <div className="flex flex-col">{chats.map(item)}</div>
          ) : (
            <p className="px-2 py-3 text-[12.5px] text-muted">No chat mentions “{query.trim()}”.</p>
          )
        ) : (
          <>
            {chats.some((c) => c.pinned) && group("pinned", t("chat.pinned"), chats.filter((c) => c.pinned), <Pin className="size-3" />)}
            {data.folders.map((f) =>
              renaming === `folder:${f}` ? (
                <InlineName
                  key={f}
                  initial={f}
                  onDone={(name) => {
                    setRenaming(null);
                    if (name && name !== f) void run(api("/api/chats/folders/rename", { json: { name: f, new_name: name } }));
                  }}
                />
              ) : (
                group(
                  `folder:${f}`,
                  f,
                  chats.filter((c) => c.folder === f && !c.pinned),
                  <Folder className="size-3" />,
                  <span className="hidden items-center gap-1 group-hover/section:flex">
                    <button className="hover:text-text" onClick={() => setRenaming(`folder:${f}`)} title={`Rename ${f}`}>
                      <Pencil className="size-3" aria-label={`Rename ${f}`} />
                    </button>
                    <button
                      className="hover:text-danger"
                      onClick={() => confirm(`Remove the folder ${f}? Its chats stay.`) && void run(api("/api/chats/folders/remove", { json: { name: f } }))}
                      title={`Remove ${f}`}
                    >
                      <X className="size-3" aria-label={`Remove ${f}`} />
                    </button>
                  </span>,
                  false,
                )
              ),
            )}
            {newFolder && (
              <InlineName
                initial=""
                placeholder={t("chat.folder_name")}
                onDone={(name) => {
                  setNewFolder(false);
                  if (name) void run(api("/api/chats/folders/add", { json: { name } }));
                }}
              />
            )}
            {group(
              "recent",
              t("chat.list"),
              chats.filter((c) => !c.pinned && !c.folder && !c.source),
              null,
              <button className="hover:text-text" onClick={() => setNewFolder(true)} title={t("chat.new_folder")}>
                <FolderPlus className="size-3.5" aria-label={t("chat.new_folder")} />
              </button>,
            )}
            {imported.map((source) => group(`source:${source}`, t("chat.from", { source }), chats.filter((c) => c.source === source && !c.pinned && !c.folder), <History className="size-3" />, undefined, false))}
            {chats.length === 0 && <p className="px-2 py-3 text-[12.5px] text-muted">{t("chat.empty_list")}</p>}
          </>
        )}
      </div>
      <div className="border-t border-line px-3 py-2">
        <button className="flex items-center gap-1.5 text-[12.5px] text-muted hover:text-text" onClick={() => setImporting(true)}>
          <Download className="size-3.5" /> {t("chat.import")}
        </button>
      </div>
      {importing && <ImportPanel onClose={() => setImporting(false)} onImported={load} />}
    </aside>
  );
}

function ChatRow({
  chat,
  active,
  folders,
  onOpen,
  onRename,
  onPin,
  onMove,
  onExport,
  onDelete,
}: {
  chat: ChatSummary;
  active: boolean;
  folders: string[];
  onOpen: () => void;
  onRename: () => void;
  onPin: () => void;
  onMove: (folder: string) => void;
  onExport: (format: "md" | "json") => void;
  onDelete: () => void;
}) {
  const [menu, setMenu] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setMenu(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setMenu(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [menu]);
  const act = (fn: () => void) => () => {
    setMenu(false);
    fn();
  };
  const menuItem = "flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-[13px] hover:bg-surface-2";

  return (
    <div ref={ref} className="group/row relative">
      <button
        onClick={onOpen}
        aria-current={active ? "true" : undefined}
        className={cx("flex w-full flex-col rounded-md px-2.5 py-1.5 text-left", active ? "bg-surface shadow-[0_0_0_1px_var(--line)]" : "hover:bg-surface/70")}
        title={[chat.project && `Project: ${chat.project}`, chat.model && `Model: ${chat.model}`, relativeDate(chat.updated)].filter(Boolean).join("\n")}
      >
        <span className={cx("truncate pr-5 text-[13px]", active ? "font-medium" : "")}>{chat.title}</span>
        {chat.snippet && <Snippet text={chat.snippet} />}
      </button>
      <button
        onClick={() => setMenu((m) => !m)}
        className={cx("absolute top-1.5 right-1.5 rounded p-0.5 text-muted hover:bg-surface-2 hover:text-text", menu ? "block" : "hidden group-hover/row:block group-focus-within/row:block")}
        aria-label={`Actions for ${chat.title}`}
        aria-expanded={menu}
      >
        <MoreHorizontal className="size-3.5" />
      </button>
      {menu && (
        <div className="absolute top-8 right-1 z-20 w-52 rounded-lg border border-line bg-surface p-1 shadow-lg" role="menu">
          <button className={menuItem} onClick={act(onRename)} role="menuitem">
            <Pencil className="size-3.5" /> Rename
          </button>
          <button className={menuItem} onClick={act(onPin)} role="menuitem">
            {chat.pinned ? <PinOff className="size-3.5" /> : <Pin className="size-3.5" />} {chat.pinned ? "Unpin" : "Pin"}
          </button>
          {(folders.length > 0 || chat.folder) && (
            <label className="flex items-center gap-2 px-2 py-1.5 text-[13px]">
              <Folder className="size-3.5 shrink-0" />
              <select value={chat.folder ?? ""} onChange={(e) => act(() => onMove(e.target.value))()} className="min-w-0 flex-1 rounded border border-line bg-surface px-1 py-0.5 text-[12.5px]" aria-label="Move to folder">
                <option value="">No folder</option>
                {folders.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button className={menuItem} onClick={act(() => onExport("md"))} role="menuitem">
            <Download className="size-3.5" /> Export as Markdown
          </button>
          <button className={menuItem} onClick={act(() => onExport("json"))} role="menuitem">
            <Download className="size-3.5" /> Export as JSON
          </button>
          <button className={cx(menuItem, "text-danger")} onClick={act(onDelete)} role="menuitem">
            <Trash2 className="size-3.5" /> Delete
          </button>
        </div>
      )}
    </div>
  );
}

function InlineName({ initial, placeholder, onDone }: { initial: string; placeholder?: string; onDone: (value: string | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const finished = useRef(false);
  useEffect(() => ref.current?.select(), []);
  const finish = (value: string | null) => {
    if (finished.current) return;
    finished.current = true;
    onDone(value);
  };
  return (
    <form
      className="px-1 py-0.5"
      onSubmit={(e) => {
        e.preventDefault();
        finish(ref.current!.value.trim());
      }}
    >
      <Input ref={ref} defaultValue={initial} placeholder={placeholder} autoFocus className="h-7 px-2 text-[13px]" onKeyDown={(e) => e.key === "Escape" && finish(null)} onBlur={(e) => finish(e.currentTarget.value.trim())} aria-label={placeholder ?? "Name"} />
    </form>
  );
}

interface ImportSource {
  source: string;
  label: string;
  path: string;
  found: number;
  imported: number;
}

/** Finds Claude Code, Codex and Cursor history on this computer and imports it. */
function ImportPanel({ onClose, onImported }: { onClose: () => void; onImported: () => void }) {
  const [sources, setSources] = useState<ImportSource[] | null>(null);
  const load = useCallback(() => {
    api<ImportSource[]>("/api/chats/import/sources").then(setSources, () => setSources([]));
  }, []);
  useEffect(load, [load]);

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/30 p-6" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="w-full max-w-[460px] rounded-xl border border-line bg-bg p-5 shadow-xl" role="dialog" aria-label="Import chats">
        <div className="mb-1 flex items-center justify-between">
          <h2 className="font-semibold">Import chats</h2>
          <button className="text-muted hover:text-text" onClick={onClose} aria-label="Close">
            <X className="size-4" />
          </button>
        </div>
        <p className="mb-4 text-[12.5px] text-muted">
          Copies your conversations from these apps into Temper, so you can search them and continue them with a local model. Only what you and the assistant wrote comes across; tool calls and their output stay behind. Importing again adds what's new.
        </p>
        {!sources ? (
          <p className="text-muted">Looking for chats on this computer…</p>
        ) : (
          <ul className="divide-y divide-line rounded-lg border border-line bg-surface">
            {sources.map((s) => (
              <ImportRow
                key={s.source}
                source={s}
                onDone={() => {
                  load();
                  onImported();
                }}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function ImportRow({ source, onDone }: { source: ImportSource; onDone: () => void }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const last = [...events].reverse().find((e) => e.event === "progress") as { value?: number; added?: number; updated?: number } | undefined;
  const result = events.find((e) => e.event === "status")?.result as { added?: number; updated?: number; unchanged?: number } | undefined;
  useEffect(() => {
    if (status && status !== "running") onDone();
    // onDone changes identity on every render of the panel; the status change is what matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function start() {
    setError(null);
    try {
      setJobId((await api<{ id: string }>("/api/chats/import", { json: { source: source.source } })).id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <li className="px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="font-medium">
            {source.label}
          </p>
          <p className="truncate text-[12px] text-muted" title={source.path}>
            {source.found ? `${source.found} conversation${source.found === 1 ? "" : "s"} found` : "Nothing found"}
            {source.imported ? `, ${source.imported} already in Temper` : ""}
          </p>
        </div>
        <Button size="sm" variant={source.found ? "primary" : "secondary"} disabled={!source.found || running} onClick={() => void start()}>
          {running ? "Importing…" : source.imported ? "Import new" : "Import"}
        </Button>
      </div>
      {running && (
        <div className="mt-2">
          <ProgressBar value={last?.value ?? null} />
        </div>
      )}
      {status === "done" && result && (
        <p className="mt-1.5 text-[12px] text-muted">
          {result.added ?? 0} added, {result.updated ?? 0} updated{result.unchanged ? `, ${result.unchanged} already up to date` : ""}.
        </p>
      )}
      {(error || status === "failed") && <p className="mt-1.5 text-[12px] text-danger">{error ?? "The import stopped. Try again."}</p>}
    </li>
  );
}
