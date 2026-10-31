import { Check, Folder, FolderOpen, FolderPlus, Pencil, Plus, Search, Star, Trash2, X } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AddToOllama } from "../components/AddToOllama";
import { LoadButton, useEngines } from "../components/LoadButton";
import { PublishToHub } from "../components/PublishToHub";
import { Badge, Button, EmptyState, Input, Notice, PageHeader, Select, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { useFileDrop } from "../lib/useFileDrop";

type Kind = "upload" | "dataset" | "fine-tune" | "export" | "model" | "attachment" | "picture" | "speech" | "stems" | "sandbox";

interface Item {
  id: string;
  kind: Kind;
  name: string;
  /** The file's or run's own name, which renaming doesn't change. */
  original: string;
  path: string;
  size: number;
  modified: number;
  favourite: boolean;
  folder: string | null;
  /** Something the Train page can read. */
  dataset?: boolean;
  gguf?: string[];
  base_model?: string;
  /** The dataset a fine-tune was trained on. */
  trained_on?: string;
  adapter?: boolean;
  /** How a fine-tune was trained: "mlx" on Apple Silicon, else on a GPU. */
  backend?: string;
  formats?: string[];
  /** The model a picture was made with. */
  made_with?: string;
  voice?: string;
  stems?: string[];
  /** The chat whose code wrote a sandbox file. */
  chat?: string;
}

const KINDS: { kind: Kind; label: Key; plural: Key }[] = [
  { kind: "upload", label: "library.kind_upload", plural: "library.kinds_upload" },
  { kind: "dataset", label: "library.kind_dataset", plural: "library.kinds_dataset" },
  { kind: "fine-tune", label: "library.kind_fine_tune", plural: "library.kinds_fine_tune" },
  { kind: "export", label: "library.kind_export", plural: "library.kinds_export" },
  { kind: "model", label: "library.kind_model", plural: "library.kinds_model" },
  { kind: "attachment", label: "library.kind_attachment", plural: "library.kinds_attachment" },
  { kind: "picture", label: "library.kind_picture", plural: "library.kinds_picture" },
  { kind: "speech", label: "library.kind_speech", plural: "library.kind_speech" },
  { kind: "stems", label: "library.kind_stems", plural: "library.kind_stems" },
  { kind: "sandbox", label: "library.kind_sandbox", plural: "library.kind_sandbox" },
];
const KIND_LABEL = Object.fromEntries(KINDS.map((k) => [k.kind, k.label])) as Record<Kind, Key>;

type View = { type: "all" } | { type: "favourites" } | { type: "kind"; kind: Kind } | { type: "folder"; name: string };

/** Chat with a fine-tune before exporting it: its base model with the adapter on top. MLX runs MLX adapters; vLLM and SGLang
 * run the GPU trainer's. */
function ChatWithRun({ item }: { item: Item }) {
  const { engines } = useEngines();
  const t = useT();
  const engine = item.backend === "mlx" ? engines?.find((e) => e.id === "mlx" && e.available) : engines?.find((e) => (e.id === "vllm" || e.id === "sglang") && e.available);
  if (!engine) return null;
  return <LoadButton path={item.base_model!} name={item.name} engine={engine.id} adapter={`${item.path}/adapter`} label={t("library.chat_with_it")} />;
}

function details(item: Item, t: ReturnType<typeof useT>): string {
  if (item.kind === "fine-tune") return [item.base_model, t(item.adapter ? "library.adapter_saved" : "library.no_adapter")].filter(Boolean).join(", ");
  if (item.kind === "export") return item.gguf?.length ? (item.gguf.length > 1 ? t("library.gguf_many", { n: item.gguf.length }) : t("library.gguf_one")) : (item.formats ?? []).join(", ");
  if (item.kind === "picture" && item.made_with) return item.made_with.split("/").pop()!;
  if (item.kind === "speech" && item.voice) return t("library.voice", { voice: item.voice });
  if (item.kind === "stems") return (item.stems ?? []).join(", ");
  if (item.kind === "sandbox") return t(item.chat === "scratch" ? "library.code_outside_chat" : "library.code_in_chat");
  if (item.name !== item.original) return item.original;
  return "";
}

/** Everything Temper keeps on disk that people make or add: one list, with folders, favourites and names of their own. */
export function LibraryPage() {
  const t = useT();
  const [data, setData] = useState<{ folders: string[]; items: Item[] } | null>(null);
  const [view, setView] = useState<View>({ type: "all" });
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api<{ folders: string[]; items: Item[] }>("/api/files").then(setData, (e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const run = useCallback(
    async (call: Promise<unknown>) => {
      setError(null);
      try {
        await call;
      } catch (e) {
        setError((e as Error).message);
      }
      load();
    },
    [load],
  );

  const upload = useCallback((paths: string[]) => void run(api("/api/files/upload", { json: { paths } })), [run]);
  const dropping = useFileDrop(upload);

  async function chooseFiles() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({ multiple: true, title: t("library.add_files_title") });
    if (Array.isArray(picked) && picked.length) upload(picked);
    else if (typeof picked === "string") upload([picked]);
  }

  const shown = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    return data.items.filter((i) => {
      if (view.type === "favourites" && !i.favourite) return false;
      if (view.type === "kind" && i.kind !== view.kind) return false;
      if (view.type === "folder" && i.folder !== view.name) return false;
      return !q || i.name.toLowerCase().includes(q) || i.original.toLowerCase().includes(q);
    });
  }, [data, view, query]);

  const title = view.type === "all" ? t("library.all_files") : view.type === "favourites" ? t("library.favourites") : view.type === "kind" ? t(KINDS.find((k) => k.kind === view.kind)!.plural) : view.name;

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <PageHeader
        title={t("nav.library")}
        description={t("library.description")}
        actions={
          inTauri ? (
            <Button variant="primary" onClick={() => void chooseFiles()}>
              <Plus className="size-4" /> {t("library.add_files")}
            </Button>
          ) : null
        }
      />
      {dropping && (
        <div className="pointer-events-none absolute inset-3 z-10 flex items-center justify-center rounded-xl border-2 border-dashed border-accent bg-bg/85 font-medium">{t("library.drop")}</div>
      )}
      <div className="flex min-h-0 flex-1">
        <Sidebar data={data} view={view} setView={setView} run={run} />
        <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
          <div className="mb-4 flex items-center gap-3">
            <h2 className="font-semibold">{title}</h2>
            <span className="text-[12.5px] text-muted tabular-nums">{data ? shown.length : ""}</span>
            <div className="relative ml-auto w-64">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted" />
              <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("library.search_placeholder")} className="h-8 pl-8 text-[13px]" aria-label={t("library.search_label")} />
            </div>
          </div>
          {!inTauri && <BrowserUpload onUpload={upload} />}
          {error && (
            <div className="mb-3">
              <Notice tone="error">{error}</Notice>
            </div>
          )}
          {!data ? (
            <p className="text-muted">{t("library.reading")}</p>
          ) : shown.length === 0 ? (
            <EmptyState icon={<Folder className="size-7" />} title={t(query ? "library.no_match" : data.items.length ? "library.nothing_here" : "library.empty")}>
              {t(query ? "library.no_match_hint" : view.type === "folder" ? "library.folder_hint" : view.type === "favourites" ? "library.favourites_hint" : "library.empty_hint")}
            </EmptyState>
          ) : (
            <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
              {shown.map((item) => (
                <Row key={item.id} item={item} folders={data.folders} run={run} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function Sidebar({ data, view, setView, run }: { data: { folders: string[]; items: Item[] } | null; view: View; setView: (v: View) => void; run: (p: Promise<unknown>) => Promise<void> }) {
  const t = useT();
  const [adding, setAdding] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const count = (f: (i: Item) => boolean) => data?.items.filter(f).length ?? 0;
  const same = (v: View) => JSON.stringify(v) === JSON.stringify(view);

  const entry = (v: View, label: string, n: number, icon?: ReactNode) => (
    <button
      key={label}
      onClick={() => setView(v)}
      aria-current={same(v) ? "true" : undefined}
      className={cx("flex h-8 w-full items-center gap-2 rounded-md px-2.5 text-left text-[13px]", same(v) ? "bg-surface font-medium shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:bg-surface/60 hover:text-text")}
    >
      {icon}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      <span className="text-[11.5px] tabular-nums opacity-70">{n || ""}</span>
    </button>
  );

  return (
    <nav className="w-52 shrink-0 overflow-y-auto border-r border-line px-2.5 py-4" aria-label={t("library.sections")}>
      {entry({ type: "all" }, t("library.all_files"), count(() => true))}
      {entry({ type: "favourites" }, t("library.favourites"), count((i) => i.favourite), <Star className="size-3.5" />)}
      <div className="mt-3 mb-1 px-2.5 text-[12px] font-medium text-muted">{t("library.kinds")}</div>
      {KINDS.map((k) => entry({ type: "kind", kind: k.kind }, t(k.plural), count((i) => i.kind === k.kind)))}
      <div className="mt-3 mb-1 flex items-center justify-between px-2.5 text-[12px] font-medium text-muted">
        {t("library.folders")}
        <button className="hover:text-text" onClick={() => setAdding(true)} title={t("library.new_folder")}>
          <FolderPlus className="size-3.5" aria-label={t("library.new_folder")} />
        </button>
      </div>
      {data?.folders.map((name) =>
        renaming === name ? (
          <NameForm
            key={name}
            initial={name}
            onDone={(value) => {
              setRenaming(null);
              if (value && value !== name) void run(api("/api/files/folders", { method: "PUT", json: { name, new_name: value } })).then(() => same({ type: "folder", name }) && setView({ type: "folder", name: value }));
            }}
          />
        ) : (
          <div key={name} className="group relative">
            {entry({ type: "folder", name }, name, count((i) => i.folder === name), <Folder className="size-3.5" />)}
            <span className="absolute top-1/2 right-1.5 hidden -translate-y-1/2 items-center gap-1 rounded bg-surface-2 px-1 group-hover:flex">
              <button className="text-muted hover:text-text" onClick={() => setRenaming(name)} title={t("library.rename_named", { name })}>
                <Pencil className="size-3" aria-label={t("library.rename_named", { name })} />
              </button>
              <button
                className="text-muted hover:text-danger"
                onClick={() => {
                  if (!confirm(t("library.remove_folder_confirm", { name }))) return;
                  if (same({ type: "folder", name })) setView({ type: "all" });
                  void run(api(`/api/files/folders?name=${encodeURIComponent(name)}`, { method: "DELETE" }));
                }}
                title={t("library.remove_named", { name })}
              >
                <X className="size-3" aria-label={t("library.remove_named", { name })} />
              </button>
            </span>
          </div>
        ),
      )}
      {adding && (
        <NameForm
          initial=""
          placeholder={t("library.folder_name")}
          onDone={(value) => {
            setAdding(false);
            if (value) void run(api("/api/files/folders", { json: { name: value } }));
          }}
        />
      )}
      {data && !data.folders.length && !adding && <p className="px-2.5 text-[12px] text-muted">{t("library.folders_hint")}</p>}
    </nav>
  );
}

/** A one-line name editor: Enter saves, Escape cancels. */
function NameForm({ initial, placeholder, onDone }: { initial: string; placeholder?: string; onDone: (value: string | null) => void }) {
  const t = useT();
  const ref = useRef<HTMLInputElement>(null);
  const finished = useRef(false);
  useEffect(() => ref.current?.select(), []);
  // Enter, Escape and leaving the field can all fire as the form closes; only the first counts.
  const finish = (value: string | null) => {
    if (finished.current) return;
    finished.current = true;
    onDone(value);
  };
  return (
    <form
      className="flex items-center gap-1"
      onSubmit={(e) => {
        e.preventDefault();
        finish(ref.current!.value.trim());
      }}
    >
      <Input ref={ref} defaultValue={initial} placeholder={placeholder} autoFocus className="h-7 px-2 text-[13px]" onKeyDown={(e) => e.key === "Escape" && finish(null)} onBlur={(e) => finish(e.currentTarget.value.trim())} aria-label={placeholder ?? t("library.name")} />
      <button type="submit" className="text-muted hover:text-text" aria-label={t("library.save")}>
        <Check className="size-3.5" />
      </button>
    </form>
  );
}

function Row({ item, folders, run }: { item: Item; folders: string[]; run: (p: Promise<unknown>) => Promise<void> }) {
  const { go, setPendingDataset } = useApp();
  const t = useT();
  const [renaming, setRenaming] = useState(false);
  const patch = (json: Partial<Item>) => run(api("/api/files", { method: "PATCH", json: { id: item.id, ...json } }));
  const extra = details(item, t);
  const gguf = item.gguf?.[0];

  async function remove() {
    const what = t(item.kind === "fine-tune" ? "library.what_run" : item.kind === "export" ? "library.what_export" : item.kind === "stems" ? "library.what_stems" : "library.what_file");
    if (!confirm(t("library.delete_confirm", { what, name: item.name, size: bytes(item.size) }))) return;
    await run(api(`/api/files?id=${encodeURIComponent(item.id)}`, { method: "DELETE" }));
  }

  async function reveal() {
    const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
    await revealItemInDir(item.path);
  }

  return (
    <li className="group flex items-center gap-3 px-4 py-2.5">
      <button onClick={() => void patch({ favourite: !item.favourite })} className={cx(item.favourite ? "text-straw" : "text-muted hover:text-text")} title={t(item.favourite ? "library.unfavourite" : "library.favourite")}>
        <Star className={cx("size-4", item.favourite && "fill-current")} aria-label={t(item.favourite ? "library.is_favourite" : "library.not_favourite")} />
      </button>
      <div className="min-w-0 flex-1">
        {renaming ? (
          <NameForm
            initial={item.name}
            onDone={(value) => {
              setRenaming(false);
              if (value !== null && value !== item.name) void patch({ name: value });
            }}
          />
        ) : (
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate font-medium" title={item.path}>
              {item.name}
            </span>
            <button onClick={() => setRenaming(true)} className="hidden text-muted group-hover:inline hover:text-text" title={t("library.rename")}>
              <Pencil className="size-3" aria-label={t("library.rename_named", { name: item.name })} />
            </button>
            <Badge>{t(KIND_LABEL[item.kind])}</Badge>
            {item.folder && (
              <span className="flex items-center gap-1 text-[12px] text-muted">
                <Folder className="size-3" /> {item.folder}
              </span>
            )}
          </div>
        )}
        <p className="truncate text-[12px] text-muted tabular-nums">
          {bytes(item.size)} · {relativeDate(item.modified)}
          {extra && ` · ${extra}`}
        </p>
      </div>
      <span className="flex shrink-0 items-center gap-2">
        {item.dataset && (
          <Button
            size="sm"
            onClick={() => {
              setPendingDataset({ source: item.path });
              go("train");
            }}
          >
            {t("library.train_on_it")}
          </Button>
        )}
        {item.kind === "fine-tune" && item.adapter && (
          <Button size="sm" onClick={() => go("export")}>
            {t("library.export")}
          </Button>
        )}
        {item.kind === "fine-tune" && item.adapter && item.base_model && <ChatWithRun item={item} />}
        {item.kind === "picture" && (
          <Button size="sm" onClick={() => go("images")}>
            {t("library.open_images")}
          </Button>
        )}
        {(item.kind === "speech" || item.kind === "stems") && (
          <Button size="sm" onClick={() => go("audio")}>
            {t("library.open_audio")}
          </Button>
        )}
        {gguf && <LoadButton path={gguf} name={gguf.split(/[\\/]/).pop()!} />}
        {gguf && item.kind === "export" && <AddToOllama path={gguf} />}
        {item.kind === "fine-tune" && item.adapter && <PublishToHub what={{ run_id: item.path.split(/[\\/]/).pop()! }} name={item.original} />}
        {item.kind === "export" && <PublishToHub what={{ export: item.path }} name={item.original} />}
        {item.kind === "dataset" && item.path.endsWith(".jsonl") && <PublishToHub what={{ dataset: item.path }} name={item.original} />}
        {folders.length > 0 && (
          <Select value={item.folder ?? ""} onChange={(e) => void patch({ folder: e.target.value })} className="h-7 w-auto max-w-36 text-[12.5px]" aria-label={t("library.folder_for", { name: item.name })} title={t("library.move_to_folder")}>
            <option value="">{t("library.no_folder")}</option>
            {folders.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </Select>
        )}
        {inTauri && (
          <button className="text-muted hover:text-text" onClick={() => void reveal()} title={t("library.show_in_folder")}>
            <FolderOpen className="size-3.5" aria-label={t("library.show_named", { name: item.name })} />
          </button>
        )}
        <button className="text-muted hover:text-danger" onClick={() => void remove()} title={t("library.delete")}>
          <Trash2 className="size-3.5" aria-label={t("library.delete_named", { name: item.name })} />
        </button>
      </span>
    </li>
  );
}

/** In a browser there's no file dialog with paths, so files are added by path. */
function BrowserUpload({ onUpload }: { onUpload: (paths: string[]) => void }) {
  const t = useT();
  return (
    <form
      className="mb-4 flex gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        const path = (new FormData(e.currentTarget).get("path") as string).trim();
        if (path) onUpload([path]);
        e.currentTarget.reset();
      }}
    >
      <Input name="path" placeholder="/path/to/file.jsonl" spellCheck={false} className="h-8 flex-1 text-[13px]" aria-label={t("library.file_to_add")} />
      <Button size="sm" type="submit">
        {t("library.add_file")}
      </Button>
    </form>
  );
}
