import { Check, Folder, FolderOpen, FolderPlus, Pencil, Plus, Search, Star, Trash2, X } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LoadButton } from "../components/LoadButton";
import { Badge, Button, EmptyState, Input, Notice, PageHeader, Select, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import { useFileDrop } from "../lib/useFileDrop";

type Kind = "upload" | "dataset" | "fine-tune" | "export" | "model";

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
  formats?: string[];
}

const KINDS: { kind: Kind; label: string; plural: string }[] = [
  { kind: "upload", label: "Upload", plural: "Uploads" },
  { kind: "dataset", label: "Dataset", plural: "Datasets" },
  { kind: "fine-tune", label: "Fine-tune", plural: "Fine-tunes" },
  { kind: "export", label: "Export", plural: "Exports" },
  { kind: "model", label: "Converted model", plural: "Converted models" },
];
const KIND_LABEL = Object.fromEntries(KINDS.map((k) => [k.kind, k.label])) as Record<Kind, string>;

type View = { type: "all" } | { type: "favourites" } | { type: "kind"; kind: Kind } | { type: "folder"; name: string };

function details(item: Item): string {
  if (item.kind === "fine-tune") return [item.base_model, item.adapter ? "adapter saved" : "no adapter yet"].filter(Boolean).join(", ");
  if (item.kind === "export") return item.gguf?.length ? `${item.gguf.length} GGUF file${item.gguf.length > 1 ? "s" : ""}` : (item.formats ?? []).join(", ");
  if (item.name !== item.original) return item.original;
  return "";
}

/** Everything Temper keeps on disk that people make or add: one list, with folders, favourites and names of their own. */
export function LibraryPage() {
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
    const picked = await open({ multiple: true, title: "Add files to the library" });
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

  const title = view.type === "all" ? "All files" : view.type === "favourites" ? "Favourites" : view.type === "kind" ? KINDS.find((k) => k.kind === view.kind)!.plural : view.name;

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Library"
        description="Your uploads, datasets, fine-tunes, exports and converted models. Renaming changes the name shown here; files on disk keep theirs."
        actions={
          inTauri ? (
            <Button variant="primary" onClick={() => void chooseFiles()}>
              <Plus className="size-4" /> Add files
            </Button>
          ) : null
        }
      />
      {dropping && (
        <div className="pointer-events-none absolute inset-3 z-10 flex items-center justify-center rounded-xl border-2 border-dashed border-accent bg-bg/85 font-medium">Drop to copy into your library</div>
      )}
      <div className="flex min-h-0 flex-1">
        <Sidebar data={data} view={view} setView={setView} run={run} />
        <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
          <div className="mb-4 flex items-center gap-3">
            <h2 className="font-semibold">{title}</h2>
            <span className="text-[12.5px] text-muted tabular-nums">{data ? shown.length : ""}</span>
            <div className="relative ml-auto w-64">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted" />
              <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by name" className="h-8 pl-8 text-[13px]" aria-label="Search the library" />
            </div>
          </div>
          {!inTauri && <BrowserUpload onUpload={upload} />}
          {error && (
            <div className="mb-3">
              <Notice tone="error">{error}</Notice>
            </div>
          )}
          {!data ? (
            <p className="text-muted">Reading the library…</p>
          ) : shown.length === 0 ? (
            <EmptyState icon={<Folder className="size-7" />} title={query ? "Nothing matches that name" : data.items.length ? "Nothing here yet" : "Your library is empty"}>
              {query
                ? "Try part of the name, or the file's own name."
                : view.type === "folder"
                  ? "Move files here with the folder menu on each one."
                  : view.type === "favourites"
                    ? "Star a file to keep it here."
                    : "Datasets you prepare or generate, training runs, exports and converted models appear here. Drop files on this page to add your own."}
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
    <nav className="w-52 shrink-0 overflow-y-auto border-r border-line px-2.5 py-4" aria-label="Library sections">
      {entry({ type: "all" }, "All files", count(() => true))}
      {entry({ type: "favourites" }, "Favourites", count((i) => i.favourite), <Star className="size-3.5" />)}
      <div className="mt-3 mb-1 px-2.5 text-[12px] font-medium text-muted">Kinds</div>
      {KINDS.map((k) => entry({ type: "kind", kind: k.kind }, k.plural, count((i) => i.kind === k.kind)))}
      <div className="mt-3 mb-1 flex items-center justify-between px-2.5 text-[12px] font-medium text-muted">
        Folders
        <button className="hover:text-text" onClick={() => setAdding(true)} title="New folder">
          <FolderPlus className="size-3.5" aria-label="New folder" />
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
              <button className="text-muted hover:text-text" onClick={() => setRenaming(name)} title={`Rename ${name}`}>
                <Pencil className="size-3" aria-label={`Rename ${name}`} />
              </button>
              <button
                className="text-muted hover:text-danger"
                onClick={() => {
                  if (!confirm(`Remove the folder ${name}? The files in it stay in your library.`)) return;
                  if (same({ type: "folder", name })) setView({ type: "all" });
                  void run(api(`/api/files/folders?name=${encodeURIComponent(name)}`, { method: "DELETE" }));
                }}
                title={`Remove ${name}`}
              >
                <X className="size-3" aria-label={`Remove ${name}`} />
              </button>
            </span>
          </div>
        ),
      )}
      {adding && (
        <NameForm
          initial=""
          placeholder="Folder name"
          onDone={(value) => {
            setAdding(false);
            if (value) void run(api("/api/files/folders", { json: { name: value } }));
          }}
        />
      )}
      {data && !data.folders.length && !adding && <p className="px-2.5 text-[12px] text-muted">Group files your own way, e.g. by project.</p>}
    </nav>
  );
}

/** A one-line name editor: Enter saves, Escape cancels. */
function NameForm({ initial, placeholder, onDone }: { initial: string; placeholder?: string; onDone: (value: string | null) => void }) {
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
      <Input ref={ref} defaultValue={initial} placeholder={placeholder} autoFocus className="h-7 px-2 text-[13px]" onKeyDown={(e) => e.key === "Escape" && finish(null)} onBlur={(e) => finish(e.currentTarget.value.trim())} aria-label={placeholder ?? "Name"} />
      <button type="submit" className="text-muted hover:text-text" aria-label="Save">
        <Check className="size-3.5" />
      </button>
    </form>
  );
}

function Row({ item, folders, run }: { item: Item; folders: string[]; run: (p: Promise<unknown>) => Promise<void> }) {
  const { go, setPendingDataset } = useApp();
  const [renaming, setRenaming] = useState(false);
  const patch = (json: Partial<Item>) => run(api("/api/files", { method: "PATCH", json: { id: item.id, ...json } }));
  const extra = details(item);
  const gguf = item.gguf?.[0];

  async function remove() {
    const what = item.kind === "fine-tune" ? "this training run and its adapter" : item.kind === "export" ? "this export and all its files" : "this file";
    if (!confirm(`Delete ${what} (${item.name})? This frees ${bytes(item.size)} and can't be undone.`)) return;
    await run(api(`/api/files?id=${encodeURIComponent(item.id)}`, { method: "DELETE" }));
  }

  async function reveal() {
    const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
    await revealItemInDir(item.path);
  }

  return (
    <li className="group flex items-center gap-3 px-4 py-2.5">
      <button onClick={() => void patch({ favourite: !item.favourite })} className={cx(item.favourite ? "text-straw" : "text-muted/50 hover:text-muted")} title={item.favourite ? "Remove from favourites" : "Add to favourites"}>
        <Star className={cx("size-4", item.favourite && "fill-current")} aria-label={item.favourite ? "Favourite" : "Not a favourite"} />
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
            <button onClick={() => setRenaming(true)} className="hidden text-muted group-hover:inline hover:text-text" title="Rename">
              <Pencil className="size-3" aria-label={`Rename ${item.name}`} />
            </button>
            <Badge>{KIND_LABEL[item.kind]}</Badge>
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
            Train on it
          </Button>
        )}
        {item.kind === "fine-tune" && item.adapter && (
          <Button size="sm" onClick={() => go("export")}>
            Export
          </Button>
        )}
        {gguf && <LoadButton path={gguf} name={gguf.split(/[\\/]/).pop()!} />}
        {folders.length > 0 && (
          <Select value={item.folder ?? ""} onChange={(e) => void patch({ folder: e.target.value })} className="h-7 w-auto max-w-36 text-[12.5px]" aria-label={`Folder for ${item.name}`} title="Move to a folder">
            <option value="">No folder</option>
            {folders.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </Select>
        )}
        {inTauri && (
          <button className="text-muted hover:text-text" onClick={() => void reveal()} title="Show in folder">
            <FolderOpen className="size-3.5" aria-label={`Show ${item.name} in its folder`} />
          </button>
        )}
        <button className="text-muted hover:text-danger" onClick={() => void remove()} title="Delete">
          <Trash2 className="size-3.5" aria-label={`Delete ${item.name}`} />
        </button>
      </span>
    </li>
  );
}

/** In a browser there's no file dialog with paths, so files are added by path. */
function BrowserUpload({ onUpload }: { onUpload: (paths: string[]) => void }) {
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
      <Input name="path" placeholder="/path/to/file.jsonl" spellCheck={false} className="h-8 flex-1 text-[13px]" aria-label="File to add" />
      <Button size="sm" type="submit">
        Add file
      </Button>
    </form>
  );
}
