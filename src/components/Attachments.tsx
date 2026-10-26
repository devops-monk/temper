import { FileText, Music, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, apiBlob, inTauri } from "../lib/engine";
import { bytes } from "../lib/format";
import { cx } from "./ui";

/** A file sent with a chat message (engine attachments.py). */
export interface Attachment {
  id: string;
  name: string;
  kind: "image" | "audio" | "document";
  size?: number;
  pages?: number;
  rows?: number;
  /** Characters of text read from a document. */
  chars?: number;
  truncated?: boolean;
}

export const ACCEPT = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".wav", ".mp3", ".pdf", ".docx", ".csv", ".tsv", ".txt", ".md", ".json", ".jsonl", ".html", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".tsx", ".rs", ".go", ".java", ".c", ".cpp", ".sh", ".sql"];

/** About four characters a token, which is close enough to warn before a file overflows the context. */
export const approxTokens = (a: Attachment) => Math.ceil((a.chars ?? 0) / 4);

/** The part of a chat message that refers to an attachment; /v1/chat/completions expands it. */
export const attachmentPart = (a: Attachment) => ({ type: "attachment", id: a.id });

/** Keeps what a saved message needs to show and resend the file. */
export const stored = ({ id, name, kind, size, pages, rows, chars, truncated }: Attachment): Attachment => ({ id, name, kind, size, pages, rows, chars, truncated });

function readBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
    reader.onerror = () => reject(new Error(`${file.name} couldn't be read.`));
    reader.readAsDataURL(file);
  });
}

/** Attaches files by path (the desktop app) or by contents (a browser, or pasted images). Returns what attached and the errors. */
export async function attach(files: (string | File)[]): Promise<{ added: Attachment[]; errors: string[] }> {
  const results = await Promise.allSettled(
    files.map(async (f) => (typeof f === "string" ? api<Attachment>("/api/attachments", { json: { path: f } }) : api<Attachment>("/api/attachments", { json: { name: f.name || "pasted-image.png", data: await readBase64(f) } }))),
  );
  return {
    added: results.flatMap((r) => (r.status === "fulfilled" ? [r.value] : [])),
    errors: results.flatMap((r) => (r.status === "rejected" ? [(r.reason as Error).message] : [])),
  };
}

/** Opens the file picker: the system dialog in the desktop app, a file input in a browser. */
export async function pickFiles(): Promise<(string | File)[]> {
  if (inTauri) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({ multiple: true, title: "Attach files", filters: [{ name: "Images, audio and documents", extensions: ACCEPT.map((e) => e.slice(1)) }] });
    return Array.isArray(picked) ? picked : picked ? [picked] : [];
  }
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ACCEPT.join(",");
    input.onchange = () => resolve([...(input.files ?? [])]);
    input.click();
  });
}

function useBlobUrl(path: string | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!path) return;
    let made: string | null = null;
    let gone = false;
    apiBlob(path).then(
      (blob) => {
        if (gone) return;
        made = URL.createObjectURL(blob);
        setUrl(made);
      },
      () => setUrl(null),
    );
    return () => {
      gone = true;
      if (made) URL.revokeObjectURL(made);
    };
  }, [path]);
  return url;
}

function describe(a: Attachment, context: number): { text: string; warn: boolean } {
  if (a.kind !== "document") return { text: bytes(a.size), warn: false };
  const tokens = approxTokens(a);
  const size = a.pages ? `${a.pages} page${a.pages > 1 ? "s" : ""}` : a.rows ? `${a.rows} rows` : null;
  const length = tokens >= 1000 ? `about ${(tokens / 1000).toFixed(tokens >= 10_000 ? 0 : 1)}K tokens` : `about ${tokens} tokens`;
  return { text: [size, length, a.truncated && "cut short"].filter(Boolean).join(", "), warn: tokens > context * 0.75 };
}

/** A file in the message box (with a remove button) or in a sent message. */
export function AttachmentChip({ attachment: a, context, onRemove }: { attachment: Attachment; context: number; onRemove?: () => void }) {
  const src = useBlobUrl(a.kind === "document" ? null : `/api/attachments/${a.id}/file`);
  const { text, warn } = describe(a, context);
  if (a.kind === "image" && !onRemove)
    return src ? <img src={src} alt={a.name} title={a.name} className="max-h-60 max-w-full rounded-lg border border-line object-contain" /> : <div className="h-24 w-32 rounded-lg bg-surface-2" />;
  return (
    <div className="flex max-w-[260px] min-w-0 items-center gap-2 rounded-lg border border-line bg-surface px-2 py-1.5" title={warn ? `${a.name} may not fit in the model's ${context.toLocaleString()}-token context.` : a.name}>
      {a.kind === "image" ? (
        src ? <img src={src} alt="" className="size-9 shrink-0 rounded object-cover" /> : <div className="size-9 shrink-0 rounded bg-surface-2" />
      ) : (
        <span className="flex size-9 shrink-0 items-center justify-center rounded bg-surface-2 text-muted">{a.kind === "audio" ? <Music className="size-4" /> : <FileText className="size-4" />}</span>
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12.5px] font-medium">{a.name}</span>
        {a.kind === "audio" && !onRemove && src ? <audio src={src} controls className="mt-1 h-7 w-48" /> : <span className={cx("block truncate text-[11.5px]", warn ? "text-straw" : "text-muted")}>{text}</span>}
      </span>
      {onRemove && (
        <button className="rounded p-0.5 text-muted hover:bg-surface-2 hover:text-text" onClick={onRemove} aria-label={`Remove ${a.name}`}>
          <X className="size-3.5" />
        </button>
      )}
    </div>
  );
}
