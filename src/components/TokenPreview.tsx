import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { Notice, cx } from "./ui";

/** What the preview needs from the Train form. */
export interface PreviewSource {
  base_model: string;
  dataset: string;
  dataset_split: string;
  dataset_config: string;
  dataset_format: string;
  max_seq_length: number;
}

interface Preview {
  tokens: { id: number; text: string }[];
  count: number;
  masked: number;
  max_seq_length: number;
  format: string;
  row: number;
  more: boolean;
}

/** One dataset row token by token, as training sees it: the prompt dimmed (no loss on it), anything past the context cut. */
export function TokenPreview({ source, onClose }: { source: PreviewSource; onClose: () => void }) {
  const [row, setRow] = useState(0);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { base_model, dataset, dataset_split, dataset_config, dataset_format, max_seq_length } = source;

  useEffect(() => {
    let stale = false;
    // Waits for typing in the form to settle.
    const t = setTimeout(() => {
      api<Preview>("/api/train/preview", { json: { base_model, dataset, dataset_split, dataset_config: dataset_config || null, dataset_format, max_seq_length, row } }).then(
        (p) => !stale && (setPreview(p), setError(null)),
        (e: Error) => !stale && (setPreview(null), setError(e.message)),
      );
    }, 400);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [base_model, dataset, dataset_split, dataset_config, dataset_format, max_seq_length, row]);

  const cut = preview ? Math.max(0, preview.count - preview.max_seq_length) : 0;
  const trained = preview ? Math.min(preview.count, preview.max_seq_length) - Math.min(preview.masked, preview.max_seq_length) : 0;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
      <div className="flex items-center gap-2 text-[13px]">
        <span className="font-medium">Tokens, row {row + 1}</span>
        <button className="text-muted hover:text-text disabled:opacity-40" disabled={row === 0} onClick={() => setRow((r) => r - 1)} aria-label="Previous row">
          <ChevronLeft className="size-4" />
        </button>
        <button className="text-muted hover:text-text disabled:opacity-40" disabled={!preview?.more} onClick={() => setRow((r) => r + 1)} aria-label="Next row">
          <ChevronRight className="size-4" />
        </button>
        {preview && (
          <span className="text-[12px] text-muted">
            {preview.count} tokens: {trained} trained on{preview.masked > 0 && `, ${preview.masked} of prompt (dimmed)`}
          </span>
        )}
        <button className="ml-auto text-muted hover:text-text" onClick={onClose} aria-label="Close the token preview">
          <X className="size-4" />
        </button>
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      {cut > 0 && <Notice>This row is {preview!.count} tokens; the last {cut} are past the context length ({preview!.max_seq_length}) and are cut off in training.</Notice>}
      {preview && (
        <div className="max-h-[420px] overflow-y-auto font-mono text-[12px] leading-[1.9] break-words whitespace-pre-wrap">
          {preview.tokens.map((t, i) => (
            <span
              key={i}
              title={`Token ${i + 1}: id ${t.id}`}
              className={cx(
                "rounded-[3px]",
                i % 2 ? "bg-accent/12" : "bg-straw/12",
                i < preview.masked && "opacity-40",
                i >= preview.max_seq_length && "text-danger line-through",
              )}
            >
              {t.text.includes("\n") ? t.text.replace(/\n/g, "↵\n") : t.text}
            </span>
          ))}
          {preview.count > preview.tokens.length && <span className="text-muted"> … {preview.count - preview.tokens.length} more</span>}
        </div>
      )}
    </div>
  );
}
