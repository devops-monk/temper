import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Notice, cx } from "./ui";

/** What the preview needs from the Train form. */
export interface PreviewSource {
  base_model: string;
  dataset: string;
  dataset_split: string;
  dataset_config: string;
  dataset_format: string;
  max_seq_length: number;
  chat_template: string;
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
  const t = useT();
  const [row, setRow] = useState(0);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { base_model, dataset, dataset_split, dataset_config, dataset_format, max_seq_length, chat_template } = source;

  useEffect(() => {
    let stale = false;
    // Waits for typing in the form to settle.
    const timer = setTimeout(() => {
      api<Preview>("/api/train/preview", { json: { base_model, dataset, dataset_split, dataset_config: dataset_config || null, dataset_format, max_seq_length, chat_template: chat_template || null, row } }).then(
        (p) => !stale && (setPreview(p), setError(null)),
        (e: Error) => !stale && (setPreview(null), setError(e.message)),
      );
    }, 400);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [base_model, dataset, dataset_split, dataset_config, dataset_format, max_seq_length, chat_template, row]);

  const cut = preview ? Math.max(0, preview.count - preview.max_seq_length) : 0;
  const trained = preview ? Math.min(preview.count, preview.max_seq_length) - Math.min(preview.masked, preview.max_seq_length) : 0;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
      <div className="flex items-center gap-2 text-[13px]">
        <span className="font-medium">{t("tokens.row", { n: row + 1 })}</span>
        <button className="text-muted hover:text-text disabled:opacity-40" disabled={row === 0} onClick={() => setRow((r) => r - 1)} aria-label={t("tokens.previous")}>
          <ChevronLeft className="size-4" />
        </button>
        <button className="text-muted hover:text-text disabled:opacity-40" disabled={!preview?.more} onClick={() => setRow((r) => r + 1)} aria-label={t("tokens.next")}>
          <ChevronRight className="size-4" />
        </button>
        {preview && (
          <span className="text-[12px] text-muted">
            {t("tokens.summary", { count: preview.count, trained })}
            {preview.masked > 0 && t("tokens.masked", { n: preview.masked })}
          </span>
        )}
        <button className="ml-auto text-muted hover:text-text" onClick={onClose} aria-label={t("tokens.close")}>
          <X className="size-4" />
        </button>
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      {cut > 0 && <Notice>{t("tokens.cut", { count: preview!.count, cut, max: preview!.max_seq_length })}</Notice>}
      {preview && (
        <div className="max-h-[420px] overflow-y-auto font-mono text-[12px] leading-[1.9] break-words whitespace-pre-wrap">
          {preview.tokens.map((token, i) => (
            <span
              key={i}
              title={t("tokens.token", { n: i + 1, id: token.id })}
              className={cx(
                "rounded-[3px]",
                i % 2 ? "bg-accent/12" : "bg-straw/12",
                i < preview.masked && "opacity-40",
                i >= preview.max_seq_length && "text-danger line-through",
              )}
            >
              {token.text.includes("\n") ? token.text.replace(/\n/g, "↵\n") : token.text}
            </span>
          ))}
          {preview.count > preview.tokens.length && <span className="text-muted"> {t("tokens.more", { n: preview.count - preview.tokens.length })}</span>}
        </div>
      )}
    </div>
  );
}
