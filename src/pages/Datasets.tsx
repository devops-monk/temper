import { Database, FlaskConical, FolderOpen } from "lucide-react";
import { useState } from "react";
import { Badge, Button, EmptyState, Input, Notice, PageHeader } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { useApp } from "../lib/store";

interface Preview {
  source: string;
  columns: string[];
  rows: Record<string, unknown>[];
  format: string | null;
  local: boolean;
}

const FORMAT_LABEL: Record<string, string> = { messages: "OpenAI messages", sharegpt: "ShareGPT", alpaca: "Alpaca", text: "Plain text" };
const SUGGESTIONS = ["mlabonne/FineTome-100k", "yahma/alpaca-cleaned", "HuggingFaceH4/ultrachat_200k", "openai/gsm8k"];

function cell(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 280 ? text.slice(0, 280) + "…" : text;
}

export function DatasetsPage() {
  const { go, setPendingDataset } = useApp();
  const [source, setSource] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function open(src = source) {
    if (!src.trim()) return;
    setSource(src);
    setError(null);
    setLoading(true);
    try {
      setPreview(await api<Preview>(`/api/datasets/preview?source=${encodeURIComponent(src.trim())}`));
    } catch (e) {
      setPreview(null);
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function browse() {
    const { open: pick } = await import("@tauri-apps/plugin-dialog");
    const file = await pick({ filters: [{ name: "Datasets", extensions: ["jsonl", "json", "csv"] }] });
    if (typeof file === "string") void open(file);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Datasets"
        description="Check a dataset's columns and format before you train on it."
        actions={
          preview && (
            <Button variant="primary" onClick={() => { setPendingDataset(preview.source); go("train"); }} disabled={!preview.format}>
              <FlaskConical className="size-4" /> Train on this dataset
            </Button>
          )
        }
      />
      <div className="border-b border-line px-7 py-4">
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void open(); }}>
          <Input value={source} onChange={(e) => setSource(e.target.value)} placeholder="Hugging Face dataset id or a local file path" spellCheck={false} />
          {inTauri && (
            <Button type="button" onClick={browse}>
              <FolderOpen className="size-4" /> Choose file
            </Button>
          )}
          <Button type="submit" variant="primary" disabled={loading || !source.trim()}>
            {loading ? "Opening…" : "Open"}
          </Button>
        </form>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-7 py-5">
        {error && <Notice tone="error">{error}</Notice>}
        {!preview && !error && (
          <EmptyState icon={<Database className="size-8" />} title="Open a dataset to preview it">
            <p>Try one of these:</p>
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <Button key={s} size="sm" onClick={() => void open(s)}>
                  {s}
                </Button>
              ))}
            </div>
          </EmptyState>
        )}
        {preview && (
          <>
            <div className="mb-3 flex items-center gap-3 text-[13px]">
              <span className="font-medium">{preview.source}</span>
              {preview.format ? <Badge tone="ok">{FORMAT_LABEL[preview.format]}</Badge> : <Badge tone="warn">Unknown format</Badge>}
              <span className="text-muted">First {preview.rows.length} rows</span>
            </div>
            {!preview.format && (
              <div className="mb-4">
                <Notice>
                  Temper can't tell how to turn these columns into chat turns. Datasets need a <code>messages</code>, <code>conversations</code>, <code>instruction</code> + <code>output</code>, or <code>text</code> column. Column mapping is planned.
                </Notice>
              </div>
            )}
            <div className="overflow-x-auto rounded-lg border border-line bg-surface">
              <table className="w-full text-left text-[12.5px]">
                <thead className="bg-surface-2">
                  <tr>
                    <th className="w-10 px-3 py-2 font-medium text-muted">#</th>
                    {preview.columns.map((c) => (
                      <th key={c} className="px-3 py-2 font-mono font-medium">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.rows.map((row, i) => (
                    <tr key={i} className="border-t border-line align-top">
                      <td className="px-3 py-2 text-muted tabular-nums">{i + 1}</td>
                      {preview.columns.map((c) => (
                        <td key={c} className="max-w-[420px] px-3 py-2 font-mono text-[11.5px] break-words">
                          {cell(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
