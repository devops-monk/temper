import { GripVertical, Pencil, Play, X } from "lucide-react";
import { useState } from "react";
import { cx } from "./ui";

export interface Queued {
  id: number;
  text: string;
}

function move<T>(items: T[], from: number, to: number): T[] {
  const next = [...items];
  next.splice(to, 0, ...next.splice(from, 1));
  return next;
}

function QueueRow({ item, onSave, onRemove }: { item: Queued; onSave: (text: string) => void; onRemove: () => void }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(item.text);
  const finish = (save: boolean) => {
    if (save && text.trim()) onSave(text.trim());
    else setText(item.text);
    setEditing(false);
  };
  return editing ? (
    <textarea
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => finish(true)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
          e.preventDefault();
          finish(true);
        } else if (e.key === "Escape") finish(false);
      }}
      rows={Math.min(6, Math.max(2, text.split("\n").length))}
      autoFocus
      className="min-w-0 flex-1 resize-none rounded-md border border-accent bg-surface px-2 py-1 text-[13px] outline-none"
      aria-label="Edit queued message"
    />
  ) : (
    <>
      <button className="line-clamp-2 min-w-0 flex-1 text-left text-[13px] whitespace-pre-wrap" onClick={() => setEditing(true)} title="Edit">
        {item.text}
      </button>
      <button className="rounded p-1 text-muted opacity-0 group-hover:opacity-100 hover:bg-surface-2 hover:text-text focus:opacity-100" onClick={() => setEditing(true)} aria-label="Edit queued message">
        <Pencil className="size-3" />
      </button>
      <button className="rounded p-1 text-muted hover:bg-surface-2 hover:text-danger" onClick={onRemove} aria-label="Remove from queue">
        <X className="size-3.5" />
      </button>
    </>
  );
}

/** Follow-ups waiting for the reply being written, sent in order; edit them, drag to reorder, or remove them. */
export function PromptQueue({ items, paused, onChange, onResume }: { items: Queued[]; paused: boolean; onChange: (items: Queued[]) => void; onResume: () => void }) {
  const [dragging, setDragging] = useState<number | null>(null);
  const [over, setOver] = useState<number | null>(null);
  if (!items.length) return null;
  return (
    <div className="mb-2 rounded-xl border border-line bg-surface/60 px-2 py-1.5">
      <div className="flex items-center gap-2 px-1 pb-1 text-[12px] text-muted">
        <span>{paused ? `${items.length} queued, paused` : `${items.length} queued, sent when the reply finishes`}</span>
        <span className="ml-auto flex items-center gap-1">
          {paused && (
            <button className="flex items-center gap-1 rounded px-1.5 py-0.5 text-accent hover:bg-surface-2" onClick={onResume}>
              <Play className="size-3" /> Send next
            </button>
          )}
          <button className="rounded px-1.5 py-0.5 hover:bg-surface-2 hover:text-text" onClick={() => onChange([])}>
            Clear
          </button>
        </span>
      </div>
      <ol className="flex max-h-48 flex-col overflow-y-auto">
        {items.map((item, i) => (
          <li
            key={item.id}
            onDragOver={(e) => {
              if (dragging === null) return;
              e.preventDefault();
              setOver(i);
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (dragging !== null && dragging !== i) onChange(move(items, dragging, i));
              setDragging(null);
              setOver(null);
            }}
            className={cx("group flex items-start gap-1 rounded-md px-1 py-1", over === i && dragging !== i && "bg-surface-2", dragging === i && "opacity-50")}
          >
            <span
              draggable
              onDragStart={(e) => {
                e.dataTransfer.effectAllowed = "move";
                setDragging(i);
              }}
              onDragEnd={() => {
                setDragging(null);
                setOver(null);
              }}
              tabIndex={0}
              onKeyDown={(e) => {
                const to = e.key === "ArrowUp" ? i - 1 : e.key === "ArrowDown" ? i + 1 : -1;
                if (to < 0 || to >= items.length) return;
                e.preventDefault();
                onChange(move(items, i, to));
              }}
              className="mt-0.5 cursor-grab rounded text-muted hover:text-text focus-visible:outline-2 focus-visible:outline-accent"
              role="button"
              aria-label={`Move queued message ${i + 1}. Drag, or use the arrow keys.`}
            >
              <GripVertical className="size-3.5" />
            </span>
            <span className="mt-px w-4 shrink-0 text-right text-[12px] text-muted tabular-nums">{i + 1}.</span>
            <QueueRow item={item} onSave={(text) => onChange(items.map((q) => (q.id === item.id ? { ...q, text } : q)))} onRemove={() => onChange(items.filter((q) => q.id !== item.id))} />
          </li>
        ))}
      </ol>
    </div>
  );
}
