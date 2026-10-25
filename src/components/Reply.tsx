import { Brain, ChevronRight } from "lucide-react";
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cx } from "./ui";

/** A reply's reasoning, open while the model thinks and folded away once it starts answering. */
export function ThinkingBlock({ text, live, seconds }: { text: string; live: boolean; seconds?: number }) {
  const [open, setOpen] = useState<boolean | null>(null);
  const shown = open ?? live;
  const label = live ? "Thinking…" : seconds !== undefined ? `Thought for ${seconds < 1 ? "under a second" : `${Math.round(seconds)} s`}` : "Thinking";
  return (
    <div className="mb-3">
      <button onClick={() => setOpen(!shown)} className="flex items-center gap-1.5 text-[12.5px] text-muted hover:text-text" aria-expanded={shown}>
        <Brain className={cx("size-3.5", live && "animate-pulse-dot")} />
        {label}
        <ChevronRight className={cx("size-3 transition-transform", shown && "rotate-90")} />
      </button>
      {shown && <div className="mt-2 max-h-80 overflow-y-auto border-l-2 border-line pl-3 text-[13px] leading-relaxed whitespace-pre-wrap text-muted">{text}</div>}
    </div>
  );
}

/**
 * Markdown lets `*` emphasise inside a word, so "4*3 = 12 … 6*1.5" renders half a sentence in
 * italics. Escape a `*` with a letter or digit on both sides, outside code; real emphasis always has
 * a space or punctuation on one side.
 */
function literalAsterisks(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) => (i % 2 ? part : part.replace(/([\p{L}\p{N}])\*(?=[\p{L}\p{N}])/gu, "$1\\*")))
    .join("");
}

/** A model's answer, as Markdown. */
export function ReplyText({ text }: { text: string }) {
  return <Markdown remarkPlugins={[remarkGfm]}>{literalAsterisks(text)}</Markdown>;
}
