import { ChevronDown, ChevronUp, X } from "lucide-react";
import { type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { Input } from "./ui";

const highlights = typeof CSS !== "undefined" && "highlights" in CSS ? CSS.highlights : null;

function matches(root: HTMLElement, query: string): Range[] {
  const needle = query.toLowerCase();
  const found: Range[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = (node.textContent ?? "").toLowerCase();
    for (let i = text.indexOf(needle); i !== -1; i = text.indexOf(needle, i + needle.length)) {
      const range = document.createRange();
      range.setStart(node, i);
      range.setEnd(node, i + needle.length);
      found.push(range);
    }
  }
  return found;
}

/** Finds text in the chat's messages (Mod+F): highlights every match and steps through them. */
export function FindBar({ root, onClose }: { root: RefObject<HTMLElement | null>; onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [ranges, setRanges] = useState<Range[]>([]);
  const [current, setCurrent] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  const search = useCallback(() => {
    const found = root.current && query ? matches(root.current, query) : [];
    setRanges(found);
    setCurrent((c) => (c < found.length ? c : 0));
  }, [root, query]);

  useEffect(() => {
    input.current?.focus();
    input.current?.select();
    const refocus = () => {
      input.current?.focus();
      input.current?.select();
    };
    window.addEventListener("temper:find", refocus);
    return () => {
      window.removeEventListener("temper:find", refocus);
      highlights?.delete("temper-find");
      highlights?.delete("temper-find-current");
    };
  }, []);

  useEffect(search, [search]);

  // Replies stream in and chats change underneath; search again when the messages do.
  useEffect(() => {
    if (!root.current) return;
    let timer: ReturnType<typeof setTimeout>;
    const observer = new MutationObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(search, 200);
    });
    observer.observe(root.current, { subtree: true, childList: true, characterData: true });
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, [root, search]);

  useEffect(() => {
    const range = ranges[current];
    if (highlights) {
      highlights.set("temper-find", new Highlight(...ranges));
      if (range) highlights.set("temper-find-current", new Highlight(range));
      else highlights.delete("temper-find-current");
    }
    range?.startContainer.parentElement?.scrollIntoView({ block: "center" });
  }, [ranges, current]);

  const step = (by: number) => ranges.length && setCurrent((c) => (c + by + ranges.length) % ranges.length);

  return (
    <div className="absolute top-2 right-4 z-20 flex items-center gap-1 rounded-lg border border-line bg-surface p-1 shadow-lg" role="search">
      <Input
        ref={input}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            step(e.shiftKey ? -1 : 1);
          } else if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            onClose();
          }
        }}
        placeholder="Find in chat"
        aria-label="Find in chat"
        className="h-7 w-52 text-[13px]"
      />
      <span className="w-16 text-center text-[12px] text-muted tabular-nums" aria-live="polite">
        {query ? (ranges.length ? `${current + 1} of ${ranges.length}` : "None") : ""}
      </span>
      <button onClick={() => step(-1)} disabled={!ranges.length} className="rounded p-1 text-muted hover:bg-surface-2 hover:text-text disabled:opacity-40" aria-label="Previous match" title="Previous (Shift+Enter)">
        <ChevronUp className="size-4" />
      </button>
      <button onClick={() => step(1)} disabled={!ranges.length} className="rounded p-1 text-muted hover:bg-surface-2 hover:text-text disabled:opacity-40" aria-label="Next match" title="Next (Enter)">
        <ChevronDown className="size-4" />
      </button>
      <button onClick={onClose} className="rounded p-1 text-muted hover:bg-surface-2 hover:text-text" aria-label="Close find" title="Close (Esc)">
        <X className="size-4" />
      </button>
    </div>
  );
}
