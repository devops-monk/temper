import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { create } from "zustand";
import { Button } from "./ui";

const SEEN = "temper.tour";

interface Step {
  /** The rail item to point at (its data-tour); none for the welcome. */
  target?: string;
  title: string;
  body: string;
}

const STEPS: Step[] = [
  { title: "Welcome to Temper", body: "Temper runs, fine-tunes and exports AI models on this computer, without sending your data anywhere. Here's where everything is; it takes a minute." },
  { target: "models", title: "Models", body: "Find models on Hugging Face and download them, or add files you already have. Temper says which ones fit this computer. Load one to chat with it." },
  { target: "chat", title: "Chat", body: "Talk to the loaded model. Attach files and images, let it use tools, keep projects, or compare models side by side." },
  { target: "datasets", title: "Datasets", body: "Turn your own files, a Hugging Face dataset or a web link into training data, and check it before you train." },
  { target: "train", title: "Train", body: "Fine-tune a model on your data. Temper fits the settings to this computer, shows the loss and sample replies as it learns, and keeps the best checkpoint." },
  { target: "export", title: "Export", body: "Save what you trained as GGUF, MLX or an adapter, then add it to Ollama or publish it to Hugging Face." },
  { target: "connect", title: "Connect", body: "Use the loaded model from Claude Code, Codex, editors and your own code, through OpenAI- and Anthropic-compatible APIs." },
  { target: "settings", title: "Settings", body: "Appearance, accounts, privacy and your statistics. You can take this tour again from here." },
];

/** Whether the tour is showing; Settings starts it again. */
export const useTour = create<{ open: boolean; start: () => void; close: () => void }>((set) => ({
  open: false,
  start: () => set({ open: true }),
  close: () => {
    try {
      localStorage.setItem(SEEN, "1");
    } catch {
      // a blocked store only means the tour may show again
    }
    set({ open: false });
  },
}));

/** Opens the tour once, the first time the app is ready. */
export function useFirstRunTour(ready: boolean) {
  const start = useTour((t) => t.start);
  useEffect(() => {
    if (!ready) return;
    let seen = true;
    try {
      seen = localStorage.getItem(SEEN) === "1";
    } catch {
      // no store: don't nag
    }
    if (!seen) start();
  }, [ready, start]);
}

const CARD = 320;

/** A short walk around the rail: a spotlight on each part of the app, with a card beside it. */
export function Tour() {
  const { open, close } = useTour();
  const [i, setI] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const card = useRef<HTMLDivElement>(null);
  const step = STEPS[i];

  const measure = useCallback(() => {
    const el = step?.target ? document.querySelector(`[data-tour="${step.target}"]`) : null;
    setRect(el ? el.getBoundingClientRect() : null);
  }, [step]);

  useLayoutEffect(() => {
    if (!open) return;
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [open, measure]);

  useEffect(() => {
    if (open) card.current?.focus();
  }, [open, i]);

  useEffect(() => {
    if (!open) setI(0);
  }, [open]);

  if (!open || !step) return null;
  const last = i === STEPS.length - 1;
  const next = () => (last ? close() : setI((n) => Math.min(n + 1, STEPS.length - 1)));
  const back = () => setI((n) => Math.max(n - 1, 0));

  // Beside the item, kept on screen; centred for the welcome.
  const style = rect
    ? { left: rect.right + 14, top: Math.max(12, Math.min(rect.top - 8, window.innerHeight - 220)), width: CARD }
    : { left: "50%", top: "40%", width: CARD + 40, transform: "translate(-50%, -50%)" };

  return (
    <div
      className="fixed inset-0 z-[60]"
      onKeyDown={(e) => {
        if (e.key === "Escape") close();
        else if (e.key === "ArrowRight") next();
        else if (e.key === "ArrowLeft") back();
        else if (e.key === "Tab") {
          // Keep focus in the card: the app behind it is dimmed and out of reach.
          const items = [...(card.current?.querySelectorAll<HTMLElement>("button") ?? [])];
          const at = items.indexOf(document.activeElement as HTMLElement);
          e.preventDefault();
          items[(at + (e.shiftKey ? -1 : 1) + items.length) % items.length]?.focus();
        }
      }}
    >
      {rect ? (
        <div
          className="pointer-events-none absolute rounded-lg outline-2 outline-accent transition-all duration-200 outline-solid"
          style={{ left: rect.left - 4, top: rect.top - 4, width: rect.width + 8, height: rect.height + 8, boxShadow: "0 0 0 9999px rgb(0 0 0 / 0.55)" }}
        />
      ) : (
        <div className="absolute inset-0 bg-black/55" />
      )}
      <div
        ref={card}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-title"
        aria-describedby="tour-body"
        className="absolute rounded-xl border border-line bg-surface p-4 shadow-xl outline-none"
        style={style}
      >
        <div className="text-[11.5px] text-muted tabular-nums">
          {i + 1} of {STEPS.length}
        </div>
        <h2 id="tour-title" className="mt-1 text-[15px] font-semibold">
          {step.title}
        </h2>
        <p id="tour-body" className="mt-1.5 text-[13px] leading-relaxed text-muted">
          {step.body}
        </p>
        <div className="mt-4 flex items-center gap-2">
          <button className="text-[12.5px] text-muted hover:text-text" onClick={close}>
            {last ? "Close" : "Skip the tour"}
          </button>
          <span className="flex-1" />
          {i > 0 && (
            <Button size="sm" variant="ghost" onClick={back}>
              Back
            </Button>
          )}
          <Button size="sm" variant="primary" onClick={next}>
            {i === 0 ? "Show me" : last ? "Done" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  );
}
