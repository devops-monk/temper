import { ArrowUp, Boxes, Eraser, MessageSquare, RotateCcw, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button, CopyButton, EmptyState, Field, Input, Notice, cx } from "../components/ui";
import { stream } from "../lib/engine";
import { useApp } from "../lib/store";

interface Message {
  role: "user" | "assistant";
  content: string;
  error?: boolean;
}

interface Settings {
  system: string;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  max_tokens: number;
}

const DEFAULTS: Settings = { system: "", temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, repeat_penalty: 1.05, max_tokens: 4096 };

function Slider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="mb-1 flex justify-between text-[13px]">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-[12px] text-muted">{value}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
    </label>
  );
}

export function ChatPage() {
  const { runtime, go } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState(DEFAULTS);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const ready = runtime?.state === "ready";

  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [messages]);

  async function send(history: Message[]) {
    setMessages([...history, { role: "assistant", content: "" }]);
    setBusy(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const { system, ...sampling } = settings;
    let text = "";
    try {
      await stream(
        "/v1/chat/completions",
        (data) => {
          if (data === "[DONE]") return;
          const delta = JSON.parse(data).choices?.[0]?.delta;
          const piece = (delta?.content ?? "") as string;
          if (!piece) return;
          text += piece;
          setMessages((m) => [...m.slice(0, -1), { role: "assistant", content: text }]);
        },
        {
          signal: ctrl.signal,
          json: {
            stream: true,
            messages: [...(system.trim() ? [{ role: "system", content: system }] : []), ...history.map(({ role, content }) => ({ role, content }))],
            ...sampling,
          },
        },
      );
    } catch (err) {
      if (!ctrl.signal.aborted) {
        setMessages((m) => [...m.slice(0, -1), { role: "assistant", content: text || (err as Error).message, error: !text }]);
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }

  function submit() {
    const content = draft.trim();
    if (!content || busy || !ready) return;
    setDraft("");
    void send([...messages, { role: "user", content }]);
  }

  const set = <K extends keyof Settings>(k: K) => (v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }));

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            ready ? (
              <EmptyState icon={<MessageSquare className="size-8" />} title={`Chat with ${runtime?.model}`}>
                Everything runs on this computer. Nothing you type leaves it.
              </EmptyState>
            ) : (
              <EmptyState icon={<Boxes className="size-8" />} title={runtime?.state === "loading" ? `Loading ${runtime.model}…` : "Load a model to start chatting"}>
                {runtime?.state === "loading" ? (
                  "Large models can take a minute to load into memory."
                ) : (
                  <>
                    <p>Pick a GGUF model you've downloaded, or find one on Hugging Face.</p>
                    <Button variant="primary" className="mt-4" onClick={() => go("models")}>
                      Open models
                    </Button>
                  </>
                )}
              </EmptyState>
            )
          ) : (
            <div className="mx-auto flex max-w-[760px] flex-col gap-6 px-6 py-8">
              {messages.map((m, i) => (
                <div key={i} className={cx("group", m.role === "user" && "self-end")}>
                  {m.role === "user" ? (
                    <div className="prose-chat max-w-[620px] rounded-2xl rounded-br-md bg-surface px-4 py-2.5 whitespace-pre-wrap shadow-[0_0_0_1px_var(--line)]">{m.content}</div>
                  ) : m.error ? (
                    <Notice tone="error">{m.content}</Notice>
                  ) : (
                    <div className="prose-chat leading-relaxed">
                      {m.content ? <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown> : <span className="animate-pulse-dot text-muted">Thinking…</span>}
                    </div>
                  )}
                  {m.role === "assistant" && m.content && !(busy && i === messages.length - 1) && (
                    <div className="mt-1 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                      <CopyButton text={m.content} />
                      {i === messages.length - 1 && (
                        <Button size="sm" variant="ghost" onClick={() => void send(messages.slice(0, -1))}>
                          <RotateCcw className="size-3.5" /> Regenerate
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mx-auto w-full max-w-[760px] px-6 pb-5">
          <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 focus-within:border-accent">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  submit();
                }
              }}
              rows={Math.min(8, Math.max(1, draft.split("\n").length))}
              placeholder={ready ? "Message the model" : "Load a model first"}
              disabled={!ready}
              className="max-h-52 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 outline-none placeholder:text-muted/70"
              aria-label="Message"
            />
            {busy ? (
              <Button variant="secondary" onClick={() => abort.current?.abort()} aria-label="Stop generating" className="size-9 rounded-xl px-0">
                <Square className="size-3.5 fill-current" />
              </Button>
            ) : (
              <Button variant="primary" onClick={submit} disabled={!draft.trim() || !ready} aria-label="Send" className="size-9 rounded-xl px-0">
                <ArrowUp className="size-4" />
              </Button>
            )}
          </div>
          <p className="mt-1.5 text-center text-[11.5px] text-muted">Enter to send, Shift+Enter for a new line</p>
        </div>
      </div>

      <aside className="w-[272px] shrink-0 overflow-y-auto border-l border-line bg-surface/40 px-4 py-5">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-semibold">Generation</h2>
          <Button size="sm" variant="ghost" onClick={() => setMessages([])} disabled={busy || !messages.length}>
            <Eraser className="size-3.5" /> Clear chat
          </Button>
        </div>
        <div className="flex flex-col gap-4">
          <Field label="System prompt">
            <textarea
              value={settings.system}
              onChange={(e) => set("system")(e.target.value)}
              rows={4}
              placeholder="You are a helpful assistant."
              className="w-full resize-y rounded-md border border-line bg-surface px-3 py-2 placeholder:text-muted/70 focus:border-accent focus:outline-none"
            />
          </Field>
          <Slider label="Temperature" value={settings.temperature} min={0} max={2} step={0.05} onChange={set("temperature")} />
          <Slider label="Top P" value={settings.top_p} min={0} max={1} step={0.01} onChange={set("top_p")} />
          <Slider label="Top K" value={settings.top_k} min={0} max={200} step={1} onChange={set("top_k")} />
          <Slider label="Min P" value={settings.min_p} min={0} max={1} step={0.01} onChange={set("min_p")} />
          <Slider label="Repeat penalty" value={settings.repeat_penalty} min={1} max={2} step={0.01} onChange={set("repeat_penalty")} />
          <Field label="Max new tokens">
            <Input type="number" min={16} value={settings.max_tokens} onChange={(e) => set("max_tokens")(Number(e.target.value))} />
          </Field>
          <Button size="sm" variant="ghost" onClick={() => setSettings(DEFAULTS)}>
            Reset to defaults
          </Button>
        </div>
      </aside>
    </div>
  );
}
