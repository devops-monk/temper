import { ArrowUp, Boxes, Brain, ChevronRight, History, MessageSquare, RotateCcw, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { samplingText, type CatalogModel } from "../components/Catalog";
import { ChatList } from "../components/ChatList";
import { Button, CopyButton, EmptyState, Field, Input, Notice, Select, Toggle, cx } from "../components/ui";
import { api, stream } from "../lib/engine";
import { useApp } from "../lib/store";

interface Message {
  role: "user" | "assistant";
  content: string;
  error?: boolean;
  /** The model that wrote a reply. */
  model?: string | null;
  /** What a thinking model reasoned before answering. */
  reasoning?: string | null;
  /** Seconds spent thinking, known only for replies written in this session. */
  thought?: number;
}

/** Per-model thinking choices, remembered in this browser. */
interface Thinking {
  on: boolean;
  effort: "default" | "low" | "medium" | "high";
  /** Most tokens to spend thinking; 0 for no limit. */
  budget: number;
}

const THINKING_DEFAULTS: Thinking = { on: true, effort: "default", budget: 0 };

function loadThinking(model: string | null): Thinking {
  try {
    return { ...THINKING_DEFAULTS, ...JSON.parse(localStorage.getItem(`temper.thinking.${model}`) ?? "{}") };
  } catch {
    return THINKING_DEFAULTS;
  }
}

function saveThinking(model: string | null, value: Thinking) {
  try {
    localStorage.setItem(`temper.thinking.${model}`, JSON.stringify(value));
  } catch {
    /* private window: the choice lasts until the page closes */
  }
}

/** A reply's reasoning, open while the model thinks and folded away once it starts answering. */
function ThinkingBlock({ text, live, seconds }: { text: string; live: boolean; seconds?: number }) {
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

interface SavedChat {
  id: string;
  title: string;
  source: string | null;
  messages: Message[];
  settings: Partial<Settings> | null;
}

interface Settings {
  system: string;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  presence_penalty: number;
  max_tokens: number;
}

const DEFAULTS: Settings = { system: "", temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, repeat_penalty: 1.05, presence_penalty: 0, max_tokens: 4096 };
/** What a publisher's recommendation leaves unset means "off", as in the libraries they test with. */
const NEUTRAL = { top_p: 1, top_k: 0, min_p: 0, repeat_penalty: 1, presence_penalty: 0 };

type Recommended = Pick<CatalogModel, "id" | "sampling"> & { publisher: string; system?: string | null };

/** "Qwen/Qwen3.5-4B" → "Qwen3.5-4B"; an Ollama name like "qwen2.5:7b" stays as it is. */
const shortName = (id: string) => id.split("/").pop()!;

function recommendedSettings(base: Settings, rec: Recommended): Settings {
  const { repetition_penalty, ...rest } = rec.sampling;
  return { ...base, ...NEUTRAL, ...rest, ...(repetition_penalty !== undefined && { repeat_penalty: repetition_penalty }) };
}

const SAMPLING_KEYS = ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty"] as const;

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

export function ChatPage() {
  const { runtime, go } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState(DEFAULTS);
  const [recommended, setRecommended] = useState<Recommended | null>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const ready = runtime?.state === "ready";
  // The saved thread on screen; null until the first message of a new chat is sent.
  const [chat, setChat] = useState<{ id: string; title: string; source: string | null } | null>(null);
  const chatId = useRef<string | null>(null);
  const [listVersion, setListVersion] = useState(0);
  const [thinking, setThinking] = useState<Thinking>(THINKING_DEFAULTS);
  const reasoning = ready ? runtime?.reasoning : null;
  useEffect(() => setThinking(loadThinking(runtime?.model ?? null)), [runtime?.model]);
  const changeThinking = (next: Partial<Thinking>) =>
    setThinking((t) => {
      const value = { ...t, ...next };
      saveThinking(runtime?.model ?? null, value);
      return value;
    });
  const [saveError, setSaveError] = useState<string | null>(null);

  /** Saves the whole thread to the chat it belongs to, creating the chat on its first message. */
  async function persist(id: string | null, history: Message[]): Promise<string | null> {
    const body = { messages: history.map(({ role, content, error, model, reasoning }) => ({ role, content, error: !!error, model: model ?? null, reasoning: reasoning || null })), model: runtime?.model ?? null, settings };
    try {
      if (id) await api(`/api/chats/${id}/messages`, { method: "PUT", json: body });
      else {
        const created = await api<SavedChat>("/api/chats", { json: body });
        id = created.id;
        if (chatId.current === null) {
          chatId.current = id;
          setChat({ id, title: created.title, source: null });
        }
      }
      setSaveError(null);
      setListVersion((v) => v + 1);
    } catch (e) {
      setSaveError(`This chat couldn't be saved: ${(e as Error).message}`);
    }
    return id;
  }

  async function openChat(id: string) {
    if (id === chatId.current) return;
    abort.current?.abort();
    const saved = await api<SavedChat>(`/api/chats/${id}`).catch((e: Error) => {
      setSaveError(e.message);
      return null;
    });
    if (!saved) return;
    chatId.current = id;
    setChat({ id, title: saved.title, source: saved.source });
    setMessages(saved.messages.map(({ role, content, error, model, reasoning }) => ({ role: role === "assistant" ? "assistant" : "user", content, error, model, reasoning })));
    // The thread's own system prompt comes back with it; sampling stays with the loaded model.
    setSettings((s) => ({ ...s, system: saved.settings?.system ?? "" }));
  }

  function newChat() {
    abort.current?.abort();
    chatId.current = null;
    setChat(null);
    setMessages([]);
  }

  // A model from the catalog gets its publisher's sampling settings when it loads.
  const loaded = ready ? (runtime?.path ?? runtime?.model ?? null) : null;
  useEffect(() => {
    if (!loaded) return;
    let stale = false;
    api<{ match: (CatalogModel & { publisher: string; system?: string | null }) | null }>(`/api/catalog/match?model=${encodeURIComponent(loaded)}`)
      .then(({ match }) => {
        if (stale) return;
        const rec = match && Object.keys(match.sampling).length ? match : null;
        setRecommended(rec);
        setSettings((s) => {
          const next = rec ? recommendedSettings(s, rec) : { ...s, ...Object.fromEntries(SAMPLING_KEYS.map((k) => [k, DEFAULTS[k]])) };
          // A system prompt that came with the model (an Ollama Modelfile) fills an empty one, never replaces yours.
          return match?.system && !s.system.trim() ? { ...next, system: match.system } : next;
        });
      })
      .catch(() => setRecommended(null));
    return () => {
      stale = true;
    };
  }, [loaded]);
  const usingRecommended = recommended !== null && SAMPLING_KEYS.every((k) => settings[k] === recommendedSettings(settings, recommended)[k]);

  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [messages]);

  /** Request fields for the thinking choices, only those the model's template understands. */
  function thinkingFields() {
    if (!reasoning) return {};
    const thinkingOn = !reasoning.toggle || thinking.on;
    return {
      ...(reasoning.toggle && !thinking.on && { chat_template_kwargs: { enable_thinking: false } }),
      ...(reasoning.effort && thinking.effort !== "default" && { reasoning_effort: thinking.effort }),
      ...(thinkingOn && thinking.budget > 0 && { thinking_budget_tokens: thinking.budget }),
    };
  }

  async function send(history: Message[]) {
    setMessages([...history, { role: "assistant", content: "" }]);
    setBusy(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const { system, ...sampling } = settings;
    const model = runtime?.model ?? null;
    let text = "";
    let thoughts = "";
    let thought: number | undefined;
    const started = performance.now();
    let failure: string | null = null;
    const current = (): Message => ({ role: "assistant", content: text, model, reasoning: thoughts || null, thought });
    // Saved before the reply streams, so a long answer or a crash never loses the question.
    const id = await persist(chatId.current, history);
    try {
      await stream(
        "/v1/chat/completions",
        (data) => {
          if (data === "[DONE]") return;
          const delta = JSON.parse(data).choices?.[0]?.delta;
          const thinkingPiece = (delta?.reasoning_content ?? "") as string;
          const piece = (delta?.content ?? "") as string;
          if (!piece && !thinkingPiece) return;
          thoughts += thinkingPiece;
          if (piece && thoughts && thought === undefined) thought = (performance.now() - started) / 1000;
          text += piece;
          if (chatId.current === id) setMessages((m) => [...m.slice(0, -1), current()]);
        },
        {
          signal: ctrl.signal,
          json: {
            stream: true,
            messages: [...(system.trim() ? [{ role: "system", content: system }] : []), ...history.map(({ role, content }) => ({ role, content }))],
            ...sampling,
            ...thinkingFields(),
          },
        },
      );
    } catch (err) {
      if (!ctrl.signal.aborted) {
        failure = (err as Error).message;
        if (chatId.current === id) setMessages((m) => [...m.slice(0, -1), text || thoughts ? current() : { role: "assistant", content: failure!, error: true, model }]);
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
    if (thoughts && thought === undefined) thought = (performance.now() - started) / 1000;
    // A stopped reply keeps what was written (thinking included); an empty one isn't saved.
    const reply: Message | null = text || thoughts ? current() : failure ? { role: "assistant", content: failure, error: true, model } : null;
    if (reply && chatId.current === id) setMessages([...history, reply]);
    if (id) await persist(id, reply ? [...history, reply] : history);
    if (!reply && chatId.current === id) setMessages(history);
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
      <ChatList activeId={chat?.id ?? null} version={listVersion} onOpen={(id) => void openChat(id)} onNew={newChat} onDeleted={(id) => id === chatId.current && newChat()} />
      <div className="flex min-w-0 flex-1 flex-col">
        {chat?.source && (
          <div className="flex items-center gap-2 border-b border-line px-6 py-2 text-[12.5px] text-muted">
            <History className="size-3.5" /> Imported from {chat.source}. {ready ? "Messages you send continue it with the loaded model." : "Load a model to continue it."}
          </div>
        )}
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
                      {m.reasoning && <ThinkingBlock text={m.reasoning} live={busy && i === messages.length - 1 && !m.content} seconds={m.thought} />}
                      {m.content ? (
                        <Markdown remarkPlugins={[remarkGfm]}>{literalAsterisks(m.content)}</Markdown>
                      ) : !m.reasoning ? (
                        <span className="animate-pulse-dot text-muted">Thinking…</span>
                      ) : !(busy && i === messages.length - 1) ? (
                        <p className="text-[13px] text-muted">The model stopped while still thinking. Raise Max new tokens, or set a thinking budget, and regenerate.</p>
                      ) : null}
                    </div>
                  )}
                  {m.role === "assistant" && (m.content || m.reasoning) && !(busy && i === messages.length - 1) && (
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
          {saveError && <p className="mb-2 text-center text-[12px] text-danger">{saveError}</p>}
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
        </div>
        {recommended && (
          <p className="-mt-1 mb-4 text-[12px] text-muted" title={samplingText(recommended.sampling)}>
            {recommended.publisher === "Ollama"
              ? usingRecommended
                ? `Using the settings from ${recommended.id}'s Ollama Modelfile.`
                : `Changed from ${recommended.id}'s Ollama Modelfile settings.`
              : usingRecommended
                ? `Using ${recommended.publisher}'s recommended settings for ${shortName(recommended.id)}.`
                : `Changed from ${recommended.publisher}'s recommended settings for ${shortName(recommended.id)}.`}
          </p>
        )}
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
          <Slider label="Presence penalty" value={settings.presence_penalty} min={0} max={2} step={0.05} onChange={set("presence_penalty")} />
          <Field label="Max new tokens" hint={reasoning?.thinks ? "Thinking counts toward this." : undefined}>
            <Input type="number" min={16} value={settings.max_tokens} onChange={(e) => set("max_tokens")(Number(e.target.value))} />
          </Field>
          {reasoning && (reasoning.toggle || reasoning.effort || reasoning.thinks) && (
            <div className="flex flex-col gap-3 border-t border-line pt-4">
              <h3 className="flex items-center gap-1.5 text-[13px] font-semibold">
                <Brain className="size-3.5" /> Thinking
              </h3>
              {reasoning.toggle && <Toggle label="Think before answering" checked={thinking.on} onChange={(on) => changeThinking({ on })} />}
              {reasoning.effort && (
                <Field label="Reasoning effort">
                  <Select value={thinking.effort} onChange={(e) => changeThinking({ effort: e.target.value as Thinking["effort"] })}>
                    <option value="default">Model default</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </Select>
                </Field>
              )}
              {(!reasoning.toggle || thinking.on) && (
                <Field label="Thinking budget" hint="Most tokens to spend thinking before answering. Empty for no limit.">
                  <Input type="number" min={0} step={256} placeholder="No limit" value={thinking.budget || ""} onChange={(e) => changeThinking({ budget: Math.max(0, Number(e.target.value) || 0) })} />
                </Field>
              )}
              <p className="text-[12px] text-muted">Remembered for {runtime?.model}.</p>
            </div>
          )}
          {recommended ? (
            <Button size="sm" variant="ghost" disabled={usingRecommended} onClick={() => setSettings((s) => recommendedSettings(s, recommended))}>
              Use recommended settings
            </Button>
          ) : (
            <Button size="sm" variant="ghost" onClick={() => setSettings(DEFAULTS)}>
              Reset to defaults
            </Button>
          )}
        </div>
      </aside>
    </div>
  );
}
