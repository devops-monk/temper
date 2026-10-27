import { ArrowUp, Boxes, Brain, ChevronRight, Columns2, History, Layers, ListPlus, MessageSquare, Paperclip, RotateCcw, Square } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";
import { samplingText, type CatalogModel } from "../components/Catalog";
import { AttachmentChip, attach, attachmentPart, pickFiles, stored, type Attachment } from "../components/Attachments";
import { ChatList } from "../components/ChatList";
import { FindBar } from "../components/FindBar";
import { PromptLibrary } from "../components/PromptLibrary";
import { useT } from "../lib/i18n";
import { CHAT_CONTEXT } from "../components/LoadButton";
import { ModelSettings, type ReplyStats } from "../components/ModelSettings";
import { Compare } from "../components/Compare";
import { PresetBar, type Preset } from "../components/Presets";
import { PromptQueue, type Queued } from "../components/PromptQueue";
import { ReplyText, ThinkingBlock } from "../components/Reply";
import { Button, CopyButton, EmptyState, Field, Input, Notice, Select, Toggle, cx } from "../components/ui";
import { api, inTauri, stream } from "../lib/engine";
import { useApp } from "../lib/store";
import { useFileDrop } from "../lib/useFileDrop";

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
  /** Files sent with a user message. */
  attachments?: Attachment[];
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

/** The summary standing in for messages[:upto] once the thread outgrew the model's context (engine compaction.py). */
interface Compaction {
  upto: number;
  text: string;
}

interface Prepared {
  messages: unknown[];
  summary: Compaction | null;
  compacted: boolean;
  prompt: number;
  context: number;
  reserve: number;
}

function loadFlag(key: string, fallback: boolean): boolean {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch {
    return fallback;
  }
}

/** A message as llama.cpp takes it; attachments become parts the engine expands, and failed replies are marked for it to leave out. */
const toApi = ({ role, content, attachments, error }: Message) => ({ role, content: attachments?.length ? [...attachments.map(attachmentPart), ...(content ? [{ type: "text", text: content }] : [])] : content, ...(error && { error }) });

const kTokens = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));

interface SavedChat {
  id: string;
  title: string;
  source: string | null;
  messages: Message[];
  settings: Partial<Settings> | null;
  compaction: Compaction | null;
  /** A reply the engine is still writing (engine generations.py). */
  generating: Generation | null;
}

interface Generation {
  id: string;
  chat_id: string | null;
  upto: number;
}

interface Followed {
  text: string;
  thoughts: string;
  thought?: number;
  tokens: number;
  error: string | null;
  cancelled: boolean;
}

const fromSaved = (messages: Message[]): Message[] =>
  messages.map(({ role, content, error, model, reasoning, attachments }) => ({ role: role === "assistant" ? "assistant" : "user", content, error, model, reasoning, attachments: attachments ?? undefined }));

/** Leaving a chat stops following its reply without stopping it; the engine finishes and saves it. */
const DETACH = "detach";

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

/** Where recommended settings came from: a publisher (catalog), an Ollama Modelfile, or `source` (the GGUF, generation_config.json). */
type Recommended = Pick<CatalogModel, "id" | "sampling"> & { publisher: string | null; system?: string | null; source?: string | null };

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

/** Where the summary starts: the messages above it reach the model only as this summary. */
function CompactionDivider({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3 text-[12px] text-muted">
        <span className="h-px flex-1 bg-line" />
        <button onClick={() => setOpen(!open)} className="flex items-center gap-1 hover:text-text" aria-expanded={open}>
          <Layers className="size-3.5" /> Earlier messages are summarised to fit the model's context
          <ChevronRight className={cx("size-3 transition-transform", open && "rotate-90")} />
        </button>
        <span className="h-px flex-1 bg-line" />
      </div>
      {open && <p className="rounded-lg border border-line bg-surface px-4 py-3 text-[13px] leading-relaxed whitespace-pre-wrap text-muted">{text}</p>}
    </div>
  );
}

export function ChatPage() {
  const { runtime, go } = useApp();
  const t = useT();
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState(DEFAULTS);
  const [recommended, setRecommended] = useState<Recommended | null>(null);
  const abort = useRef<AbortController | null>(null);
  const generation = useRef<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const [finding, setFinding] = useState(false);
  const [library, setLibrary] = useState(false);
  // Typing / and a few letters in an empty message searches the prompt library; Esc leaves it as typed.
  const [slashDismissed, setSlashDismissed] = useState<string | null>(null);
  const slash = /^\/([^\s/]*)$/.exec(draft) && draft !== slashDismissed ? draft.slice(1) : null;
  const ready = runtime?.state === "ready";
  // The saved thread on screen; null until the first message of a new chat is sent.
  const [chat, setChat] = useState<{ id: string; title: string; source: string | null } | null>(null);
  const chatId = useRef<string | null>(null);
  const [listVersion, setListVersion] = useState(0);
  const [thinking, setThinking] = useState<Thinking>(THINKING_DEFAULTS);
  const reasoning = ready ? runtime?.reasoning : null;
  // Templates that mention <think> only to strip it from history don't make a model think, so without a
  // switch or effort level, thinking controls appear once the model has actually thought.
  const thinks = !!reasoning && (reasoning.toggle || reasoning.effort || messages.some((m) => m.reasoning && m.model === runtime?.model));
  useEffect(() => setThinking(loadThinking(runtime?.model ?? null)), [runtime?.model]);
  const changeThinking = (next: Partial<Thinking>) =>
    setThinking((t) => {
      const value = { ...t, ...next };
      saveThinking(runtime?.model ?? null, value);
      return value;
    });
  const [saveError, setSaveError] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);
  // Auto-compaction: the thread's summary (kept in a ref too, for the queue's send loop) and the context used.
  const [compaction, setCompactionState] = useState<Compaction | null>(null);
  const compactionRef = useRef<Compaction | null>(null);
  const setCompaction = (c: Compaction | null) => {
    compactionRef.current = c;
    setCompactionState(c);
  };
  const [usage, setUsage] = useState<{ tokens: number; context: number } | null>(null);
  const [compacting, setCompacting] = useState(false);
  const [stats, setStats] = useState<ReplyStats | null>(null);
  const context = runtime?.context ?? CHAT_CONTEXT;
  const [autoCompact, setAutoCompact] = useState(() => loadFlag("temper.autoCompact", true));
  const changeAutoCompact = (on: boolean) => {
    setAutoCompact(on);
    try {
      localStorage.setItem("temper.autoCompact", on ? "1" : "0");
    } catch {
      /* private window */
    }
  };
  // Follow-ups typed while a reply is written; the ref is what the send loop reads.
  const [queue, setQueueState] = useState<Queued[]>([]);
  const queueRef = useRef<Queued[]>([]);
  const nextQueued = useRef(0);
  const [paused, setPaused] = useState(false);
  const setQueue = (items: Queued[]) => {
    queueRef.current = items;
    setQueueState(items);
    if (!items.length) setPaused(false);
  };

  /** Saves the whole thread to the chat it belongs to, creating the chat on its first message. */
  async function persist(id: string | null, history: Message[]): Promise<string | null> {
    const body = { messages: history.map(({ role, content, error, model, reasoning, attachments }) => ({ role, content, error: !!error, model: model ?? null, reasoning: reasoning || null, attachments: attachments?.length ? attachments.map(stored) : null })), model: runtime?.model ?? null, settings, compaction: compactionRef.current && compactionRef.current.upto <= history.length ? compactionRef.current : null };
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
    setComparing(false);
    if (id === chatId.current) return;
    setQueue([]);
    abort.current?.abort(DETACH);
    const saved = await api<SavedChat>(`/api/chats/${id}`).catch((e: Error) => {
      setSaveError(e.message);
      return null;
    });
    if (!saved) return;
    chatId.current = id;
    setChat({ id, title: saved.title, source: saved.source });
    setCompaction(saved.compaction);
    setUsage(null);
    setMessages(fromSaved(saved.messages));
    // The thread's own system prompt comes back with it; sampling stays with the loaded model.
    setSettings((s) => ({ ...s, system: saved.settings?.system ?? "" }));
    if (saved.generating) void rejoin(saved.generating, fromSaved(saved.messages));
  }

  /** Follows a reply that was being written when the chat was left, from its first event, then shows it as saved. */
  async function rejoin(gen: Generation, saved: Message[]) {
    const id = gen.chat_id;
    const ctrl = new AbortController();
    abort.current = ctrl;
    generation.current = gen.id;
    setBusy(true);
    setMessages([...saved.slice(0, gen.upto), { role: "assistant", content: "" }]);
    try {
      await follow(gen.id, id, null, ctrl.signal);
    } catch {
      // finished between opening the chat and following it: the saved chat has the reply
    } finally {
      setBusy(false);
      generation.current = null;
      abort.current = null;
    }
    if (ctrl.signal.reason === DETACH || !id) return;
    const done = await api<SavedChat>(`/api/chats/${id}`).catch(() => null);
    if (done && chatId.current === id) setMessages(fromSaved(done.messages));
    setListVersion((v) => v + 1);
  }

  /** Stops the reply: the engine keeps what it wrote. */
  function stop() {
    if (generation.current) void api(`/api/chat/generations/${generation.current}/cancel`, { method: "POST" }).catch(() => abort.current?.abort());
    else abort.current?.abort();
  }

  // Keyboard shortcuts (lib/shortcuts.ts); the ref keeps the handlers current without re-subscribing.
  const keys = useRef({ newChat: () => {}, stop: () => {}, comparing: false, busy: false });
  useEffect(() => {
    const find = () => !keys.current.comparing && setFinding(true);
    const fresh = () => keys.current.newChat();
    const focus = () => input.current?.focus();
    const stop = (e: KeyboardEvent) => e.key === "Escape" && keys.current.busy && keys.current.stop();
    window.addEventListener("temper:find", find);
    window.addEventListener("temper:new-chat", fresh);
    window.addEventListener("temper:focus-input", focus);
    window.addEventListener("keydown", stop);
    return () => {
      window.removeEventListener("temper:find", find);
      window.removeEventListener("temper:new-chat", fresh);
      window.removeEventListener("temper:focus-input", focus);
      window.removeEventListener("keydown", stop);
    };
  }, []);

  function newChat() {
    setComparing(false);
    setQueue([]);
    setCompaction(null);
    setUsage(null);
    abort.current?.abort(DETACH);
    chatId.current = null;
    setChat(null);
    setMessages([]);
  }

  // When a model loads: its default preset if it has one, else its recommended settings (catalog,
  // Ollama Modelfile, the GGUF or generation_config.json), else Temper's defaults.
  const loaded = ready ? (runtime?.path ?? runtime?.model ?? null) : null;
  const [presets, setPresets] = useState<Preset[]>([]);
  const [defaultPreset, setDefaultPreset] = useState<string | null>(null);
  const [selected, setSelected] = useState("defaults");
  const presetsFor = (model: string | null) => api<{ presets: Preset[]; default: string | null }>(`/api/presets${model ? `?model=${encodeURIComponent(model)}` : ""}`);
  useEffect(() => {
    if (!loaded) return;
    let stale = false;
    const model = runtime?.model ?? null;
    void Promise.all([
      api<{ match: Recommended | null }>(`/api/catalog/match?model=${encodeURIComponent(loaded)}`).catch(() => ({ match: null })),
      presetsFor(model).catch(() => ({ presets: [], default: null })),
    ]).then(([{ match }, saved]) => {
      if (stale) return;
      const rec = match && Object.keys(match.sampling).length ? match : null;
      setRecommended(rec);
      setPresets(saved.presets);
      setDefaultPreset(saved.default);
      const preset = saved.presets.find((p) => p.id === saved.default);
      if (preset) {
        applyPreset(preset);
        setSelected(preset.id);
        return;
      }
      setSelected(rec ? "recommended" : "defaults");
      setSettings((s) => {
        const next = rec ? recommendedSettings(s, rec) : { ...s, ...Object.fromEntries(SAMPLING_KEYS.map((k) => [k, DEFAULTS[k]])) };
        // A system prompt that came with the model (an Ollama Modelfile) fills an empty one, never replaces yours.
        return match?.system && !s.system.trim() ? { ...next, system: match.system } : next;
      });
    });
    return () => {
      stale = true;
    };
    // runtime.model changes together with `loaded`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded]);

  function applyPreset(preset: Preset) {
    setSettings({ ...DEFAULTS, ...(preset.settings as Partial<Settings>) });
    if (preset.thinking) changeThinking(preset.thinking as Partial<Thinking>);
  }

  function choose(value: string) {
    setSelected(value);
    if (value === "recommended" && recommended) setSettings((s) => recommendedSettings(s, recommended));
    else if (value === "defaults") setSettings((s) => ({ ...DEFAULTS, system: s.system }));
    else {
      const preset = presets.find((p) => p.id === value);
      if (preset) applyPreset(preset);
    }
  }

  async function reloadPresets(select?: string) {
    const saved = await presetsFor(runtime?.model ?? null).catch(() => null);
    if (!saved) return;
    setPresets(saved.presets);
    setDefaultPreset(saved.default);
    if (select) setSelected(select);
    if (select === "recommended" || select === "defaults") choose(select);
  }

  const usingRecommended = recommended !== null && SAMPLING_KEYS.every((k) => settings[k] === recommendedSettings(settings, recommended)[k]);
  const chosenPreset = presets.find((p) => p.id === selected);
  const modified =
    selected === "recommended"
      ? !usingRecommended
      : selected === "defaults"
        ? [...SAMPLING_KEYS, "max_tokens" as const].some((k) => settings[k] !== DEFAULTS[k])
        : !!chosenPreset && (Object.keys(DEFAULTS) as (keyof Settings)[]).some((k) => settings[k] !== ({ ...DEFAULTS, ...chosenPreset.settings } as Settings)[k]);

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

  /** Follows a generation's events into the chat's last message; resolves with what it wrote and how it ended. */
  async function follow(gen: string, id: string | null, model: string | null, signal: AbortSignal): Promise<Followed> {
    const out: Followed = { text: "", thoughts: "", tokens: 0, error: null, cancelled: false };
    const started = performance.now();
    const current = (): Message => ({ role: "assistant", content: out.text, model, reasoning: out.thoughts || null, thought: out.thought });
    await stream(
      `/api/chat/generations/${gen}/stream`,
      (data) => {
        const chunk = JSON.parse(data);
        if (chunk.temper_end) {
          out.error = chunk.temper_end.error;
          out.cancelled = chunk.temper_end.cancelled;
          return;
        }
        const t = chunk.timings;
        if (t?.predicted_n) {
          out.tokens = t.predicted_n;
          setStats({ speed: t.predicted_per_second, drafted: t.draft_n, accepted: t.draft_n_accepted });
        }
        const delta = chunk.choices?.[0]?.delta;
        const thinkingPiece = (delta?.reasoning_content ?? delta?.reasoning ?? "") as string;
        const piece = (delta?.content ?? "") as string;
        if (!piece && !thinkingPiece) return;
        out.thoughts += thinkingPiece;
        if (piece && out.thoughts && out.thought === undefined) out.thought = (performance.now() - started) / 1000;
        out.text += piece;
        if (chatId.current === id) setMessages((m) => [...m.slice(0, -1), current()]);
      },
      { signal },
    );
    if (out.thoughts && out.thought === undefined) out.thought = (performance.now() - started) / 1000;
    return out;
  }

  /** Streams a reply to `history`; returns the thread with it, or null if it was stopped, failed or the chat changed.
   * The engine writes the reply (engine generations.py) and saves it, so it survives the window closing or the chat changing. */
  async function send(history: Message[]): Promise<Message[] | null> {
    setMessages([...history, { role: "assistant", content: "" }]);
    setBusy(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const { system, ...sampling } = settings;
    const model = runtime?.model ?? null;
    let got: Followed | null = null;
    let started = false;
    let failure: string | null = null;
    // A summary of turns that regenerating or editing removed no longer applies.
    if (compactionRef.current && compactionRef.current.upto > history.length) setCompaction(null);
    // Saved before the reply streams, so a long answer or a crash never loses the question.
    const id = await persist(chatId.current, history);
    try {
      // Fits the thread into the model's context, summarising its oldest turns if it has to.
      const ask = { messages: history.map(toApi), system, summary: compactionRef.current, max_tokens: settings.max_tokens, compact: autoCompact };
      const check = await api<{ fits: boolean }>("/api/chat/prepare", { json: { ...ask, check: true }, signal: ctrl.signal });
      if (!check.fits && autoCompact) setCompacting(true);
      const prepared = await api<Prepared>("/api/chat/prepare", { json: ask, signal: ctrl.signal }).finally(() => setCompacting(false));
      if (chatId.current === id) setCompaction(prepared.summary);
      const body = { messages: prepared.messages, ...sampling, ...thinkingFields() };
      const gen = await api<Generation>("/api/chat/generations", { json: { chat_id: id, upto: history.length, model, body }, signal: ctrl.signal });
      started = true;
      generation.current = gen.id;
      got = await follow(gen.id, id, model, ctrl.signal);
      if (got.error && !got.text && !got.thoughts) failure = got.error;
      if (chatId.current === id) setUsage({ tokens: prepared.prompt + got.tokens, context: prepared.context });
    } catch (err) {
      if (!ctrl.signal.aborted) failure = (err as Error).message;
    } finally {
      setBusy(false);
      abort.current = null;
      generation.current = null;
    }
    // Another chat was opened: the engine finishes this reply and saves it.
    if (ctrl.signal.reason === DETACH) return null;
    const stopped = ctrl.signal.aborted || !!got?.cancelled;
    const text = got?.text ?? "";
    const thoughts = got?.thoughts ?? "";
    // A stopped reply keeps what was written (thinking included); one stopped before it began isn't saved.
    // A model that ends without writing anything says so, rather than leaving the question unanswered.
    if (!text && !thoughts && !failure && !stopped) failure = "The model ended its reply without writing anything. Regenerate, or rephrase the message.";
    const reply: Message | null = text || thoughts ? { role: "assistant", content: text, model, reasoning: thoughts || null, thought: got?.thought } : failure ? { role: "assistant", content: failure, error: true, model } : null;
    if (reply && chatId.current === id) setMessages([...history, reply]);
    // The engine saved the reply if it got to write it; otherwise (it failed before starting) the page does.
    if (id && !started) await persist(id, reply ? [...history, reply] : history);
    else setListVersion((v) => v + 1);
    if (!reply && chatId.current === id) setMessages(history);
    return reply && !failure && !stopped && chatId.current === id ? [...history, reply] : null;
  }

  /** Sends, then the queued follow-ups one by one; a stopped or failed reply pauses the queue. */
  async function run(history: Message[]) {
    setPaused(false);
    let thread = await send(history);
    while (thread && queueRef.current.length) {
      const [item, ...rest] = queueRef.current;
      setQueue(rest);
      thread = await send([...thread, { role: "user", content: item.text, attachments: item.files }]);
    }
    if (!thread && queueRef.current.length) setPaused(true);
  }

  function resume() {
    const [item, ...rest] = queueRef.current;
    if (!item || busy || !ready) return;
    setQueue(rest);
    void run([...messages, { role: "user", content: item.text, attachments: item.files }]);
  }

  function submit() {
    const content = draft.trim();
    if ((!content && !files.length) || !ready || attaching || cantTake) return;
    setDraft("");
    setFiles([]);
    const sent = files.length ? files : undefined;
    if (busy) setQueue([...queueRef.current, { id: nextQueued.current++, text: content, files: sent }]);
    else void run([...messages, { role: "user", content, attachments: sent }]);
  }

  // Files waiting in the message box; they're copied into Temper's home as soon as they're added.
  const [files, setFiles] = useState<Attachment[]>([]);
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);
  async function addFiles(picked: (string | File)[]) {
    if (!picked.length) return;
    setAttaching(true);
    const { added, errors } = await attach(picked);
    setAttaching(false);
    setFiles((f) => [...f, ...added]);
    setAttachError(errors.length ? errors.join(" ") : null);
  }
  function removeFile(a: Attachment) {
    setFiles((f) => f.filter((x) => x.id !== a.id));
    void api(`/api/attachments/${a.id}`, { method: "DELETE" }).catch(() => {}); // never sent, so nothing refers to it
  }
  const dropping = useFileDrop((paths) => !comparing && ready && void addFiles(paths));
  const modalities = runtime?.modalities ?? {};
  const kinds = new Set(files.map((f) => f.kind));
  const lacking = kinds.has("image") && !modalities.vision ? "see images" : kinds.has("audio") && !modalities.audio ? "hear audio" : null;
  const cantTake =
    ready && lacking
      ? `${runtime?.model} can't ${lacking}. ${runtime?.projector ? "" : "Load a model that can; Temper loads its projector (an mmproj .gguf) from beside the model."}`.trim()
      : null;

  keys.current = { newChat, stop, comparing, busy };
  const set = <K extends keyof Settings>(k: K) => (v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }));

  return (
    <div className="flex min-h-0 flex-1">
      <ChatList activeId={chat?.id ?? null} version={listVersion} onOpen={(id) => void openChat(id)} onNew={newChat} onDeleted={(id) => id === chatId.current && newChat()} />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-line px-6 py-2">
          <div className="flex rounded-lg bg-surface-2 p-0.5 text-[12.5px]" role="tablist" aria-label={t("chat.mode")}>
            {([[t("chat.tab_chat"), MessageSquare, false], [t("chat.tab_compare"), Columns2, true]] as const).map(([label, Icon, value]) => (
              <button
                key={label}
                role="tab"
                aria-selected={comparing === value}
                onClick={() => setComparing(value)}
                disabled={busy}
                className={cx("flex items-center gap-1.5 rounded-md px-2.5 py-1", comparing === value ? "bg-surface font-medium shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:text-text")}
              >
                <Icon className="size-3.5" /> {label}
              </button>
            ))}
          </div>
          {chat?.source && !comparing && (
            <span className="flex min-w-0 items-center gap-2 text-[12.5px] text-muted">
              <History className="size-3.5 shrink-0" /> <span className="truncate">Imported from {chat.source}. {ready ? "Messages you send continue it with the loaded model." : "Load a model to continue it."}</span>
            </span>
          )}
        </div>
        {comparing ? (
          <Compare settings={settings} />
        ) : (
          <>
            <div className="relative flex min-h-0 flex-1 flex-col">
            {finding && <FindBar root={scroller} onClose={() => setFinding(false)} />}
            <div ref={scroller} className={cx("min-h-0 flex-1 overflow-y-auto", finding && "pt-10")}>
              {messages.length === 0 ? (
                ready ? (
                  <EmptyState icon={<MessageSquare className="size-8" />} title={t("chat.with", { model: runtime?.model ?? "" })}>
                    {t("chat.private")}
                  </EmptyState>
                ) : (
                  <EmptyState icon={<Boxes className="size-8" />} title={runtime?.state === "loading" ? t("chat.loading_model", { model: runtime.model ?? "" }) : t("chat.load_first")}>
                    {runtime?.state === "loading" ? (
                      t("chat.loading_hint")
                    ) : (
                      <>
                        <p>{t("chat.load_hint")}</p>
                        <Button variant="primary" className="mt-4" onClick={() => go("models")}>
                          {t("chat.open_models")}
                        </Button>
                      </>
                    )}
                  </EmptyState>
                )
              ) : (
                <div className="mx-auto flex max-w-[760px] flex-col gap-6 px-6 py-8">
                  {messages.map((m, i) => (
                    <Fragment key={i}>
                      {compaction && i === compaction.upto && <CompactionDivider text={compaction.text} />}
                      <div className={cx("group", m.role === "user" && "self-end", compaction && i < compaction.upto && "opacity-60")}>
                        {m.role === "user" ? (
                          <div className="flex max-w-[620px] flex-col items-end gap-2">
                            {m.attachments?.length ? (
                              <div className="flex flex-wrap items-start justify-end gap-2">
                                {m.attachments.map((a) => (
                                  <AttachmentChip key={a.id} attachment={a} context={context} />
                                ))}
                              </div>
                            ) : null}
                            {m.content && <div className="prose-chat rounded-2xl rounded-br-md bg-surface px-4 py-2.5 whitespace-pre-wrap shadow-[0_0_0_1px_var(--line)]">{m.content}</div>}
                          </div>
                        ) : m.error ? (
                          <Notice tone="error">{m.content}</Notice>
                        ) : (
                          <div className="prose-chat leading-relaxed">
                            {m.reasoning && <ThinkingBlock text={m.reasoning} live={busy && i === messages.length - 1 && !m.content} seconds={m.thought} />}
                            {m.content ? (
                              <ReplyText text={m.content} />
                            ) : !m.reasoning ? (
                              <span className="animate-pulse-dot text-muted">{compacting && i === messages.length - 1 ? t("chat.summarising") : t("chat.thinking")}</span>
                            ) : !(busy && i === messages.length - 1) ? (
                              <p className="text-[13px] text-muted">{t("chat.stopped_thinking")}</p>
                            ) : null}
                          </div>
                        )}
                        {m.role === "assistant" && (m.content || m.reasoning) && !(busy && i === messages.length - 1) && (
                          <div className="mt-1 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                            <CopyButton text={m.content} />
                            {i === messages.length - 1 && (
                              <Button size="sm" variant="ghost" onClick={() => void run(messages.slice(0, -1))}>
                                <RotateCcw className="size-3.5" /> {t("chat.regenerate")}
                              </Button>
                            )}
                          </div>
                        )}
                      </div>
                    </Fragment>
                  ))}
                </div>
              )}
            </div>
            </div>

            <div className="mx-auto w-full max-w-[760px] px-6 pb-5">
              {saveError && <p className="mb-2 text-center text-[12px] text-danger">{saveError}</p>}
              <PromptQueue items={queue} paused={paused && !busy} onChange={setQueue} onResume={resume} />
              {cantTake && <p className="mb-2 text-center text-[12px] text-danger">{cantTake}</p>}
              {attachError && <p className="mb-2 text-center text-[12px] text-danger">{attachError}</p>}
              <div
                className={cx("flex flex-col gap-2 rounded-2xl border bg-surface p-2 focus-within:border-accent", dropping && ready ? "border-accent border-dashed" : "border-line")}
                onDragOver={(e) => e.dataTransfer.types.includes("Files") && e.preventDefault()}
                onDrop={(e) => {
                  // The desktop app gets paths through useFileDrop; a browser only gets the files.
                  if (inTauri || !e.dataTransfer.files.length) return;
                  e.preventDefault();
                  if (ready) void addFiles([...e.dataTransfer.files]);
                }}
              >
                {(files.length > 0 || attaching) && (
                  <div className="flex flex-wrap items-start gap-2">
                    {files.map((a) => (
                      <AttachmentChip key={a.id} attachment={a} context={context} onRemove={() => removeFile(a)} />
                    ))}
                    {attaching && <span className="animate-pulse-dot self-center px-1 text-[12px] text-muted">{t("chat.reading")}</span>}
                  </div>
                )}
                <div className="flex items-end gap-2">
                  <Button variant="ghost" onClick={() => void pickFiles().then(addFiles)} disabled={!ready} aria-label={t("chat.attach")} title={t("chat.attach_hint")} className="size-9 rounded-xl px-0 text-muted">
                    <Paperclip className="size-[18px]" />
                  </Button>
                  <PromptLibrary
                    open={library || slash !== null}
                    onOpenChange={(open) => {
                      setLibrary(open && slash === null);
                      if (!open && slash !== null) setSlashDismissed(draft);
                    }}
                    query={slash}
                    draft={draft}
                    onUse={(text) => {
                      setDraft((d) => (slash !== null || !d.trim() ? text : `${d.trimEnd()}\n\n${text}`));
                      setTimeout(() => input.current?.focus());
                    }}
                  />
                  <textarea
                    ref={input}
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                        e.preventDefault();
                        submit();
                      }
                    }}
                    onPaste={(e) => {
                      const pasted = [...e.clipboardData.files];
                      if (!pasted.length || !ready) return;
                      e.preventDefault();
                      void addFiles(pasted);
                    }}
                    rows={Math.min(8, Math.max(1, draft.split("\n").length))}
                    placeholder={!ready ? t("chat.placeholder_first") : busy ? t("chat.placeholder_queue") : t("chat.placeholder")}
                    disabled={!ready}
                    className="max-h-52 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 outline-none placeholder:text-muted/70"
                    aria-label="Message"
                  />
                  {busy && (
                    <Button variant="secondary" onClick={stop} aria-label={t("chat.stop")} className="size-9 rounded-xl px-0">
                      <Square className="size-3.5 fill-current" />
                    </Button>
                  )}
                  {busy ? (
                    (draft.trim() || files.length > 0) && (
                      <Button variant="primary" onClick={submit} aria-label="Add to queue" title="Sent when the reply finishes" className="size-9 rounded-xl px-0">
                        <ListPlus className="size-4" />
                      </Button>
                    )
                  ) : (
                    <Button variant="primary" onClick={submit} disabled={(!draft.trim() && !files.length) || !ready || attaching || !!cantTake} aria-label={t("chat.send")} className="size-9 rounded-xl px-0">
                      <ArrowUp className="size-4" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="mt-1.5 flex justify-center gap-3 text-[11.5px] text-muted">
                <span>{busy ? t("chat.queue_hint") : t("chat.send_hint")}</span>
                {usage && (
                  <span className={cx("tabular-nums", usage.tokens > usage.context * 0.75 && "text-straw")} title={autoCompact ? "When it fills, older messages are summarised to make room." : "Auto-compaction is off: a full context stops the chat."}>
                    {t("chat.tokens", { used: kTokens(usage.tokens), total: kTokens(usage.context) })}
                  </span>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <aside className="w-[272px] shrink-0 overflow-y-auto border-l border-line bg-surface/40 px-4 py-5">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold">Generation</h2>
        </div>
        <PresetBar
          model={ready ? (runtime?.model ?? null) : null}
          presets={presets}
          selected={selected}
          modified={modified}
          defaultId={defaultPreset}
          hasRecommended={recommended !== null}
          current={{ settings, thinking }}
          onSelect={choose}
          onReload={(id) => void reloadPresets(id)}
        />
        {recommended && selected === "recommended" && (
          <p className="-mt-2 mb-4 text-[12px] text-muted" title={samplingText(recommended.sampling)}>
            {recommended.publisher === "Ollama"
              ? `From ${recommended.id}'s Ollama Modelfile.`
              : recommended.source
                ? `From ${recommended.source}.`
                : `${recommended.publisher}'s recommended settings for ${shortName(recommended.id)}.`}
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
          <Field label="Max new tokens" hint={thinks ? "Thinking counts toward this." : undefined}>
            <Input type="number" min={16} value={settings.max_tokens} onChange={(e) => set("max_tokens")(Number(e.target.value))} />
          </Field>
          <div>
            <Toggle label="Auto-compact long chats" checked={autoCompact} onChange={changeAutoCompact} />
            <p className="text-[12px] text-muted">When the model's context fills, summarise the oldest messages instead of stopping.</p>
          </div>
          {reasoning && thinks && !comparing && (
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
          {!comparing && <ModelSettings stats={stats} />}
        </div>
      </aside>
    </div>
  );
}
