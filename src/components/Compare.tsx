import { ArrowUp, Columns2, Plus, RotateCcw, Square, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, stream } from "../lib/engine";
import { quantLabel } from "../lib/format";
import { useApp } from "../lib/store";
import type { LocalModel, RuntimeStatus } from "../lib/types";
import { ReplyText, ThinkingBlock } from "./Reply";
import { Button, CopyButton, EmptyState, Notice, Select, Toggle } from "./ui";

const MAX_MODELS = 4;

interface Option {
  group: "Fine-tuned" | "Models";
  path: string;
  /** What the engine calls the model, as LoadButton names it. */
  name: string;
  label: string;
}

interface Reply {
  status: "waiting" | "loading" | "writing" | "done" | "stopped" | "failed";
  content: string;
  reasoning: string;
  thought?: number;
  error?: string;
  /** llama-server's timings for the reply. */
  speed?: number;
  tokens?: number;
  firstToken?: number;
}

interface Turn {
  prompt: string;
  replies: Reply[];
}

const EMPTY: Reply = { status: "waiting", content: "", reasoning: "" };

/** Every complete GGUF on this computer: exported fine-tunes, then downloaded, imported and converted models. */
function ggufOptions(exports: { name: string; gguf?: string[] }[], models: LocalModel[]): Option[] {
  const fileName = (path: string) => path.split(/[\\/]/).pop()!;
  return [
    ...exports.flatMap((e) => (e.gguf ?? []).map((path): Option => ({ group: "Fine-tuned", path, name: fileName(path), label: e.gguf!.length > 1 ? `${e.name} · ${quantLabel(path)}` : e.name }))),
    ...models.flatMap((m) =>
      m.gguf
        .filter((f) => f.complete && f.path)
        .map((f): Option => ({ group: "Models", path: f.path!, name: fileName(f.name), label: m.imported || m.gguf.length === 1 ? m.id : `${m.id} · ${quantLabel(f.name)}` })),
    ),
  ];
}

interface CompareStatus {
  servers: RuntimeStatus[];
}

/** Waits until compare mode's servers for `paths` are ready; throws a server's error if it fails to load. */
async function waitReady(paths: string[], signal: AbortSignal) {
  while (!signal.aborted) {
    const { servers } = await api<CompareStatus>("/api/compare");
    const mine = servers.filter((s) => s.path && paths.includes(s.path));
    const failed = mine.find((s) => s.state === "failed" || s.state === "idle");
    if (failed) throw new Error(failed.error ?? `${failed.model} stopped loading.`);
    if (mine.length === paths.length && mine.every((s) => s.state === "ready")) return;
    await new Promise((r) => setTimeout(r, 500));
  }
}

function Stats({ reply }: { reply: Reply }) {
  const parts = [
    reply.speed !== undefined && `${reply.speed.toFixed(1)} tokens/s`,
    reply.tokens !== undefined && `${reply.tokens} tokens`,
    reply.firstToken !== undefined && `${reply.firstToken < 1000 ? `${Math.round(reply.firstToken)} ms` : `${(reply.firstToken / 1000).toFixed(1)} s`} to first token`,
  ].filter(Boolean);
  return parts.length ? <span className="text-[11.5px] text-muted tabular-nums">{parts.join(", ")}</span> : null;
}

/** A failure's first line, with llama-server's log folded away under it. */
function FailedReply({ error }: { error: string }) {
  const [reason, ...log] = error.split("\n");
  return (
    <Notice tone="error">
      {reason}
      {log.length > 0 && (
        <details className="mt-1.5 text-[11.5px]">
          <summary className="cursor-pointer opacity-80">Details</summary>
          <pre className="mt-1 max-h-48 overflow-auto font-mono break-all whitespace-pre-wrap">{log.join("\n")}</pre>
        </details>
      )}
    </Notice>
  );
}

function ReplyCard({ reply }: { reply: Reply }) {
  const live = reply.status === "writing";
  return (
    <div className="group flex min-w-0 flex-col rounded-xl border border-line bg-surface px-4 py-3">
      {reply.status === "failed" ? (
        <FailedReply error={reply.error ?? "Something went wrong."} />
      ) : (
        <div className="prose-chat min-w-0 leading-relaxed">
          {reply.reasoning && <ThinkingBlock text={reply.reasoning} live={live && !reply.content} seconds={reply.thought} />}
          {reply.content ? (
            <ReplyText text={reply.content} />
          ) : reply.status === "waiting" ? (
            <span className="text-[13px] text-muted">Waiting for its turn…</span>
          ) : reply.status === "loading" ? (
            <span className="animate-pulse-dot text-[13px] text-muted">Loading the model…</span>
          ) : live && !reply.reasoning ? (
            <span className="animate-pulse-dot text-muted">Thinking…</span>
          ) : reply.status === "stopped" && !reply.reasoning ? (
            <span className="text-[13px] text-muted">Stopped before it answered.</span>
          ) : !live && reply.reasoning ? (
            <p className="text-[13px] text-muted">The model stopped while still thinking. Raise Max new tokens and try again.</p>
          ) : null}
        </div>
      )}
      {(reply.status === "done" || reply.status === "stopped") && (
        <div className="mt-2 flex items-center justify-between gap-2">
          <Stats reply={reply} />
          {reply.content && (
            <span className="opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
              <CopyButton text={reply.content} />
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * One prompt to two to four models side by side, for example a base model and its fine-tune. Each column
 * keeps its own thread; the Generation panel's system prompt and sampling apply to all of them.
 */
export function Compare({ settings }: { settings: { system: string } }) {
  const { system, ...sampling } = settings;
  const { runtime, go } = useApp();
  const [options, setOptions] = useState<Option[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [think, setThink] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const chatPath = runtime?.state === "ready" ? (runtime.path ?? null) : null;

  useEffect(() => {
    Promise.all([api<{ items: { kind: string; name: string; gguf?: string[] }[] }>("/api/files"), api<LocalModel[]>("/api/models/local")]).then(
      ([files, models]) => {
        const found = ggufOptions(files.items.filter((i) => i.kind === "export"), models);
        setOptions(found);
        // Starts with chat's model, if one is loaded, against the next model on the list.
        const first = found.find((o) => o.path === chatPath)?.path ?? found[0]?.path;
        setColumns([first, found.find((o) => o.path !== first)?.path].filter((p): p is string => !!p));
      },
      () => setOptions([]),
    );
    // Compare mode's models go when it closes, so chat has the memory back.
    return () => {
      abort.current?.abort();
      void api("/api/compare/unload", { method: "POST" }).catch(() => {});
    };
    // chatPath only picks the first column
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const option = (path: string) => options?.find((o) => o.path === path);

  function changeColumns(next: string[]) {
    if (turns.length && !confirm("Changing the models starts a new comparison. Clear this one?")) return;
    setTurns([]);
    setColumns(next);
  }

  const update = (turn: number, column: number, change: Partial<Reply>) =>
    setTurns((all) => all.map((t, i) => (i === turn ? { ...t, replies: t.replies.map((r, j) => (j === column ? { ...r, ...change } : r)) } : t)));

  async function ask(history: Turn[], turn: number, column: number, signal: AbortSignal) {
    const path = columns[column];
    const messages = [
      ...(system.trim() ? [{ role: "system", content: system }] : []),
      ...history.flatMap((t) => {
        const reply = t.replies[column];
        return [{ role: "user", content: t.prompt }, ...(reply?.content ? [{ role: "assistant", content: reply.content }] : [])];
      }),
    ];
    let content = "";
    let reasoning = "";
    let thought: number | undefined;
    const started = performance.now();
    update(turn, column, { status: "writing" });
    try {
      await stream(
        "/v1/chat/completions",
        (data) => {
          if (data === "[DONE]") return;
          const chunk = JSON.parse(data);
          const t = chunk.timings;
          if (t) update(turn, column, { speed: t.predicted_per_second, tokens: t.predicted_n, firstToken: t.prompt_ms });
          const delta = chunk.choices?.[0]?.delta;
          const thinking = (delta?.reasoning_content ?? "") as string;
          const piece = (delta?.content ?? "") as string;
          if (!piece && !thinking) return;
          reasoning += thinking;
          if (piece && reasoning && thought === undefined) thought = (performance.now() - started) / 1000;
          content += piece;
          update(turn, column, { content, reasoning, thought });
        },
        {
          signal,
          json: {
            stream: true,
            model: path,
            messages,
            ...sampling,
            // Templates without the switch ignore it, so it's safe to send to every model.
            ...(!think && { chat_template_kwargs: { enable_thinking: false } }),
          },
        },
      );
      if (reasoning && thought === undefined) thought = (performance.now() - started) / 1000;
      update(turn, column, { status: "done", thought });
    } catch (e) {
      update(turn, column, signal.aborted ? { status: "stopped" } : { status: "failed", error: (e as Error).message });
    }
  }

  async function send(prompt: string, history: Turn[]) {
    const turn = history.length;
    setTurns([...history, { prompt, replies: columns.map(() => EMPTY) }]);
    setBusy(true);
    setNotice(null);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const models = columns.map((path) => ({ path, name: option(path)?.name ?? null }));
    const fail = (column: number, e: unknown) => !ctrl.signal.aborted && update(turn, column, { status: "failed", error: (e as Error).message });
    try {
      const { together } = await api<{ together: boolean }>("/api/compare/load", { json: { models } });
      if (together) {
        // Each column starts as soon as its own model is ready; one that fails to load doesn't hold up the rest.
        await Promise.all(
          columns.map(async (path, i) => {
            if (path !== chatPath) {
              update(turn, i, { status: "loading" });
              try {
                await waitReady([path], ctrl.signal);
              } catch (e) {
                return fail(i, e);
              }
            }
            await ask(history, turn, i, ctrl.signal);
          }),
        );
      } else {
        setNotice("These models don't fit in memory together, so they answer one at a time.");
        for (let i = 0; i < columns.length && !ctrl.signal.aborted; i++) {
          if (columns[i] !== chatPath) {
            update(turn, i, { status: "loading" });
            try {
              await api("/api/compare/load", { json: { models: [models[i]] } });
              await waitReady([columns[i]], ctrl.signal);
            } catch (e) {
              fail(i, e);
              continue;
            }
          }
          await ask(history, turn, i, ctrl.signal);
        }
      }
    } catch (e) {
      columns.forEach((_, i) => fail(i, e));
    } finally {
      // Replies that never started (stopped while loading or waiting) say so.
      setTurns((all) => all.map((t, i) => (i === turn ? { ...t, replies: t.replies.map((r) => (r.status === "waiting" || r.status === "loading" ? { ...r, status: "stopped" } : r)) } : t)));
      setBusy(false);
      abort.current = null;
    }
  }

  function submit() {
    const prompt = draft.trim();
    if (!prompt || busy || columns.length < 2) return;
    setDraft("");
    void send(prompt, turns);
  }

  if (!options) return <p className="px-7 py-5 text-muted">Finding your models…</p>;
  if (options.length < 2)
    return (
      <EmptyState icon={<Columns2 className="size-8" />} title="Compare needs two models">
        <p>Download or import another GGUF model, or export a fine-tuned one, to compare them side by side.</p>
        <Button variant="primary" className="mt-4" onClick={() => go("models")}>
          Open models
        </Button>
      </EmptyState>
    );

  const grid = { gridTemplateColumns: `repeat(${columns.length}, minmax(0, 1fr))` };
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-start gap-3 border-b border-line px-6 py-3">
        <div className="grid min-w-0 flex-1 gap-3" style={grid}>
          {columns.map((path, i) => (
            <div key={i} className="flex min-w-0 items-center gap-1">
              <Select value={path} onChange={(e) => changeColumns(columns.map((p, j) => (j === i ? e.target.value : p)))} disabled={busy} className="h-8 min-w-0 flex-1 text-[13px]" aria-label={`Model ${i + 1}`}>
                {(["Fine-tuned", "Models"] as const).map(
                  (group) =>
                    options.some((o) => o.group === group) && (
                      <optgroup key={group} label={group}>
                        {options
                          .filter((o) => o.group === group)
                          .map((o) => (
                            <option key={o.path} value={o.path}>
                              {o.label}
                              {o.path === chatPath ? " (loaded in chat)" : ""}
                            </option>
                          ))}
                      </optgroup>
                    ),
                )}
              </Select>
              {columns.length > 2 && (
                <button className="rounded p-1 text-muted hover:bg-surface-2 hover:text-text disabled:opacity-40" onClick={() => changeColumns(columns.filter((_, j) => j !== i))} disabled={busy} title="Remove this model" aria-label={`Remove ${option(path)?.label ?? "model"}`}>
                  <X className="size-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
        {columns.length < MAX_MODELS && (
          <Button size="sm" variant="ghost" onClick={() => changeColumns([...columns, options.find((o) => !columns.includes(o.path))?.path ?? columns[0]])} disabled={busy} title="Compare another model">
            <Plus className="size-3.5" /> Model
          </Button>
        )}
      </div>

      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
        {turns.length === 0 ? (
          <EmptyState icon={<Columns2 className="size-8" />} title="Ask them all the same thing">
            Each model answers in its own column, with the system prompt and sampling from the Generation panel. Comparisons aren't saved.
          </EmptyState>
        ) : (
          <div className="flex flex-col gap-5 px-6 py-6">
            {turns.map((t, i) => (
              <div key={i} className="flex flex-col gap-3">
                <div className="prose-chat max-w-[620px] self-end rounded-2xl rounded-br-md bg-surface px-4 py-2.5 whitespace-pre-wrap shadow-[0_0_0_1px_var(--line)]">{t.prompt}</div>
                <div className="grid items-start gap-3" style={grid}>
                  {t.replies.map((r, j) => (
                    <ReplyCard key={j} reply={r} />
                  ))}
                </div>
              </div>
            ))}
            {!busy && (
              <div className="flex justify-end">
                <Button size="sm" variant="ghost" onClick={() => void send(turns[turns.length - 1].prompt, turns.slice(0, -1))}>
                  <RotateCcw className="size-3.5" /> Ask again
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-[760px] px-6 pb-5">
        {notice && <p className="mb-2 text-center text-[12px] text-muted">{notice}</p>}
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
            placeholder={`Message ${columns.length} models`}
            className="max-h-52 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 outline-none placeholder:text-muted/70"
            aria-label="Message"
          />
          {busy ? (
            <Button variant="secondary" onClick={() => abort.current?.abort()} aria-label="Stop generating" className="size-9 rounded-xl px-0">
              <Square className="size-3.5 fill-current" />
            </Button>
          ) : (
            <Button variant="primary" onClick={submit} disabled={!draft.trim()} aria-label="Send" className="size-9 rounded-xl px-0">
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[11.5px] text-muted">
          <span>Enter to send, Shift+Enter for a new line</span>
          <Toggle label="Let thinking models think" checked={think} onChange={setThink} />
        </div>
      </div>
    </div>
  );
}
