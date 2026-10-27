import { Layers, MessageSquare, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/engine";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { cx, Select } from "./ui";

export const IDLE_CHOICES: [number, Key, number?][] = [
  [5, "idle.minutes", 5],
  [15, "idle.minutes", 15],
  [30, "idle.minutes", 30],
  [60, "idle.hour"],
  [240, "idle.hours", 4],
  [0, "idle.never"],
];

/** Unloads models nobody has used for a while (engine loaded.py); shared by the status bar and Settings. */
export function IdleSelect({ className }: { className?: string }) {
  const [minutes, setMinutes] = useState<number | null>(null);
  const t = useT();
  useEffect(() => {
    api<{ idle_minutes: number }>("/api/runtime/loaded").then((r) => setMinutes(r.idle_minutes), () => {});
  }, []);
  if (minutes === null) return null;
  return (
    <Select
      value={minutes}
      onChange={(e) => {
        const value = Number(e.target.value);
        setMinutes(value);
        void api("/api/runtime/loaded/idle", { method: "PUT", json: { minutes: value } });
      }}
      className={cx("h-7 w-auto text-[12.5px]", className)}
      aria-label={t("loaded.idle")}
    >
      {IDLE_CHOICES.map(([value, label, n]) => (
        <option key={value} value={value}>
          {t(label, { n: n ?? "" })}
        </option>
      ))}
    </Select>
  );
}

/** The status bar's model: chat's, plus a list of every loaded model to switch chat to or unload. */
export function LoadedModels() {
  const { runtime, refreshRuntime, go } = useApp();
  const t = useT();
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const alongside = runtime?.alongside ?? [];

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  async function act(path: string, what: "switch" | "unload") {
    setError(null);
    try {
      await api(`/api/runtime/loaded/${what}`, { json: { path } });
      await refreshRuntime();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const chat = runtime && runtime.state !== "idle" ? runtime : null;
  const rows = [...(chat?.path ? [{ ...chat, path: chat.path, chat: true }] : []), ...alongside.map((m) => ({ ...m, chat: false }))];

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => (rows.length ? setOpen((o) => !o) : go("models"))}
        className="flex items-center gap-1.5 hover:text-text"
        aria-expanded={open}
        title={rows.length ? t("loaded.title") : t("loaded.load")}
      >
        {chat?.state === "ready" ? (
          <>
            <span className="size-2 rounded-full bg-accent" /> {chat.model}
          </>
        ) : chat?.state === "loading" ? (
          <>
            <span className="animate-pulse-dot size-2 rounded-full bg-straw" /> {t("loaded.loading", { model: chat.model ?? "" })}
          </>
        ) : chat?.state === "failed" ? (
          <>
            <span className="size-2 rounded-full bg-danger" /> {t("loaded.failed", { model: chat.model ?? "" })}
          </>
        ) : (
          t("loaded.none")
        )}
        {alongside.length > 0 && (
          <span className="flex items-center gap-0.5 rounded bg-surface-2 px-1 text-[11px]">
            <Layers className="size-3" />+{alongside.length}
          </span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 bottom-8 z-30 w-80 rounded-lg border border-line bg-surface p-2 text-[13px] text-text shadow-lg">
          <p className="px-2 pt-1 pb-2 text-[12px] text-muted">{t("loaded.intro")}</p>
          <ul className="flex flex-col">
            {rows.map((m) => (
              <li key={m.path} className="flex items-center gap-2 rounded px-2 py-1.5 hover:bg-surface-2">
                <span className={cx("size-2 shrink-0 rounded-full", m.state === "ready" ? "bg-accent" : m.state === "loading" ? "animate-pulse-dot bg-straw" : "bg-danger")} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate" title={m.error ?? m.path}>
                    {m.model}
                  </span>
                  <span className="text-[11.5px] text-muted">
                    {t(m.chat ? "loaded.in_chat" : m.state === "loading" ? "loaded.state_loading" : m.state === "failed" ? "loaded.state_failed" : "loaded.alongside")}
                    {m.engine && m.engine !== "llama.cpp" ? ` · ${m.engine}` : ""}
                  </span>
                </span>
                {!m.chat && m.state === "ready" && (
                  <button className="text-muted hover:text-text" onClick={() => void act(m.path, "switch")} title={t("loaded.use")}>
                    <MessageSquare className="size-3.5" aria-label={`Use ${m.model} in chat`} />
                  </button>
                )}
                <button className="text-muted hover:text-danger" onClick={() => void act(m.path, "unload")} title={t("loaded.unload")}>
                  <X className="size-3.5" aria-label={`Unload ${m.model}`} />
                </button>
              </li>
            ))}
          </ul>
          {error && <p className="px-2 pt-1 text-[12px] text-danger">{error}</p>}
          <div className="mt-1 flex items-center justify-between gap-2 border-t border-line px-2 pt-2">
            <span className="text-[12px] text-muted">{t("loaded.idle")}</span>
            <IdleSelect />
          </div>
        </div>
      )}
    </div>
  );
}
