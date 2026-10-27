import { useT, type Key } from "../lib/i18n";
import { Button, cx } from "./ui";

/** Why a message didn't fit the model's context (engine compaction.py explain()). */
export interface Fit {
  context: number;
  prompt: number;
  reserve: number;
  over: number;
  parts: { kind: "instructions" | "earlier" | "message" | "attachment" | "reply"; tokens: number; name?: string }[];
  fixes: FitFix[];
}

export type FitFix =
  | { kind: "max_tokens"; to: number }
  | { kind: "remove_attachment"; name: string; tokens: number }
  | { kind: "compact" | "new_chat"; tokens: number }
  | { kind: "shorten_instructions"; tokens: number; files: number }
  | { kind: "longer_context"; to: number; max: number };

const COLORS: Record<Fit["parts"][number]["kind"], string> = { instructions: "bg-violet", earlier: "bg-bronze", message: "bg-accent", attachment: "bg-straw", reply: "bg-ok" };
const n = (v: number) => v.toLocaleString();

/** Where the tokens go and what would make the message fit; `onFix` applies the fixes the page can (a shorter reply, compaction, a new chat). */
export function ContextFit({ fit, message, onFix }: { fit: Fit; message: string; onFix?: (fix: FitFix) => void }) {
  const t = useT();
  const need = fit.prompt + fit.reserve;
  const parts = fit.parts.filter((p) => p.tokens > 0);
  const label = (p: Fit["parts"][number]) => (p.kind === "attachment" ? (p.name ?? "") : t(`fit.${p.kind}` as Key));
  const action: Partial<Record<FitFix["kind"], Key>> = { max_tokens: "fit.apply_retry", compact: "fit.apply_retry", new_chat: "fit.new_chat_button" };

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-danger/40 bg-danger/8 px-3.5 py-3 text-[13px]">
      <div>
        <p className="font-medium">{t("fit.title")}</p>
        <p className="mt-0.5 text-muted">{t("fit.summary", { need: n(need), context: n(fit.context), over: n(fit.over) })}</p>
      </div>
      <div>
        <div className="relative flex h-2.5 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={parts.map((p) => `${label(p)} ${n(p.tokens)}`).join(", ")}>
          {parts.map((p, i) => (
            <div key={i} className={cx("h-full", COLORS[p.kind], i > 0 && "border-l border-bg")} style={{ width: `${(p.tokens / need) * 100}%` }} />
          ))}
          {/* The end of the context window. */}
          <div className="absolute inset-y-0 w-0.5 bg-text" style={{ left: `${Math.min((fit.context / need) * 100, 100)}%` }} />
        </div>
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-muted">
          {parts.map((p, i) => (
            <li key={i} className="flex items-center gap-1.5">
              <span className={cx("size-2 rounded-full", COLORS[p.kind])} />
              <span className="max-w-[180px] truncate">{label(p)}</span>
              <span className="tabular-nums">{n(p.tokens)}</span>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <p className="text-[12px] font-medium text-muted">{fit.fixes.length ? t("fit.fixes") : t("fit.none")}</p>
        <ul className="mt-1 flex flex-col gap-1.5">
          {fit.fixes.map((f, i) => (
            <li key={i} className="flex items-center justify-between gap-3">
              <span>{t(`fit.${f.kind}` as Key, { ...f, ...("tokens" in f ? { tokens: n(f.tokens) } : {}), ...("to" in f ? { to: n(f.to) } : {}), ...("max" in f ? { max: n(f.max) } : {}) })}</span>
              {onFix && action[f.kind] && (
                <Button size="sm" variant="ghost" className="shrink-0" onClick={() => onFix(f)}>
                  {t(action[f.kind]!)}
                </Button>
              )}
            </li>
          ))}
        </ul>
      </div>
      <p className="sr-only">{message}</p>
    </div>
  );
}
