import { bytes } from "../lib/format";
import { type Key, useT } from "../lib/i18n";
import { cx } from "./ui";

export type Verdict = "fits" | "tight" | "too_large";

export interface Budget {
  kind: "vram" | "unified" | "ram";
  device: string;
  total: number;
  free: number;
}

export interface Estimate {
  total: number;
  budget: Budget;
  verdict: Verdict;
  advice: string[];
  /** Parts of the total, in the order to show them. */
  parts: [label: string, value: number][];
  /** False when the constants come from published figures rather than runs we measured. */
  measured?: boolean;
}

const FREE: Record<Budget["kind"], Key> = { vram: "mem.free_vram", unified: "mem.free_unified", ram: "mem.free_ram" };
const VERDICT: Record<Verdict, { text: Key; tone: string; bar: string }> = {
  fits: { text: "mem.fits", tone: "text-ok", bar: "bg-ok" },
  tight: { text: "mem.tight", tone: "text-straw", bar: "bg-straw" },
  too_large: { text: "mem.too_large", tone: "text-danger", bar: "bg-danger" },
};

/** The estimate as a bar against what this machine can give it, with a breakdown and advice. */
export function MemoryEstimate({ estimate, loading }: { estimate: Estimate | null; loading?: boolean }) {
  const t = useT();
  if (!estimate) return <p className="text-[13px] text-muted">{t(loading ? "mem.estimating" : "mem.enter_model")}</p>;
  const { total, budget, verdict, advice, parts } = estimate;
  const v = VERDICT[verdict];
  const scale = Math.max(total, budget.total);
  return (
    <div className={cx("text-[13px] transition-opacity", loading && "opacity-60")}>
      <div className="flex items-baseline justify-between gap-3">
        <span className={cx("font-medium", v.tone)}>{t(v.text)}</span>
        <span className="text-muted tabular-nums">
          {t("mem.about", { need: bytes(total), total: bytes(budget.total) })}
        </span>
      </div>
      <div className="relative mt-2 h-2 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={t("mem.bar", { need: bytes(total), total: bytes(budget.total) })}>
        <div className={cx("absolute inset-y-0 left-0 rounded-full", v.bar)} style={{ width: `${Math.min(100, (total / scale) * 100)}%` }} />
        {/* Where the limit is, when the estimate runs past it. */}
        {total > budget.total && <div className="absolute inset-y-0 w-0.5 bg-bg" style={{ left: `${(budget.total / scale) * 100}%` }} title={t("mem.available", { total: bytes(budget.total) })} />}
        {/* What's free right now, with other apps open. */}
        <div className="absolute inset-y-0 w-px bg-text/50" style={{ left: `${Math.min(100, (budget.free / scale) * 100)}%` }} title={t("mem.free_now", { free: bytes(budget.free) })} />
      </div>
      <p className="mt-1.5 text-[12px] text-muted">
        {t(FREE[budget.kind], { free: bytes(budget.free), total: bytes(budget.total) })}
        {estimate.measured === false && t("mem.rough")}
      </p>
      <dl className="mt-2 grid grid-cols-[1fr_auto] gap-x-4 gap-y-0.5 text-[12.5px]">
        {parts.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted">{label}</dt>
            <dd className="text-right tabular-nums">{bytes(value)}</dd>
          </div>
        ))}
      </dl>
      {verdict !== "fits" && advice.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-4 text-[12.5px]">
          {advice.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
