import { bytes } from "../lib/format";
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

const BUDGET_LABEL = { vram: "GPU memory", unified: "memory the GPU can use", ram: "memory" };
const VERDICT = {
  fits: { text: "Fits", tone: "text-ok", bar: "bg-ok" },
  tight: { text: "Fits only if you close other apps", tone: "text-straw", bar: "bg-straw" },
  too_large: { text: "Won't fit", tone: "text-danger", bar: "bg-danger" },
};

/** The estimate as a bar against what this machine can give it, with a breakdown and advice. */
export function MemoryEstimate({ estimate, loading }: { estimate: Estimate | null; loading?: boolean }) {
  if (!estimate) return <p className="text-[13px] text-muted">{loading ? "Estimating…" : "Enter a model to see how much memory it needs."}</p>;
  const { total, budget, verdict, advice, parts } = estimate;
  const v = VERDICT[verdict];
  const scale = Math.max(total, budget.total);
  return (
    <div className={cx("text-[13px] transition-opacity", loading && "opacity-60")}>
      <div className="flex items-baseline justify-between gap-3">
        <span className={cx("font-medium", v.tone)}>{v.text}</span>
        <span className="text-muted tabular-nums">
          about {bytes(total)} of {bytes(budget.total)}
        </span>
      </div>
      <div className="relative mt-2 h-2 overflow-hidden rounded-full bg-surface-2" role="img" aria-label={`Needs about ${bytes(total)}; ${bytes(budget.total)} available`}>
        <div className={cx("absolute inset-y-0 left-0 rounded-full", v.bar)} style={{ width: `${Math.min(100, (total / scale) * 100)}%` }} />
        {/* Where the limit is, when the estimate runs past it. */}
        {total > budget.total && <div className="absolute inset-y-0 w-0.5 bg-bg" style={{ left: `${(budget.total / scale) * 100}%` }} title={`${bytes(budget.total)} available`} />}
        {/* What's free right now, with other apps open. */}
        <div className="absolute inset-y-0 w-px bg-text/50" style={{ left: `${Math.min(100, (budget.free / scale) * 100)}%` }} title={`${bytes(budget.free)} free now`} />
      </div>
      <p className="mt-1.5 text-[12px] text-muted">
        {bytes(budget.free)} of {bytes(budget.total)} {BUDGET_LABEL[budget.kind]} is free now.
        {estimate.measured === false && " A rough estimate for this GPU."}
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
