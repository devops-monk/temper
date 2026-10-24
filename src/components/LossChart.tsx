import { useId } from "react";

interface Point {
  step: number;
  loss: number;
}

/** Training loss over steps. The line uses the temper gradient: it "cools" from straw to blue as training proceeds. */
export function LossChart({ points, totalSteps }: { points: Point[]; totalSteps: number | null }) {
  const id = useId();
  const W = 560;
  const H = 180;
  const pad = { l: 40, r: 12, t: 12, b: 24 };

  if (points.length < 2) {
    return (
      <div className="flex h-[180px] items-center justify-center rounded-lg border border-dashed border-line text-[13px] text-muted">
        The loss curve appears after the first few steps.
      </div>
    );
  }

  const maxStep = Math.max(totalSteps ?? 0, points[points.length - 1].step);
  const losses = points.map((p) => p.loss);
  const lo = Math.min(...losses);
  const hi = Math.max(...losses);
  const span = hi - lo || 1;
  const x = (s: number) => pad.l + (s / maxStep) * (W - pad.l - pad.r);
  const y = (l: number) => pad.t + (1 - (l - lo) / span) * (H - pad.t - pad.b);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join("");
  const ticks = [hi, lo + span / 2, lo];
  const last = points[points.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Loss ${last.loss.toFixed(3)} at step ${last.step}`}>
      <defs>
        <linearGradient id={id} x1={pad.l} x2={W - pad.r} gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#E3BE63" />
          <stop offset=".35" stopColor="#C9864A" />
          <stop offset=".7" stopColor="#8D65CB" />
          <stop offset="1" stopColor="#4F86E8" />
        </linearGradient>
      </defs>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeDasharray="2 4" />
          <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" fontSize="10.5" fill="var(--muted)">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={pad.l} y={H - 6} fontSize="10.5" fill="var(--muted)">
        0
      </text>
      <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize="10.5" fill="var(--muted)">
        step {maxStep}
      </text>
      <path d={d} fill="none" stroke={`url(#${id})`} strokeWidth="2.25" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(last.step)} cy={y(last.loss)} r="3.5" fill="var(--surface)" stroke={`url(#${id})`} strokeWidth="2" />
    </svg>
  );
}
