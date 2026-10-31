import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { t as tr, useT } from "../lib/i18n";
import { Section } from "./ui";

/** The engine's statistics (usage.py). */
interface Stats {
  tokens: {
    output: number;
    input: number;
    replies: number;
    since: string | null;
    days: { date: string; output_tokens: number; replies: number }[];
    models: { model: string; output_tokens: number }[];
  };
  chats: number;
  messages: number;
  training: { runs: number; finished: number; steps: number; seconds: number };
  exports: number;
  datasets: number;
  dataset_rows: number;
}

const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const full = new Intl.NumberFormat();

function hours(seconds: number): string {
  if (seconds < 3600) return tr("stats.min", { n: Math.round(seconds / 60) });
  return tr("stats.h", { n: (seconds / 3600).toFixed(seconds < 36000 ? 1 : 0) });
}

function Tile({ value, label, title }: { value: string; label: string; title?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2.5" title={title}>
      <div className="text-[20px] font-semibold tabular-nums">{value}</div>
      <div className="text-[12px] text-muted">{label}</div>
    </div>
  );
}

/** Tokens written per day over the last 30 days: one series, so no legend; a tooltip on each bar. */
function DailyTokens({ days }: { days: Stats["tokens"]["days"] }) {
  const t = useT();
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...days.map((d) => d.output_tokens));
  const shown = hover != null ? days[hover] : null;
  const label = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return (
    <figure className="mt-4">
      <figcaption className="mb-2 flex items-baseline justify-between text-[12.5px]">
        <span className="text-muted">{t("stats.last30")}</span>
        <span className="tabular-nums text-muted" aria-live="polite">
          {shown
            ? t(shown.replies === 1 ? "stats.day_one" : "stats.day", { date: label(shown.date), tokens: full.format(shown.output_tokens), n: shown.replies })
            : t("stats.most", { n: full.format(max === 1 && !days.some((d) => d.output_tokens) ? 0 : max) })}
        </span>
      </figcaption>
      <div className="flex h-24 items-end gap-[2px] border-b border-line" role="list" aria-label={t("stats.per_day")} onMouseLeave={() => setHover(null)}>
        {days.map((d, i) => (
          <div
            key={d.date}
            role="listitem"
            aria-label={t("stats.bar", { date: label(d.date), n: d.output_tokens })}
            className="flex h-full flex-1 items-end"
            onMouseEnter={() => setHover(i)}
          >
            <div
              className="w-full rounded-t-[4px] bg-accent transition-opacity"
              style={{ height: d.output_tokens ? `${Math.max(3, (d.output_tokens / max) * 100)}%` : 0, opacity: hover == null || hover === i ? 1 : 0.45 }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-muted tabular-nums">
        <span>{days.length ? label(days[0].date) : ""}</span>
        <span>{t("stats.today")}</span>
      </div>
    </figure>
  );
}

/** Settings → Statistics: what Temper has done on this computer. Nothing here leaves it. */
export function StatsSection() {
  const t = useT();
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    void api<Stats>("/api/stats").then(setStats, () => {});
  }, []);
  if (!stats) return null;

  const tok = stats.tokens;
  return (
    <Section title={t("stats.title")}>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Tile value={compact.format(tok.output)} label={t("stats.written")} title={t("stats.written_title", { out: full.format(tok.output), in: full.format(tok.input), n: full.format(tok.replies) })} />
        <Tile value={compact.format(stats.messages)} label={t("stats.replies", { n: full.format(stats.chats) })} />
        <Tile value={full.format(stats.training.finished)} label={t("stats.trained")} title={t("stats.trained_title", { runs: stats.training.runs, steps: full.format(stats.training.steps) })} />
        <Tile value={hours(stats.training.seconds)} label={t("stats.spent")} />
        <Tile value={full.format(stats.exports)} label={t("stats.exports")} />
        <Tile value={full.format(stats.datasets)} label={t("stats.datasets")} title={t("stats.rows", { n: full.format(stats.dataset_rows) })} />
        <Tile value={compact.format(tok.input)} label={t("stats.read")} />
        <Tile value={compact.format(stats.dataset_rows)} label={t("stats.training_rows")} />
      </div>
      <DailyTokens days={tok.days} />
      {tok.models.length > 0 && (
        <div className="mt-4 text-[13px]">
          <div className="mb-1 text-[12.5px] text-muted">{t("stats.models")}</div>
          <ul>
            {tok.models.map((m) => (
              <li key={m.model} className="flex items-center gap-3 border-t border-line py-1.5">
                <span className="min-w-0 flex-1 truncate" title={m.model}>
                  {m.model}
                </span>
                <span className="shrink-0 text-muted tabular-nums">{t("stats.tokens", { n: full.format(m.output_tokens) })}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="mt-3 text-[12px] text-muted">
        {tok.since ? t("stats.counted_since", { date: new Date(`${tok.since}T12:00:00`).toLocaleDateString() }) : t("stats.counted")}
      </p>
    </Section>
  );
}
