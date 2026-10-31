import { useEffect, useState } from "react";
import type { Crash } from "../lib/crash";
import { api, openExternal } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Section, Toggle } from "./ui";

/** Settings → Privacy: no telemetry, and crash reports kept here until the user sends one. */
export function Privacy() {
  const t = useT();
  const [offer, setOffer] = useState(false);
  const [crashes, setCrashes] = useState<Crash[]>([]);

  const load = () =>
    void api<{ offer: boolean; crashes: Crash[] }>("/api/crashes").then((c) => {
      setOffer(c.offer);
      setCrashes(c.crashes);
    }, () => {});
  useEffect(load, []);

  async function setOn(on: boolean) {
    const c = await api<{ offer: boolean; crashes: Crash[] }>("/api/crashes/settings", { json: { offer: on } });
    setOffer(c.offer);
    window.dispatchEvent(new Event("temper:crash-settings"));
  }

  async function clear() {
    await api("/api/crashes", { method: "DELETE" });
    load();
  }

  return (
    <Section title={t("privacy.title")}>
      <p className="text-[13px] text-muted">
        {t("privacy.about")}
      </p>
      <div className="mt-3">
        <Toggle checked={offer} onChange={(on) => void setOn(on)} label={t("privacy.offer")} />
        <p className="mt-1 text-[12px] text-muted">
          {t("privacy.offer_hint")}
        </p>
      </div>
      {crashes.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between text-[12.5px] text-muted">
            <span>{t("privacy.saved")}</span>
            <button className="hover:text-text" onClick={() => void clear()}>
              {t("privacy.delete_all")}
            </button>
          </div>
          <ul className="rounded-lg border border-line bg-surface text-[13px]">
            {crashes.slice(0, 8).map((c) => (
              <li key={c.id} className="flex items-center gap-3 border-t border-line px-3 py-2 first:border-t-0">
                <span className="shrink-0 text-[12px] text-muted tabular-nums">{new Date(c.time * 1000).toLocaleString()}</span>
                <span className="min-w-0 flex-1 truncate" title={`${c.error}\n${c.context}`}>
                  {c.error}
                </span>
                <button className="shrink-0 text-accent hover:underline" onClick={() => void openExternal(c.report_url)}>
                  {t("privacy.report")}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Section>
  );
}
