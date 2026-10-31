import { useEffect, useState } from "react";
import type { Crash } from "../lib/crash";
import { api, openExternal } from "../lib/engine";
import { Section, Toggle } from "./ui";

/** Settings → Privacy: no telemetry, and crash reports kept here until the user sends one. */
export function Privacy() {
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
    <Section title="Privacy">
      <p className="text-[13px] text-muted">
        Temper sends no telemetry: no usage data, no analytics, nothing about your chats, models or files. It goes online only to do what you ask, like downloading a model or searching the
        web, and to check for updates (which you can turn off above). The libraries it uses are told not to send telemetry either.
      </p>
      <div className="mt-3">
        <Toggle checked={offer} onChange={(on) => void setOn(on)} label="Offer to report crashes" />
        <p className="mt-1 text-[12px] text-muted">
          When Temper hits an error, it saves a report here, without your home folder, keys or tokens. With this on, it also offers to open it as a GitHub issue, which you read and send
          yourself.
        </p>
      </div>
      {crashes.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between text-[12.5px] text-muted">
            <span>Saved reports</span>
            <button className="hover:text-text" onClick={() => void clear()}>
              Delete all
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
                  Report
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Section>
  );
}
