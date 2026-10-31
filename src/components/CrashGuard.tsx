import { AlertTriangle, X } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode, useEffect, useState } from "react";
import { type Crash, reportCrash } from "../lib/crash";
import { api, openExternal } from "../lib/engine";
import { Button } from "./ui";

/** Catches a page that fails to draw: says so, saves a report, and offers to try again or report it. */
export class CrashGuard extends Component<{ page: string; children: ReactNode }, { error: Error | null; crash: Crash | null }> {
  state = { error: null as Error | null, crash: null as Crash | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    if (info.componentStack) error.stack = `${error.stack ?? ""}\nComponents:${info.componentStack}`;
    void reportCrash(error, `${this.props.page} page`).then((crash) => this.setState({ crash }));
  }

  componentDidUpdate(prev: { page: string }) {
    if (prev.page !== this.props.page && this.state.error) this.setState({ error: null, crash: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    const { crash } = this.state;
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="max-w-md text-center">
          <AlertTriangle className="mx-auto size-6 text-straw" />
          <h2 className="mt-3 font-semibold">This page hit an error</h2>
          <p className="mt-1 text-[13px] text-muted">
            {this.state.error.message}. Your chats, models and runs are safe. A report was saved on this computer; it's only sent if you choose to report it.
          </p>
          <div className="mt-4 flex justify-center gap-2">
            <Button onClick={() => this.setState({ error: null, crash: null })}>Try again</Button>
            {crash && (
              <Button variant="ghost" onClick={() => void openExternal(crash.report_url)}>
                Report on GitHub
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  }
}

let offer: Promise<boolean> | null = null;

/** When "Offer to report crashes" is on, a note after a crash with a link to report it. */
export function CrashNotice() {
  const [crash, setCrash] = useState<Crash | null>(null);

  useEffect(() => {
    const on = (e: Event) => {
      offer ??= api<{ offer: boolean }>("/api/crashes").then((c) => c.offer, () => false);
      void offer.then((yes) => yes && setCrash((e as CustomEvent<Crash>).detail));
    };
    const changed = () => (offer = null);
    window.addEventListener("temper:crash", on);
    window.addEventListener("temper:crash-settings", changed);
    return () => {
      window.removeEventListener("temper:crash", on);
      window.removeEventListener("temper:crash-settings", changed);
    };
  }, []);
  if (!crash) return null;
  return (
    <div className="fixed right-4 bottom-10 z-50 flex max-w-sm items-start gap-3 rounded-lg border border-line bg-surface p-3 text-[13px] shadow-lg" role="alert">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-straw" />
      <div className="min-w-0">
        <div className="font-medium">Temper hit an error</div>
        <div className="truncate text-muted" title={crash.error}>
          {crash.error}
        </div>
        <button
          className="mt-1 text-accent hover:underline"
          onClick={() => {
            void openExternal(crash.report_url);
            setCrash(null);
          }}
        >
          Review and report on GitHub
        </button>
      </div>
      <button className="text-muted hover:text-text" onClick={() => setCrash(null)} aria-label="Dismiss">
        <X className="size-3.5" />
      </button>
    </div>
  );
}
