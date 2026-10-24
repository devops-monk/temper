import { AlertTriangle, CheckCircle2, Circle, Copy, RotateCw, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { Button, Notice, Section, cx } from "./ui";

type Status = "ok" | "warn" | "fail" | "info";
type Action = "install_llamacpp" | "reinstall_engine" | "add_training";

interface Check {
  id: string;
  title: string;
  status: Status;
  detail: string;
  fix: string | null;
  action: Action | null;
}

const ICON = {
  ok: <CheckCircle2 className="size-4 text-ok" />,
  warn: <AlertTriangle className="size-4 text-straw" />,
  fail: <XCircle className="size-4 text-danger" />,
  info: <Circle className="size-4 text-muted" />,
};
const ACTION_LABEL: Record<Action, string> = {
  install_llamacpp: "Install llama.cpp",
  reinstall_engine: "Reinstall engine",
  add_training: "Add training support",
};
const SYMBOL = { ok: "✓", warn: "!", fail: "✗", info: "·" };

/** Settings → Diagnostics: the engine's doctor checks, with one-click fixes where the app can apply them. */
export function Diagnostics({
  ready,
  installed,
  onReinstall,
}: {
  ready: boolean;
  /** The installed engine profile; reinstalling keeps it. */
  installed: "chat" | "train";
  onReinstall: (profile: "chat" | "train", clean: boolean) => void;
}) {
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);

  async function run() {
    setRunning(true);
    setError(null);
    try {
      setChecks((await api<{ checks: Check[] }>("/api/doctor")).checks);
    } catch (e) {
      setError(`The checks couldn't run. ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRunning(false);
    }
  }

  // Check again once a fix the app started has finished.
  useEffect(() => {
    if (job.status === "done" || job.status === "failed") {
      setJobId(null);
      void run();
    }
  }, [job.status]);

  async function fix(action: Action) {
    if (action === "install_llamacpp") {
      setJobId((await api<Job>("/api/llamacpp/install", { json: {} })).id);
    } else if (action === "add_training") {
      onReinstall("train", false);
    } else {
      onReinstall(installed, true);
    }
  }

  async function copy() {
    const text = (checks ?? []).map((c) => `${SYMBOL[c.status]} ${c.title}: ${c.detail}${c.fix && c.status !== "ok" ? `\n    → ${c.fix}` : ""}`).join("\n");
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  const problems = checks?.filter((c) => c.status === "fail" || c.status === "warn").length ?? 0;

  return (
    <Section
      title="Diagnostics"
      aside={
        <div className="flex gap-2">
          {checks && (
            <Button size="sm" onClick={() => void copy()}>
              <Copy className="size-3.5" /> {copied ? "Copied" : "Copy report"}
            </Button>
          )}
          <Button size="sm" disabled={!ready || running} onClick={() => void run()}>
            <RotateCw className={cx("size-3.5", running && "animate-spin")} /> {checks ? "Run again" : "Run checks"}
          </Button>
        </div>
      }
    >
      {!checks ? (
        <p className="text-[13px] text-muted">
          {ready ? "Checks drivers, GPU libraries, disk space, the API port and the engine's packages, and says how to fix what it finds." : "Diagnostics run once the engine is ready."}
        </p>
      ) : (
        <>
          <p className="mb-2 text-[13px] text-muted">{problems ? `${problems} thing${problems > 1 ? "s" : ""} to look at.` : "Everything looks good."}</p>
          <ul className="rounded-lg border border-line bg-surface">
            {checks.map((c) => (
              <li key={c.id} className="flex gap-3 border-t border-line px-4 py-2.5 first:border-t-0">
                <span className="mt-0.5 shrink-0">{ICON[c.status]}</span>
                <div className="min-w-0 flex-1 text-[13px]">
                  <div className="font-medium">{c.title}</div>
                  <div className="break-words text-muted">{c.detail}</div>
                  {c.fix && c.status !== "ok" && <div className="mt-1">{c.fix}</div>}
                </div>
                {c.action && c.status !== "ok" && (
                  <Button size="sm" className="shrink-0 self-center" disabled={!!jobId} onClick={() => void fix(c.action!)}>
                    {c.action === "install_llamacpp" && jobId ? "Installing…" : ACTION_LABEL[c.action]}
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
      {error && (
        <div className="mt-3">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
    </Section>
  );
}
