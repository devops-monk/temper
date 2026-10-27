import { ChevronRight, LoaderCircle, ShieldAlert, ShieldCheck, Wrench } from "lucide-react";
import { Fragment, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT, type Key } from "../lib/i18n";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { ReplyText } from "./Reply";
import { Button, ProgressBar, Section, Select, cx } from "./ui";

/** A tool a reply called (engine tools.py); `at` is where in the reply's text it happened, and `result` is null while it runs.
 * `approval` is set for tools that ask first: "pending" while it waits for the person. */
export interface ToolStep {
  id: string;
  name: string;
  arguments: Record<string, unknown> | string;
  result: string | null;
  error: boolean;
  at: number;
  approval?: "pending" | "allowed" | "declined";
}

/** How a chat uses a tool: calls run straight away, or each waits to be allowed. Tools left out are off. */
export type ToolMode = "auto" | "ask";
export type ToolModes = Record<string, ToolMode>;

interface ToolInfo {
  name: string;
  label: string;
  description: string;
  mode: ToolMode;
}

/** The tools a model can call, as the engine lists them; fetched once. */
let listed: Promise<{ tools: ToolInfo[]; sandbox: string | null }> | null = null;
const toolList = () => (listed ??= api<{ tools: ToolInfo[]; sandbox: string | null }>("/api/tools").catch(() => ((listed = null), { tools: [], sandbox: null })));

/** A tool's name (`tool.<name>`) or description (`tool.<name>.about`) in the app's language, else the engine's English. */
function useLabel() {
  const t = useT();
  return (key: string, fallback: string) => {
    const text = t(key as Key);
    return text === key ? fallback : text;
  };
}

const CODE = ["run_python", "run_javascript"];

/** The Generation panel's tools: each off, asking before each call, or running calls straight away. */
export function ToolsControl({ value, onChange }: { value: ToolModes; onChange: (tools: ToolModes) => void }) {
  const t = useT();
  const label = useLabel();
  const [list, setList] = useState<{ tools: ToolInfo[]; sandbox: string | null } | null>(null);
  useEffect(() => void toolList().then(setList), []);
  if (!list?.tools.length) return null;
  const set = (name: string, mode: ToolMode | "off") => {
    const { [name]: _, ...rest } = value;
    onChange(mode === "off" ? rest : { ...rest, [name]: mode });
  };
  const code = list.tools.some((tool) => CODE.includes(tool.name) && value[tool.name]);
  return (
    <div className="flex flex-col gap-2 border-t border-line pt-4">
      <h3 className="flex items-center gap-1.5 text-[13px] font-semibold">
        <Wrench className="size-3.5" /> {t("tools.title")}
      </h3>
      {list.tools.map((tool) => (
        <div key={tool.name}>
          <div className="flex items-center justify-between gap-3">
            <span className="text-[13px]">{label(`tool.${tool.name}`, tool.label)}</span>
            <Select value={value[tool.name] ?? "off"} onChange={(e) => set(tool.name, e.target.value as ToolMode | "off")} aria-label={label(`tool.${tool.name}`, tool.label)} className="h-7 w-24 text-[12.5px]">
              <option value="off">{t("tools.off")}</option>
              <option value="ask">{t("tools.ask")}</option>
              <option value="auto">{t("tools.auto")}</option>
            </Select>
          </div>
          <p className="text-[12px] text-muted">{label(`tool.${tool.name}.about`, tool.description)}</p>
        </div>
      ))}
      {code && <SandboxNote kind={list.sandbox} />}
      <p className="text-[12px] text-muted">{t("tools.hint")}</p>
    </div>
  );
}

function SandboxNote({ kind }: { kind: string | null }) {
  const t = useT();
  return kind ? (
    <p className="flex gap-1.5 text-[12px] text-muted">
      <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-ok" /> {t(kind === "seatbelt" ? "sandbox.seatbelt" : "sandbox.bubblewrap")}
    </p>
  ) : (
    <p className="flex gap-1.5 text-[12px] text-straw">
      <ShieldAlert className="mt-0.5 size-3.5 shrink-0" /> {t("sandbox.none")}
    </p>
  );
}

const shown = (args: ToolStep["arguments"]) => (typeof args === "string" ? args : Object.values(args).length === 1 ? String(Object.values(args)[0]) : JSON.stringify(args));
/** The arguments in full: code as it was written, anything else as JSON. */
const full = (args: ToolStep["arguments"]) => (typeof args === "string" ? args : typeof args.code === "string" && Object.keys(args).length === 1 ? args.code : JSON.stringify(args, null, 2));

/** What a card's buttons do with a call waiting to be allowed; `always` also sets the tool to run without asking in this chat. */
export type OnApprove = (step: ToolStep, allow: boolean, always?: boolean) => void;

/** One tool call in a reply: what it was asked, folded open to show the result; a call waiting to be allowed shows it in full with the choice. */
export function ToolCard({ step, onApprove }: { step: ToolStep; onApprove?: OnApprove }) {
  const t = useT();
  const label = useLabel();
  const [open, setOpen] = useState(false);
  const waiting = step.approval === "pending";
  const running = step.result === null && !waiting;
  const name = label(`tool.${step.name}`, step.name);
  const title = waiting ? t("tools.wants", { name }) : running ? t("tools.running", { name }) : step.approval === "declined" ? t("tools.declined", { name }) : step.error ? t("tools.failed", { name }) : t("tools.used", { name });
  return (
    <div className={cx("not-prose my-2 rounded-lg border bg-surface text-[12.5px]", waiting ? "border-accent/60" : "border-line")}>
      <button onClick={() => setOpen(!open)} disabled={running || waiting} aria-expanded={open || waiting} className="flex w-full items-center gap-2 px-3 py-2 text-left">
        {running ? <LoaderCircle className="size-3.5 shrink-0 animate-spin text-muted" /> : <Wrench className={cx("size-3.5 shrink-0", step.error ? "text-danger" : waiting ? "text-accent" : "text-muted")} />}
        <span className="shrink-0 font-medium">{title}</span>
        {!waiting && <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-muted">{shown(step.arguments)}</span>}
        {!running && !waiting && <ChevronRight className="ml-auto size-3 shrink-0 text-muted transition-transform data-[open=true]:rotate-90" data-open={open} />}
      </button>
      {waiting && (
        <div className="flex flex-col gap-2 border-t border-line px-3 py-2">
          <pre className="max-h-72 overflow-y-auto rounded-md bg-bg px-2.5 py-2 font-mono text-[12px] break-all whitespace-pre-wrap">{full(step.arguments)}</pre>
          {onApprove ? (
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="primary" onClick={() => onApprove(step, true)}>
                {t("tools.run")}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onApprove(step, true, true)}>
                {t("tools.always")}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onApprove(step, false)}>
                {t("tools.decline")}
              </Button>
            </div>
          ) : (
            <p className="text-[12px] text-muted">{t("tools.waiting")}</p>
          )}
        </div>
      )}
      {open && !running && !waiting && (
        <div className="flex flex-col gap-2 border-t border-line px-3 py-2">
          <div>
            <p className="text-[11.5px] font-medium text-muted">{t("tools.arguments")}</p>
            <pre className="mt-0.5 max-h-60 overflow-y-auto font-mono text-[12px] break-all whitespace-pre-wrap">{full(step.arguments)}</pre>
          </div>
          <div>
            <p className="text-[11.5px] font-medium text-muted">{t("tools.result")}</p>
            <pre className={cx("mt-0.5 max-h-60 overflow-y-auto font-mono text-[12px] break-all whitespace-pre-wrap", step.error && "text-danger")}>{step.result}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

/** Tool calls the model is still writing into its text (other engines' format), hidden until the engine takes them out. */
export const withoutCallText = (text: string) => text.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "").replace(/<\|python_tag\|>[\s\S]*$/, "");

/** A reply with its tool calls in place: text up to each call, the call, then the rest. */
export function ReplyWithTools({ text, steps, onApprove }: { text: string; steps: ToolStep[]; onApprove?: OnApprove }) {
  let start = 0;
  const at = [...new Set(steps.map((s) => Math.min(s.at, text.length)))].sort((a, b) => a - b);
  return (
    <>
      {at.map((offset) => {
        const before = text.slice(start, offset);
        start = offset;
        return (
          <Fragment key={offset}>
            {before.trim() && <ReplyText text={before} />}
            {steps
              .filter((s) => Math.min(s.at, text.length) === offset)
              .map((s) => (
                <ToolCard key={s.id} step={s} onApprove={onApprove} />
              ))}
          </Fragment>
        );
      })}
      {text.slice(start).trim() && <ReplyText text={withoutCallText(text.slice(start))} />}
    </>
  );
}

interface SandboxStatus {
  sandbox: string | null;
  data_libraries: boolean;
  packages: string[];
  node: string | null;
}

/** Settings: what protects code the model runs, and the data libraries for its Python (engine sandbox.py). */
export function CodeSandbox() {
  const t = useT();
  const { jobs, refreshJobs } = useApp();
  const [info, setInfo] = useState<SandboxStatus | null>(null);
  const [started, setStarted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const jobId = started ?? jobs.find((j) => j.kind === "sandbox-install" && j.status === "running")?.id ?? null;
  const { status, events } = useJob(jobId);
  const installing = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const lastLine = [...events].reverse().find((e) => e.event === "log")?.line as string | undefined;
  const failure = events.find((e) => e.event === "status" && e.status === "failed")?.error as string | undefined;
  const refresh = () => void api<SandboxStatus>("/api/sandbox").then(setInfo, () => {});
  useEffect(refresh, [status]);

  async function add() {
    setError(null);
    try {
      const job = await api<Job>("/api/sandbox/python", { method: "POST" });
      setStarted(job.id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove() {
    if (!confirm(t("sandbox.remove_confirm"))) return;
    setInfo(await api<SandboxStatus>("/api/sandbox/python", { method: "DELETE" }).catch(() => info));
  }

  if (!info) return null;
  return (
    <Section title={t("sandbox.title")}>
      <div className="flex flex-col gap-3 text-[13px]">
        <SandboxNote kind={info.sandbox} />
        <div className="flex items-center justify-between gap-3">
          <span>
            <span className="font-medium">{t("sandbox.data")}</span>
            <span className="block text-[12px] text-muted">{info.data_libraries ? t("sandbox.data_on", { packages: info.packages.join(", ") }) : t("sandbox.data_off", { packages: info.packages.join(", ") })}</span>
          </span>
          {info.data_libraries ? (
            <Button size="sm" variant="ghost" onClick={() => void remove()}>
              {t("sandbox.remove")}
            </Button>
          ) : (
            <Button size="sm" onClick={() => void add()} disabled={installing}>
              {t("sandbox.add")}
            </Button>
          )}
        </div>
        {installing && (
          <div className="flex flex-col gap-1.5">
            <ProgressBar value={progress ?? null} />
            <p className="truncate font-mono text-[11.5px] text-muted" title={lastLine}>
              {lastLine ?? "…"}
            </p>
          </div>
        )}
        {(error || (status === "failed" && failure)) && <p className="text-[12px] text-danger">{error ?? failure}</p>}
        <p className="text-[12px] text-muted">{info.node ? t("sandbox.node", { path: info.node }) : t("sandbox.no_node")}</p>
      </div>
    </Section>
  );
}
