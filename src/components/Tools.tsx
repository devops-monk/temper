import { ChevronRight, LoaderCircle, Wrench } from "lucide-react";
import { Fragment, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT, type Key } from "../lib/i18n";
import { ReplyText } from "./Reply";
import { Toggle, cx } from "./ui";

/** A tool a reply called (engine tools.py); `at` is where in the reply's text it happened, and `result` is null while it runs. */
export interface ToolStep {
  id: string;
  name: string;
  arguments: Record<string, unknown> | string;
  result: string | null;
  error: boolean;
  at: number;
}

interface ToolInfo {
  name: string;
  label: string;
  description: string;
}

/** The tools a model can call, as the engine lists them; fetched once. */
let listed: Promise<ToolInfo[]> | null = null;
const toolList = () => (listed ??= api<{ tools: ToolInfo[] }>("/api/tools").then((r) => r.tools, () => ((listed = null), [])));

/** A tool's name (`tool.<name>`) or description (`tool.<name>.about`) in the app's language, else the engine's English. */
function useLabel() {
  const t = useT();
  return (key: string, fallback: string) => {
    const text = t(key as Key);
    return text === key ? fallback : text;
  };
}

/** The Generation panel's tool switches. */
export function ToolsControl({ value, onChange }: { value: string[]; onChange: (tools: string[]) => void }) {
  const t = useT();
  const label = useLabel();
  const [tools, setTools] = useState<ToolInfo[] | null>(null);
  useEffect(() => void toolList().then(setTools), []);
  if (!tools?.length) return null;
  return (
    <div className="flex flex-col gap-1 border-t border-line pt-4">
      <h3 className="flex items-center gap-1.5 text-[13px] font-semibold">
        <Wrench className="size-3.5" /> {t("tools.title")}
      </h3>
      {tools.map((tool) => (
        <div key={tool.name}>
          <Toggle label={label(`tool.${tool.name}`, tool.label)} checked={value.includes(tool.name)} onChange={(on) => onChange(on ? [...value, tool.name] : value.filter((n) => n !== tool.name))} />
          <p className="text-[12px] text-muted">{label(`tool.${tool.name}.about`, tool.description)}</p>
        </div>
      ))}
      <p className="mt-1 text-[12px] text-muted">{t("tools.hint")}</p>
    </div>
  );
}

const shown = (args: ToolStep["arguments"]) => (typeof args === "string" ? args : Object.values(args).length === 1 ? String(Object.values(args)[0]) : JSON.stringify(args));

/** One tool call in a reply: what it was asked, folded open to show the result. */
export function ToolCard({ step }: { step: ToolStep }) {
  const t = useT();
  const label = useLabel();
  const [open, setOpen] = useState(false);
  const running = step.result === null;
  const name = label(`tool.${step.name}`, step.name);
  return (
    <div className="not-prose my-2 rounded-lg border border-line bg-surface text-[12.5px]">
      <button onClick={() => setOpen(!open)} disabled={running} aria-expanded={open} className="flex w-full items-center gap-2 px-3 py-2 text-left">
        {running ? <LoaderCircle className="size-3.5 shrink-0 animate-spin text-muted" /> : <Wrench className={cx("size-3.5 shrink-0", step.error ? "text-danger" : "text-muted")} />}
        <span className="shrink-0 font-medium">{running ? t("tools.running", { name }) : step.error ? t("tools.failed", { name }) : t("tools.used", { name })}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-muted">{shown(step.arguments)}</span>
        {!running && <ChevronRight className={cx("size-3 shrink-0 text-muted transition-transform", open && "rotate-90")} />}
      </button>
      {open && !running && (
        <div className="flex flex-col gap-2 border-t border-line px-3 py-2">
          <div>
            <p className="text-[11.5px] font-medium text-muted">{t("tools.arguments")}</p>
            <pre className="mt-0.5 font-mono text-[12px] break-all whitespace-pre-wrap">{typeof step.arguments === "string" ? step.arguments : JSON.stringify(step.arguments, null, 2)}</pre>
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
export function ReplyWithTools({ text, steps }: { text: string; steps: ToolStep[] }) {
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
                <ToolCard key={s.id} step={s} />
              ))}
          </Fragment>
        );
      })}
      {text.slice(start).trim() && <ReplyText text={withoutCallText(text.slice(start))} />}
    </>
  );
}
