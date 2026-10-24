import { Check, ChevronRight, Circle, Cpu, LoaderCircle, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { installEngine } from "../lib/engine";
import { useApp } from "../lib/store";
import type { SetupProfile, SetupState, SetupStep } from "../lib/types";
import { LogoMark } from "./Logo";
import { Badge, Button, CopyButton, Notice, ProgressBar, cx } from "./ui";

const HARDWARE_LABEL = { apple: "Apple Silicon", nvidia: "NVIDIA GPU", amd: "AMD GPU", none: "No supported GPU" } as const;

function StepIcon({ state }: { state: SetupStep["state"] }) {
  if (state === "done") return <Check className="size-4 text-ok" aria-label="Done" />;
  if (state === "running") return <LoaderCircle className="size-4 animate-spin text-accent" aria-label="In progress" />;
  if (state === "failed") return <X className="size-4 text-danger" aria-label="Failed" />;
  return <Circle className="size-3.5 text-line" aria-label="Waiting" />;
}

function Log({ lines, open }: { lines: string[]; open: boolean }) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines.length, open]);
  return (
    <details open={open} className="mt-4 text-[12.5px]">
      <summary className="flex cursor-pointer items-center justify-between text-muted">
        <span>Details</span>
        <CopyButton text={lines.join("\n")} label="Copy log" />
      </summary>
      <pre ref={ref} className="mt-2 h-56 overflow-auto rounded-lg border border-line bg-surface-2 p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap text-muted">
        {lines.join("\n")}
      </pre>
    </details>
  );
}

function Chooser({ setup, onInstall }: { setup: SetupState; onInstall: (p: SetupProfile) => void }) {
  const recommended = setup.options.find((o) => o.recommended && o.available)?.id ?? "chat";
  const [choice, setChoice] = useState<SetupProfile>(recommended);
  const hw = setup.hardware;

  return (
    <>
      <h1 className="text-[22px] leading-tight font-semibold tracking-[-0.01em]">Set up Temper</h1>
      <p className="mt-2 text-muted">
        Temper keeps its engine, Python and the packages that run models, in its own folder. Nothing else on your computer changes.
      </p>
      {hw && (
        <p className="mt-4 flex items-center gap-2 text-[13px]">
          <Cpu className="size-4 text-muted" />
          <span>
            {hw.name ?? HARDWARE_LABEL[hw.kind]}
            {hw.name && <span className="text-muted"> ({HARDWARE_LABEL[hw.kind]})</span>}
          </span>
        </p>
      )}
      <div className="mt-5 flex flex-col gap-2" role="radiogroup" aria-label="What to install">
        {setup.options.map((o) => (
          <button
            key={o.id}
            role="radio"
            aria-checked={choice === o.id}
            disabled={!o.available}
            onClick={() => setChoice(o.id)}
            className={cx(
              "rounded-xl border px-4 py-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50",
              choice === o.id ? "border-accent bg-accent/6" : "border-line bg-surface hover:bg-surface-2",
            )}
          >
            <span className="flex items-center gap-2">
              <span className="font-semibold">{o.label}</span>
              {o.recommended && o.available && <Badge tone="accent">Recommended</Badge>}
              <span className="ml-auto text-[12.5px] text-muted">{o.size}</span>
            </span>
            <span className="mt-0.5 block text-[13px] text-muted">{o.description}</span>
          </button>
        ))}
      </div>
      <Button variant="primary" className="mt-6 w-full" onClick={() => onInstall(choice)}>
        Install
      </Button>
    </>
  );
}

function Progress({ setup, failed, onRetry, onChooseAgain }: { setup: SetupState; failed: boolean; onRetry: () => void; onChooseAgain: () => void }) {
  const running = setup.steps.find((s) => s.state === "running");
  const last = [...setup.log].reverse().find((l) => !l.startsWith("$ ") && !l.startsWith("== "));
  const heavy = setup.profile === "train" && setup.hardware?.kind !== "apple";

  return (
    <>
      <h1 className="text-[22px] leading-tight font-semibold tracking-[-0.01em]">{failed ? "Setup didn't finish" : "Setting up Temper"}</h1>
      <p className="mt-2 text-muted">
        {failed
          ? "Fix the problem below and try again. Anything already downloaded is reused."
          : heavy
            ? "This downloads about 6 GB of GPU libraries, so it can take a while."
            : "This takes a minute or two on a typical connection."}
      </p>
      <ol className="mt-6 flex flex-col gap-3">
        {setup.steps.map((step) => (
          <li key={step.id}>
            <div className="flex items-center gap-3">
              <span className="flex size-5 items-center justify-center">
                <StepIcon state={step.state} />
              </span>
              <span className={cx(step.state === "pending" && "text-muted", step.state === "running" && "font-medium")}>{step.label}</span>
            </div>
            {step === running && (
              <div className="mt-2 ml-8">
                <ProgressBar value={setup.progress} />
                {last && <p className="mt-1.5 truncate font-mono text-[11.5px] text-muted">{last}</p>}
              </div>
            )}
          </li>
        ))}
      </ol>
      {failed && (
        <>
          <div className="mt-5">
            <Notice tone="error">{setup.error}</Notice>
          </div>
          <div className="mt-4 flex gap-2">
            <Button variant="primary" onClick={onRetry}>
              Try again
            </Button>
            <Button variant="ghost" onClick={onChooseAgain}>
              Choose a different setup <ChevronRight className="size-4" />
            </Button>
          </div>
        </>
      )}
      <Log lines={setup.log} open={failed} />
    </>
  );
}

/** First-run setup, shown while the shell installs the engine environment. */
export function Setup() {
  const { engine, refreshEngine } = useApp();
  const [choosing, setChoosing] = useState(false);
  const setup = engine?.setup;
  if (!engine || !setup) return null;

  const install = async (profile: SetupProfile) => {
    setChoosing(false);
    await installEngine(profile);
    await refreshEngine();
  };
  const showChooser = engine.status === "needs_setup" || (engine.status === "setup_failed" && choosing);

  return (
    <div className="flex flex-1 justify-center overflow-y-auto px-6 py-12">
      <div className="my-auto w-full max-w-[520px]">
        <LogoMark className="mb-5 size-10" />
        {showChooser ? (
          <Chooser setup={setup} onInstall={install} />
        ) : (
          <Progress
            setup={setup}
            failed={engine.status === "setup_failed"}
            onRetry={() => install(setup.profile ?? "chat")}
            onChooseAgain={() => setChoosing(true)}
          />
        )}
      </div>
    </div>
  );
}
