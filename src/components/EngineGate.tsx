import { RotateCw } from "lucide-react";
import { inTauri, restartEngine } from "../lib/engine";
import { useApp } from "../lib/store";
import { LogoMark } from "./Logo";
import { Setup } from "./Setup";
import { Button } from "./ui";

/** Shown in place of a page until the engine answers. */
export function EngineGate() {
  const { engine, refreshEngine } = useApp();
  const status = engine?.status ?? "starting";

  if (status === "needs_setup" || status === "installing" || status === "setup_failed") return <Setup />;

  if (status === "starting") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4">
        <LogoMark className="animate-pulse-dot size-12" />
        <p className="text-muted">{engine?.detail ?? "Starting the engine…"}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="max-w-[460px]">
        <LogoMark className="mb-4 size-10" />
        <h1 className="text-xl font-semibold">
          {status === "missing_uv" ? "Temper needs uv to run its engine" : "The engine isn't running"}
        </h1>
        <p className="mt-2 text-muted">
          {status === "missing_uv"
            ? "uv installs and runs the Python engine that loads and trains models. Install it, then start the engine again."
            : (engine?.detail ?? "The engine stopped. Start it again to keep working.")}
        </p>
        {status === "missing_uv" && (
          <pre className="mt-4 overflow-x-auto rounded-lg border border-line bg-surface p-3 font-mono text-[12.5px]">
            {navigator.userAgent.includes("Windows")
              ? 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
              : "curl -LsSf https://astral.sh/uv/install.sh | sh"}
          </pre>
        )}
        <div className="mt-5 flex gap-2">
          <Button
            variant="primary"
            onClick={async () => {
              await restartEngine();
              await refreshEngine();
            }}
          >
            <RotateCw className="size-4" /> {inTauri ? "Start engine" : "Check again"}
          </Button>
        </div>
      </div>
    </div>
  );
}
