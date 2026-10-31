import { KeyRound, RotateCw } from "lucide-react";
import { useState } from "react";
import { inTauri, restartEngine, saveEngineKey } from "../lib/engine";
import { useApp } from "../lib/store";
import { LogoMark } from "./Logo";
import { Setup } from "./Setup";
import { Button, Input } from "./ui";

/** Shown in place of a page until the engine answers. */
export function EngineGate() {
  const { engine, refreshEngine } = useApp();
  const status = engine?.status ?? "starting";

  if (status === "needs_setup" || status === "installing" || status === "setup_failed") return <Setup />;
  if (status === "locked") return <KeyForm wrong={!!engine?.token} onSaved={refreshEngine} />;

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

/** A browser on `temper serve` asks for the engine's key once, then keeps it. */
function KeyForm({ wrong, onSaved }: { wrong: boolean; onSaved: () => Promise<void> }) {
  const [key, setKey] = useState("");
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <form
        className="w-full max-w-[420px]"
        onSubmit={async (e) => {
          e.preventDefault();
          saveEngineKey(key);
          await onSaved();
        }}
      >
        <LogoMark className="mb-4 size-10" />
        <h1 className="text-xl font-semibold">{wrong ? "That key didn't work" : "Enter the engine's key"}</h1>
        <p className="mt-2 text-muted">
          It&apos;s in the api-key file in Temper&apos;s folder on the computer running <span className="font-mono">temper serve</span>, and in the link it printed. This browser keeps it.
        </p>
        <div className="mt-4 flex gap-2">
          <Input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="Key" aria-label="Engine key" autoFocus autoComplete="current-password" />
          <Button variant="primary" type="submit" disabled={!key.trim()}>
            <KeyRound className="size-4" /> Open
          </Button>
        </div>
      </form>
    </div>
  );
}
