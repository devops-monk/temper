import { KeyRound, RotateCw } from "lucide-react";
import { useState } from "react";
import { inTauri, restartEngine, saveEngineKey } from "../lib/engine";
import { useT, withNodes } from "../lib/i18n";
import { useApp } from "../lib/store";
import { LogoMark } from "./Logo";
import { Setup } from "./Setup";
import { Button, Input } from "./ui";

/** Shown in place of a page until the engine answers. */
export function EngineGate() {
  const t = useT();
  const { engine, refreshEngine } = useApp();
  const status = engine?.status ?? "starting";

  if (status === "needs_setup" || status === "installing" || status === "setup_failed") return <Setup />;
  if (status === "locked") return <KeyForm wrong={!!engine?.token} onSaved={refreshEngine} />;

  if (status === "starting") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4">
        <LogoMark className="animate-pulse-dot size-12" />
        <p className="text-muted">{engine?.detail ?? t("gate.starting")}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="max-w-[460px]">
        <LogoMark className="mb-4 size-10" />
        <h1 className="text-xl font-semibold">
          {t(status === "missing_uv" ? "gate.needs_uv" : "gate.not_running")}
        </h1>
        <p className="mt-2 text-muted">
          {status === "missing_uv"
            ? t("gate.uv_hint")
            : (engine?.detail ?? t("gate.stopped"))}
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
            <RotateCw className="size-4" /> {t(inTauri ? "gate.start" : "gate.check")}
          </Button>
        </div>
      </div>
    </div>
  );
}

/** A browser on `temper serve` asks for the engine's key once, then keeps it. */
function KeyForm({ wrong, onSaved }: { wrong: boolean; onSaved: () => Promise<void> }) {
  const t = useT();
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
        <h1 className="text-xl font-semibold">{t(wrong ? "gate.wrong_key" : "gate.enter_key")}</h1>
        <p className="mt-2 text-muted">
          {withNodes(t("gate.key_hint"), { cmd: <span className="font-mono">temper serve</span> })}
        </p>
        <div className="mt-4 flex gap-2">
          <Input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder={t("gate.key")} aria-label={t("gate.engine_key")} autoFocus autoComplete="current-password" />
          <Button variant="primary" type="submit" disabled={!key.trim()}>
            <KeyRound className="size-4" /> {t("gate.open")}
          </Button>
        </div>
      </form>
    </div>
  );
}
