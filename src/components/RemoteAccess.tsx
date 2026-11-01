import { Globe } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { Qr } from "./LanAccess";
import { Button, CopyButton, Notice, ProgressBar, Toggle } from "./ui";

/** The engine's Cloudflare quick tunnel (tunnel.py). */
interface Tunnel {
  state: "off" | "installing" | "starting" | "on" | "failed";
  url: string | null;
  error: string | null;
  progress: number | null;
  tools: boolean;
  installed: boolean;
  accounts: boolean;
}

/** Connect → Remote access: an HTTPS link to this Temper from anywhere, behind the accounts' sign-in. */
export function RemoteAccess() {
  const t = useT();
  const go = useApp((s) => s.go);
  const [tunnel, setTunnel] = useState<Tunnel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(() => void api<Tunnel>("/api/tunnel").then(setTunnel, () => {}), []);
  const busy = tunnel?.state === "installing" || tunnel?.state === "starting";

  useEffect(reload, [reload]);
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(reload, 700);
    return () => clearInterval(timer);
  }, [busy, reload]);
  if (!tunnel) return null;

  async function run(action: () => Promise<Tunnel>) {
    setError(null);
    try {
      setTunnel(await action());
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const on = tunnel.state !== "off" && tunnel.state !== "failed";
  return (
    <div>
      {!tunnel.accounts ? (
        <Notice>
          {t("remote.needs_accounts")}{" "}
          <button className="underline" onClick={() => go("settings")}>
            {t("accounts.title")}
          </button>
        </Notice>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <Button variant={on ? "secondary" : "primary"} disabled={busy} onClick={() => void run(() => api<Tunnel>("/api/tunnel", { json: { enabled: !on } }))}>
            <Globe className="size-3.5" />
            {tunnel.state === "installing" ? t("remote.installing") : tunnel.state === "starting" ? t("remote.starting") : on ? t("remote.stop") : t("remote.start")}
          </Button>
          <Toggle checked={tunnel.tools} onChange={(allowed) => void run(() => api<Tunnel>("/api/tunnel/tools", { method: "PUT", json: { allowed } }))} label={t("remote.tools")} />
          <span className="text-[12.5px] text-muted">{t(tunnel.tools ? "remote.tools_on" : "remote.tools_off")}</span>
        </div>
      )}
      {tunnel.state === "installing" && tunnel.progress !== null && (
        <div className="mt-2 max-w-[320px]">
          <ProgressBar value={tunnel.progress} />
        </div>
      )}
      {tunnel.state === "on" && tunnel.url && (
        <div className="mt-3 flex items-start gap-4 rounded-lg border border-line bg-surface p-3">
          <Qr text={tunnel.url} />
          <div className="min-w-0 text-[13px]">
            <div className="flex items-center gap-1.5">
              <span className="truncate font-mono text-[12.5px]">{tunnel.url}</span>
              <CopyButton text={tunnel.url} />
            </div>
            <p className="mt-1.5 text-muted">{t("remote.link_hint")}</p>
          </div>
        </div>
      )}
      {(error || (tunnel.state === "failed" && tunnel.error)) && <pre className="mt-2 text-[12px] whitespace-pre-wrap text-danger">{error ?? tunnel.error}</pre>}
    </div>
  );
}
