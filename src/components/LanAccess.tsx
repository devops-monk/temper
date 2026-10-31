import qrcode from "qrcode-generator";
import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/engine";
import { Button, CopyButton, Input, Notice, Toggle } from "./ui";

/** The engine's LAN listener (lan.py). */
interface Lan {
  enabled: boolean;
  port: number;
  running: boolean;
  error: string | null;
  addresses: string[];
  url: string | null;
}

function Qr({ text }: { text: string }) {
  const svg = useMemo(() => {
    const qr = qrcode(0, "M");
    qr.addData(text);
    qr.make();
    return qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
  }, [text]);
  // Dark modules on white whatever the theme, so phone cameras read it.
  return <div className="size-36 shrink-0 rounded-md bg-white p-1" role="img" aria-label={`QR code for ${text}`} dangerouslySetInnerHTML={{ __html: svg }} />;
}

/** Turns the model API on for other devices on the home network, with the address as text and a QR code. */
export function LanAccess() {
  const [lan, setLan] = useState<Lan | null>(null);
  const [port, setPort] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api<Lan>("/api/lan").then((l) => {
      setLan(l);
      setPort(String(l.port));
    }, () => {});
  }, []);
  if (!lan) return null;

  async function save(enabled: boolean) {
    setBusy(true);
    setError(null);
    try {
      setLan(await api<Lan>("/api/lan", { json: { enabled, port: Number(port) || lan!.port } }));
    } catch (e) {
      setError((e as Error).message);
      setLan(await api<Lan>("/api/lan"));
    } finally {
      setBusy(false);
    }
  }

  const v1 = lan.url ? `${lan.url}/v1` : null;
  return (
    <div>
      <div className="flex flex-wrap items-center gap-4">
        <Toggle checked={lan.enabled && lan.running} onChange={(on) => void save(on)} label="Serve the model API on this network" />
        <label className="flex items-center gap-2 text-[13px] text-muted">
          Port
          <span className="w-24">
            <Input value={port} onChange={(e) => setPort(e.target.value.replace(/\D/g, ""))} inputMode="numeric" aria-label="Network port" disabled={busy} />
          </span>
        </label>
        {lan.enabled && lan.running && Number(port) !== lan.port && (
          <Button size="sm" onClick={() => void save(true)} disabled={busy}>
            Use this port
          </Button>
        )}
      </div>
      {error && (
        <div className="mt-3">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
      {lan.enabled && lan.running && v1 && (
        <div className="mt-4 flex items-start gap-5">
          <Qr text={v1} />
          <div className="min-w-0 text-[13px]">
            <div className="flex items-center gap-2">
              <span className="font-mono text-[12.5px]">{v1}</span>
              <CopyButton text={v1} />
            </div>
            {lan.addresses.length > 1 && (
              <p className="mt-1 text-[12px] text-muted">
                Also on {lan.addresses.slice(1).map((a) => `http://${a}:${lan.port}/v1`).join(", ")}
              </p>
            )}
            <p className="mt-3 text-muted">
              Scan it on a phone, or type it into an app on another computer as the OpenAI or Anthropic base URL, with an API key from below. Only the model API is served; Temper itself can
              only be managed from this computer. Devices on the same Wi-Fi or wired network can reach it, so turn it off on networks you don't trust.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
