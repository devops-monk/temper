import { Link2, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { Badge, Button, CopyButton, Input, cx } from "./ui";

/** A linked computer running Temper, as the engine lists it (engine devices.py). Its key is never sent back. */
export interface Device {
  id: string;
  name: string;
  url: string;
  key_hint: string;
  online: boolean;
  error: string | null;
  models: { id: string; loaded: boolean }[];
  preferred: boolean;
}

/** The linked devices, asked again every `every` ms. */
export function useDevices(every = 30_000): { devices: Device[]; reload: (fresh?: boolean) => void } {
  const [devices, setDevices] = useState<Device[]>([]);
  const reload = useCallback((fresh = false) => void api<{ devices: Device[] }>(`/api/devices${fresh ? "?fresh=true" : ""}`).then((r) => setDevices(r.devices), () => {}), []);
  useEffect(() => {
    reload();
    const t = setInterval(reload, every);
    return () => clearInterval(t);
  }, [reload, every]);
  return { devices, reload };
}

/** Connect → Linked devices: other computers' models, used from this one. */
export function Devices() {
  const { devices, reload } = useDevices();
  const [url, setUrl] = useState("");
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function link() {
    setBusy(true);
    setError(null);
    try {
      await api("/api/devices", { json: { url: url.trim(), key: key.trim(), name: name.trim() || null } });
      setUrl("");
      setKey("");
      setName("");
      reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function unlink(d: Device) {
    if (!confirm(`Unlink ${d.name}? Its models stop showing here. The key stays valid on ${d.name} until you revoke it there.`)) return;
    await api(`/api/devices/${d.id}`, { method: "DELETE" });
    reload();
  }

  async function prefer(d: Device | null) {
    await api("/api/devices/preferred", { method: "PUT", json: { device: d?.id ?? null } });
    reload();
  }

  return (
    <div>
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (url.trim() && key.trim()) void link();
        }}
      >
        <span className="w-56">
          <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="http://192.168.1.20:8765" aria-label="The other computer's address" />
        </span>
        <span className="w-56">
          <Input value={key} onChange={(e) => setKey(e.target.value)} placeholder="An API key made there" aria-label="Its API key" type="password" />
        </span>
        <span className="w-36">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name (optional)" aria-label="What to call it" maxLength={40} />
        </span>
        <Button type="submit" disabled={busy || !url.trim() || !key.trim()}>
          <Link2 className="size-3.5" /> {busy ? "Linking…" : "Link"}
        </Button>
      </form>
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}

      {devices.length > 0 && (
        <ul className="mt-3 divide-y divide-line rounded-lg border border-line bg-surface">
          {devices.map((d) => (
            <li key={d.id} className="px-4 py-2.5 text-[13px]">
              <div className="flex items-center gap-2">
                <span className="font-medium">{d.name}</span>
                <span className="font-mono text-[12px] text-muted">{d.url}</span>
                {d.online ? <Badge tone="ok">Online</Badge> : <Badge tone="warn">Offline</Badge>}
                <span className="flex-1" />
                <label className="flex items-center gap-1.5 text-[12px] text-muted" title="Answers for models both computers have">
                  <input type="checkbox" checked={d.preferred} onChange={() => void prefer(d.preferred ? null : d)} />
                  Preferred
                </label>
                <button className="text-muted hover:text-danger" onClick={() => void unlink(d)} title={`Unlink ${d.name}`} aria-label={`Unlink ${d.name}`}>
                  <Trash2 className="size-3.5" />
                </button>
              </div>
              {d.error && <p className="text-[12px] text-muted">{d.error}</p>}
              {d.models.length > 0 && (
                <ul className="mt-1.5 flex flex-wrap gap-1.5">
                  {d.models.map((m) => (
                    <li key={m.id} className={cx("flex items-center gap-1 rounded border border-line px-1.5 py-0.5 font-mono text-[11.5px]", !m.loaded && "text-muted")} title={m.loaded ? "Loaded there" : "Loads there when asked for"}>
                      {m.id}@{d.name}
                      <CopyButton text={`${m.id}@${d.name}`} />
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
