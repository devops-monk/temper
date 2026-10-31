import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Input, Notice, Toggle } from "./ui";

/** The engine's discovery listeners (discovery.py), by name. */
type Discovery = Record<"ollama" | "llamacpp", { enabled: boolean; running: boolean; port: number; error: string | null }>;

const LABELS = {
  ollama: { label: "discovery.ollama", hint: "discovery.ollama_hint" },
  llamacpp: { label: "discovery.llamacpp", hint: "discovery.llamacpp_hint" },
} as const;

/** Lets Temper answer at the addresses other apps look for local models (Ollama's, llama-server's). */
export function DiscoverySettings() {
  const t = useT();
  const [state, setState] = useState<Discovery | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [port, setPort] = useState("");

  useEffect(() => {
    void api<Discovery>("/api/discovery").then((d) => {
      setState(d);
      setPort(String(d.llamacpp.port));
    }, () => {});
  }, []);
  if (!state) return null;

  async function set(name: keyof Discovery, on: boolean) {
    setError(null);
    try {
      setState(await api<Discovery>("/api/discovery", { json: { [name]: on, ...(name === "llamacpp" && Number(port) ? { llamacpp_port: Number(port) } : {}) } }));
    } catch (e) {
      setError((e as Error).message);
      setState(await api<Discovery>("/api/discovery"));
    }
  }

  return (
    <div className="space-y-3">
      {(Object.keys(LABELS) as (keyof Discovery)[]).map((name) => (
        <div key={name} className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <Toggle checked={state[name].enabled && state[name].running} onChange={(on) => void set(name, on)} label={t(LABELS[name].label)} />
          <span className="text-[12.5px] text-muted">
            {t(LABELS[name].hint)} {name === "ollama" && <span className="font-mono text-[12px]">localhost:{state.ollama.port}</span>}
          </span>
          {name === "llamacpp" && (
            <label className="flex items-center gap-2 text-[12.5px] text-muted">
              {t("discovery.port")}
              <span className="w-24">
                <Input
                  value={port}
                  onChange={(e) => setPort(e.target.value.replace(/\D/g, ""))}
                  onBlur={() => state.llamacpp.enabled && Number(port) !== state.llamacpp.port && void set("llamacpp", true)}
                  inputMode="numeric"
                  aria-label={t("discovery.port_label")}
                />
              </span>
            </label>
          )}
        </div>
      ))}
      {error && <Notice tone="error">{error}</Notice>}
    </div>
  );
}
