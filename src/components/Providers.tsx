import { Cloud, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { LoadButton } from "./LoadButton";
import { Badge, Button, CopyButton, Input, Select } from "./ui";

/** A cloud provider, as the engine lists it (engine providers.py). Its key is never sent back. */
export interface Provider {
  id: string;
  name: string;
  kind: string;
  url: string;
  key_hint: string;
  online: boolean;
  error: string | null;
  available: string[];
  models: string[]; // the ones chosen to list; none means all of them
}

const KINDS = ["openai", "anthropic", "google", "openrouter", "custom"] as const;
const KIND_NAME: Record<string, string> = { openai: "OpenAI", anthropic: "Anthropic", google: "Google Gemini", openrouter: "OpenRouter" };
const SHOWN = 30;

/** The models a provider lists: the chosen ones, or all it offers. */
export function listed(p: Provider): string[] {
  return p.models.length ? p.models : p.available;
}

export function useProviders(): { providers: Provider[]; reload: (fresh?: boolean) => void } {
  const [providers, setProviders] = useState<Provider[]>([]);
  const reload = useCallback((fresh = false) => void api<{ providers: Provider[] }>(`/api/providers${fresh ? "?fresh=true" : ""}`).then((r) => setProviders(r.providers), () => {}), []);
  useEffect(() => reload(), [reload]);
  return { providers, reload };
}

/** Connect → Cloud providers: OpenAI, Anthropic, Google, OpenRouter or any OpenAI-compatible API, used next to local models. */
export function Providers() {
  const { providers, reload } = useProviders();
  const t = useT();
  const [kind, setKind] = useState<string>("openai");
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const kindName = (k: string) => KIND_NAME[k] ?? t("providers.custom");

  async function add() {
    setBusy(true);
    setError(null);
    try {
      await api("/api/providers", { json: { kind, key: key.trim(), name: name.trim() || null, url: url.trim() || null } });
      setKey("");
      setName("");
      setUrl("");
      reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(p: Provider) {
    if (!confirm(t("providers.remove_confirm", { name: p.name }))) return;
    await api(`/api/providers/${p.id}`, { method: "DELETE" });
    reload();
  }

  return (
    <div>
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (key.trim() && (kind !== "custom" || url.trim())) void add();
        }}
      >
        <span className="w-48">
          <Select value={kind} onChange={(e) => setKind(e.target.value)} aria-label={t("providers.kind")}>
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {kindName(k)}
              </option>
            ))}
          </Select>
        </span>
        <span className="w-56">
          <Input value={key} onChange={(e) => setKey(e.target.value)} placeholder={t("providers.key_placeholder")} aria-label={t("providers.key")} type="password" />
        </span>
        <span className="w-44">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={kind === "custom" ? t("providers.name_placeholder") : kind} aria-label={t("providers.name")} maxLength={40} />
        </span>
        {kind === "custom" && (
          <span className="w-64">
            <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://api.example.com/v1" aria-label={t("providers.url")} />
          </span>
        )}
        <Button type="submit" disabled={busy || !key.trim() || (kind === "custom" && !url.trim())}>
          <Cloud className="size-3.5" /> {busy ? t("providers.adding") : t("providers.add")}
        </Button>
      </form>
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}

      {providers.length > 0 && (
        <ul className="mt-3 divide-y divide-line rounded-lg border border-line bg-surface">
          {providers.map((p) => (
            <ProviderRow key={p.id} provider={p} kindName={kindName(p.kind)} onRemove={() => void remove(p)} reload={reload} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ProviderRow({ provider: p, kindName, onRemove, reload }: { provider: Provider; kindName: string; onRemove: () => void; reload: () => void }) {
  const t = useT();
  const [filter, setFilter] = useState("");
  const all = [...new Set([...p.available, ...p.models])].sort();
  const matching = all.filter((m) => m.toLowerCase().includes(filter.trim().toLowerCase()));
  const isListed = (m: string) => (p.models.length ? p.models.includes(m) : true);

  async function toggle(m: string) {
    const now = listed(p);
    const next = isListed(m) ? now.filter((x) => x !== m) : [...now, m];
    // Listing every model again is the same as choosing none.
    await api(`/api/providers/${p.id}/models`, { method: "PUT", json: { models: next.length === p.available.length && p.available.every((x) => next.includes(x)) ? [] : next } });
    reload();
  }

  return (
    <li className="px-4 py-2.5 text-[13px]">
      <div className="flex items-center gap-2">
        <span className="font-medium">{p.name}</span>
        <span className="text-[12px] text-muted">{kindName}</span>
        <span className="truncate font-mono text-[12px] text-muted">{p.url}</span>
        <span className="font-mono text-[11.5px] text-muted">{p.key_hint}</span>
        {p.online ? <Badge tone="ok">{t("providers.connected")}</Badge> : <Badge tone="warn">{t("providers.unreachable")}</Badge>}
        <span className="flex-1" />
        <button className="text-muted hover:text-danger" onClick={onRemove} title={t("providers.remove", { name: p.name })} aria-label={t("providers.remove", { name: p.name })}>
          <Trash2 className="size-3.5" />
        </button>
      </div>
      {p.error && <p className="text-[12px] text-muted">{p.error}</p>}
      {all.length > 0 && (
        <>
          <div className="mt-2 flex items-center gap-2">
            <span className="w-56">
              <Input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={t("providers.filter", { count: all.length })} aria-label={t("providers.filter_label")} />
            </span>
            <span className="text-[12px] text-muted">{t("providers.listed", { count: listed(p).length })}</span>
          </div>
          <ul className="mt-1.5">
            {matching.slice(0, SHOWN).map((m) => (
              <li key={m} className="flex items-center gap-2 py-1">
                <label className="flex items-center gap-1.5" title={t("providers.list_title")}>
                  <input type="checkbox" checked={isListed(m)} onChange={() => void toggle(m)} aria-label={t("providers.list_model", { model: m })} />
                  <span className="font-mono text-[12px]">{m}</span>
                </label>
                <CopyButton text={`${m}@${p.name}`} />
                <span className="flex-1" />
                <LoadButton path={`${m}@${p.name}`} name={`${m}@${p.name}`} label={t("providers.use")} />
              </li>
            ))}
          </ul>
          {matching.length > SHOWN && <p className="text-[12px] text-muted">{t("providers.more", { count: matching.length - SHOWN })}</p>}
        </>
      )}
    </li>
  );
}
