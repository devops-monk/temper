import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Button, Field, Input, Section, Select } from "./ui";

/** The search provider settings as the engine shows them (engine search.py); saved keys come back masked. */
interface SearchSettings {
  provider: string;
  providers: { id: string; name: string; needs: "key" | "url" | null; set: boolean }[];
  searxng_url: string;
  keys: Record<string, string>;
}

const SIGN_UP: Record<string, string> = { brave: "https://brave.com/search/api/", tavily: "https://app.tavily.com/" };

/** Settings: which search provider the web_search tool goes through. */
export function WebSearch() {
  const t = useT();
  const [info, setInfo] = useState<SearchSettings | null>(null);
  const [provider, setProvider] = useState("duckduckgo");
  const [key, setKey] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);
  const show = (s: SearchSettings) => (setInfo(s), setProvider(s.provider), setUrl(s.searxng_url));
  useEffect(() => void api<SearchSettings>("/api/search").then(show, () => {}), []);
  useEffect(() => setKey(info?.keys[provider] ?? ""), [provider, info]);
  if (!info) return null;
  const needs = info.providers.find((p) => p.id === provider)?.needs ?? null;
  const name = info.providers.find((p) => p.id === provider)?.name ?? provider;

  async function save() {
    setBusy(true);
    setNote(null);
    try {
      show(await api<SearchSettings>("/api/search", { method: "PUT", json: { provider, key: needs === "key" ? key : undefined, url: needs === "url" ? url : undefined } }));
      const found = await api<{ results: { title: string }[] }>("/api/search/try");
      setNote({ ok: true, text: t("search.works", { name, count: found.results.length }) });
    } catch (e) {
      setNote({ ok: false, text: (e as Error).message });
    }
    setBusy(false);
  }

  return (
    <Section title={t("search.title")}>
      <p className="mb-3 text-[12.5px] text-muted">{t("search.about")}</p>
      <div className="flex flex-col gap-3 text-[13px]">
        <Field label={t("search.provider")} hint={t(`search.${provider}` as "search.duckduckgo")}>
          <Select value={provider} onChange={(e) => (setProvider(e.target.value), setNote(null))}>
            {info.providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </Select>
        </Field>
        {needs === "key" && (
          <Field
            label={t("search.key")}
            hint={
              <a href={SIGN_UP[provider]} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                {t("search.get_key", { name })}
              </a>
            }
          >
            <Input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder={t("search.key")} />
          </Field>
        )}
        {needs === "url" && (
          <Field label={t("search.url")} hint={t("search.url_hint")}>
            <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="http://localhost:8888" />
          </Field>
        )}
        <div className="flex items-center gap-3">
          <Button size="sm" onClick={() => void save()} disabled={busy || (needs === "key" && !key.trim()) || (needs === "url" && !url.trim())}>
            {busy ? t("search.trying") : t("search.save")}
          </Button>
          {note && <span className={note.ok ? "text-[12px] text-ok" : "text-[12px] text-danger"}>{note.text}</span>}
        </div>
      </div>
    </Section>
  );
}
