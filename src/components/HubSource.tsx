import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { type Key, t as tr, useT } from "../lib/i18n";
import { Button, Input, Notice, Section, cx } from "./ui";

export type Source = "huggingface" | "mirror" | "modelscope";

export interface HubSettings {
  source: Source;
  mirror: string;
  endpoint: string;
}

interface Check {
  ok: boolean;
  ms?: number;
  status?: number;
  error?: string;
}

export const sourceName = (source: Source) => (source === "mirror" ? tr("hub.a_mirror") : source === "modelscope" ? "ModelScope" : "Hugging Face");

const OPTIONS: [Source, string | Key][] = [
  ["huggingface", "Hugging Face"],
  ["mirror", "hub.mirror"],
  ["modelscope", "ModelScope"],
];

const ABOUT: Record<Source, Key> = {
  huggingface: "hub.about_hf",
  mirror: "hub.about_mirror",
  modelscope: "hub.about_ms",
};

/** Where models come from, read once per page. */
export function useHubSource(ready = true) {
  const [hub, setHub] = useState<HubSettings | null>(null);
  useEffect(() => {
    if (ready) api<HubSettings>("/api/hub/source").then(setHub, () => setHub(null));
  }, [ready]);
  return [hub, setHub] as const;
}

/** Settings: Hugging Face, a mirror, or ModelScope, for places where Hugging Face is slow or blocked. */
export function HubSource({ ready }: { ready: boolean }) {
  const t = useT();
  const [hub, setHub] = useHubSource(ready);
  const [mirror, setMirror] = useState("");
  const [check, setCheck] = useState<Check | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (hub) setMirror(hub.mirror);
  }, [hub]);

  async function save(source: Source, mirrorUrl = mirror) {
    setError(null);
    setCheck(null);
    try {
      const next = await api<HubSettings>("/api/hub/source", { method: "PUT", json: { source, mirror: mirrorUrl } });
      setHub(next);
      if (source !== "huggingface") void test(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function test(target = hub) {
    if (!target) return;
    setChecking(true);
    try {
      setCheck(await api<Check>(`/api/hub/source/check?source=${target.source}&mirror=${encodeURIComponent(target.mirror)}`));
    } finally {
      setChecking(false);
    }
  }

  if (!hub) return null;
  const mirrorChanged = mirror.trim().replace(/\/+$/, "") !== hub.mirror;
  return (
    <Section
      title={t("hub.title")}
      aside={
        <Button size="sm" disabled={checking} onClick={() => void test()}>
          {t(checking ? "hub.checking" : "hub.check")}
        </Button>
      }
    >
      <div className="inline-flex rounded-lg border border-line bg-surface p-1" role="radiogroup" aria-label={t("hub.title")}>
        {OPTIONS.map(([value, label]) => (
          <button
            key={value}
            role="radio"
            aria-checked={hub.source === value}
            onClick={() => hub.source !== value && void save(value)}
            className={cx("rounded-md px-3 py-1.5 text-[13px] font-medium", hub.source === value ? "bg-surface-2 text-text shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:text-text")}
          >
            {label === "hub.mirror" ? t(label) : label}
          </button>
        ))}
      </div>
      <p className="mt-2 text-[12.5px] text-muted">{t(ABOUT[hub.source])}</p>
      {hub.source === "mirror" && (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save("mirror");
          }}
        >
          <Input value={mirror} onChange={(e) => setMirror(e.target.value)} placeholder="https://hf-mirror.com" spellCheck={false} className="flex-1" aria-label={t("hub.mirror_address")} />
          <Button type="submit" disabled={!mirrorChanged}>
            {t("hub.save")}
          </Button>
        </form>
      )}
      {check && (
        <p className={cx("mt-2 text-[12.5px]", check.ok ? "text-ok" : "text-danger")}>
          {check.ok ? t("hub.connected", { ms: check.ms ?? 0 }) : t("hub.failed", { why: check.error ? ` (${check.error})` : check.status ? ` (HTTP ${check.status})` : "" })}
        </p>
      )}
      {error && (
        <div className="mt-3">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
    </Section>
  );
}
