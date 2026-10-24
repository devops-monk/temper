import { useEffect, useState } from "react";
import { api } from "../lib/engine";
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

export const SOURCE_NAME: Record<Source, string> = { huggingface: "Hugging Face", mirror: "a Hugging Face mirror", modelscope: "ModelScope" };

const OPTIONS: [Source, string][] = [
  ["huggingface", "Hugging Face"],
  ["mirror", "Mirror"],
  ["modelscope", "ModelScope"],
];

const ABOUT: Record<Source, string> = {
  huggingface: "Models download straight from huggingface.co.",
  mirror: "Same models and names as Hugging Face, through a mirror such as hf-mirror.com or one your company runs. Training and export use it too. If you've added a Hugging Face token, it's sent to the mirror, so only use mirrors you trust.",
  modelscope:
    "ModelScope hosts most popular models, often under the publisher's name. Downloads go into the same library, so chat and training work as usual. Gated models and datasets still come from Hugging Face; local dataset files always work.",
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
      title="Model source"
      aside={
        <Button size="sm" disabled={checking} onClick={() => void test()}>
          {checking ? "Checking…" : "Check connection"}
        </Button>
      }
    >
      <div className="inline-flex rounded-lg border border-line bg-surface p-1" role="radiogroup" aria-label="Model source">
        {OPTIONS.map(([value, label]) => (
          <button
            key={value}
            role="radio"
            aria-checked={hub.source === value}
            onClick={() => hub.source !== value && void save(value)}
            className={cx("rounded-md px-3 py-1.5 text-[13px] font-medium", hub.source === value ? "bg-surface-2 text-text shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:text-text")}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="mt-2 text-[12.5px] text-muted">{ABOUT[hub.source]}</p>
      {hub.source === "mirror" && (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save("mirror");
          }}
        >
          <Input value={mirror} onChange={(e) => setMirror(e.target.value)} placeholder="https://hf-mirror.com" spellCheck={false} className="flex-1" aria-label="Mirror address" />
          <Button type="submit" disabled={!mirrorChanged}>
            Save
          </Button>
        </form>
      )}
      {check && (
        <p className={cx("mt-2 text-[12.5px]", check.ok ? "text-ok" : "text-danger")}>
          {check.ok ? `Connected, answered in ${check.ms} ms.` : `Couldn't connect${check.error ? ` (${check.error})` : check.status ? ` (HTTP ${check.status})` : ""}. Try another source.`}
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
