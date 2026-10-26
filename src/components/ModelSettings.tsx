import { Cpu, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import type { ChatEstimate } from "./LoadButton";
import { Button, Field, Input, Select, Toggle, cx } from "./ui";

/** How chat loads a model (engine model_settings.py). */
export interface LoadSettings {
  context: number;
  gpu_layers: number;
  flash_attention: boolean;
  cache_type: "f16" | "q8_0" | "q4_0";
  speculation: "none" | "draft" | "ngram" | "mtp";
  draft: string | null;
  draft_max: number;
}

interface Offer {
  settings: LoadSettings;
  context_max: number | null;
  layers: number | null;
  mtp: boolean;
  drafts: { path: string; name: string; label: string; size: number }[];
}

/** How the last reply went, from llama-server's timings. */
export interface ReplyStats {
  speed: number;
  drafted?: number;
  accepted?: number;
}

const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
const ALL_LAYERS = 999;

const same = (a: LoadSettings, b: LoadSettings) => (Object.keys(a) as (keyof LoadSettings)[]).every((k) => a[k] === b[k]);

/** Context length, KV cache, flash attention, GPU offload and speculative decoding for the loaded model, saved per model. */
export function ModelSettings({ stats }: { stats: ReplyStats | null }) {
  const { runtime, refreshRuntime } = useApp();
  const path = runtime?.state === "ready" || runtime?.state === "failed" ? (runtime.path ?? null) : null;
  const [offer, setOffer] = useState<Offer | null>(null);
  const [form, setForm] = useState<LoadSettings | null>(null);
  const [estimate, setEstimate] = useState<ChatEstimate | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!path) return setOffer(null);
    let stale = false;
    api<Offer>(`/api/runtime/settings?path=${encodeURIComponent(path)}`).then(
      (o) => {
        if (stale) return;
        setOffer(o);
        setForm(o.settings);
      },
      () => !stale && setOffer(null),
    );
    return () => {
      stale = true;
    };
  }, [path]);

  const changed = !!offer && !!form && !same(offer.settings, form);
  useEffect(() => {
    if (!path || !form || !changed) return setEstimate(null);
    const timer = setTimeout(() => {
      const { context, cache_type, speculation, draft } = form;
      api<ChatEstimate>("/api/memory/chat", { json: { path, context, cache_type, speculation, draft } }).then(setEstimate, () => setEstimate(null));
    }, 250);
    return () => clearTimeout(timer);
  }, [path, form, changed]);

  if (!path || !offer || !form) return null;
  const set = (change: Partial<LoadSettings>) => setForm((f) => (f ? { ...f, ...change } : f));
  const contexts = [...new Set([...CONTEXTS.filter((c) => !offer.context_max || c <= offer.context_max), ...(offer.context_max ? [offer.context_max] : [])])].sort((a, b) => a - b);
  const layers = offer.layers ?? 0;
  const speedValue = form.speculation === "draft" ? `draft:${form.draft}` : form.speculation;

  async function apply() {
    setSaving(true);
    setError(null);
    try {
      const out = await api<{ settings: LoadSettings }>("/api/runtime/settings", { method: "PUT", json: { path, settings: form } });
      setOffer((o) => (o ? { ...o, settings: out.settings } : o));
      setForm(out.settings);
      await refreshRuntime();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex flex-col gap-3 border-t border-line pt-4">
      <h3 className="flex items-center gap-1.5 text-[13px] font-semibold">
        <Cpu className="size-3.5" /> Model
      </h3>
      <Field label="Context length" hint="How much of the chat the model can see at once. Longer needs more memory.">
        <Select value={form.context} onChange={(e) => set({ context: Number(e.target.value) })}>
          {contexts.map((c) => (
            <option key={c} value={c}>
              {c >= 1024 ? `${Math.round(c / 1024)}K` : c} tokens{c === offer.context_max ? " (the model's maximum)" : ""}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="KV cache" hint={form.cache_type !== "f16" && !form.flash_attention ? "A quantized cache needs flash attention." : "The memory that holds the context."}>
        <Select value={form.cache_type} onChange={(e) => set({ cache_type: e.target.value as LoadSettings["cache_type"] })}>
          <option value="f16">16-bit (best quality)</option>
          <option value="q8_0">8-bit (about half the memory)</option>
          <option value="q4_0">4-bit (about a quarter; may cost quality)</option>
        </Select>
      </Field>
      <Toggle label="Flash attention" checked={form.flash_attention} onChange={(flash_attention) => set({ flash_attention, ...(!flash_attention && { cache_type: "f16" }) })} />
      {layers > 0 && (
        <label className="block">
          <span className="mb-1 flex justify-between text-[13px]">
            <span className="font-medium">GPU offload</span>
            <span className="text-[12px] text-muted">{form.gpu_layers >= layers ? "All layers" : `${form.gpu_layers} of ${layers} layers`}</span>
          </span>
          <input
            type="range"
            min={0}
            max={layers}
            value={Math.min(form.gpu_layers, layers)}
            onChange={(e) => set({ gpu_layers: Number(e.target.value) >= layers ? ALL_LAYERS : Number(e.target.value) })}
            className="w-full accent-[var(--accent)]"
            aria-label="Layers on the GPU"
          />
          <span className="text-[12px] text-muted">Layers left on the CPU free GPU memory but run slower.</span>
        </label>
      )}
      <Field label="Speculative decoding" hint="Drafts tokens ahead so the model checks several at once. Replies are the same, often faster.">
        <Select
          value={speedValue}
          onChange={(e) => {
            const v = e.target.value;
            set(v.startsWith("draft:") ? { speculation: "draft", draft: v.slice(6) } : { speculation: v as LoadSettings["speculation"], draft: null });
          }}
        >
          <option value="none">Off</option>
          <option value="ngram">Prompt lookup (no extra model)</option>
          {offer.mtp && <option value="mtp">The model's own MTP layers</option>}
          {offer.drafts.length > 0 && (
            <optgroup label="Draft model">
              {offer.drafts.map((d) => (
                <option key={d.path} value={`draft:${d.path}`}>
                  {d.label} ({bytes(d.size)})
                </option>
              ))}
            </optgroup>
          )}
        </Select>
      </Field>
      {offer.drafts.length === 0 && form.speculation === "none" && <p className="-mt-2 text-[12px] text-muted">No draft model here shares this model's tokenizer. A small model from the same family (for example its 0.6B or 1B size) can draft for it.</p>}
      {(form.speculation === "draft" || form.speculation === "mtp") && (
        <Field label="Tokens drafted at a time">
          <Input type="number" min={1} max={64} value={form.draft_max} onChange={(e) => set({ draft_max: Math.max(1, Math.min(64, Number(e.target.value) || 1)) })} />
        </Field>
      )}
      {stats && runtime?.settings?.speculation !== "none" && stats.drafted ? (
        <p className={cx("text-[12px]", (stats.accepted ?? 0) / stats.drafted < 0.4 ? "text-straw" : "text-muted")}>
          Last reply: {stats.speed.toFixed(1)} tokens/s, {Math.round(((stats.accepted ?? 0) / stats.drafted) * 100)}% of {stats.drafted} drafted tokens accepted.
          {(stats.accepted ?? 0) / stats.drafted < 0.4 && " So few are accepted that drafting likely slows replies; try fewer tokens at a time, prompt lookup, or Off."}
        </p>
      ) : stats ? (
        <p className="text-[12px] text-muted">Last reply: {stats.speed.toFixed(1)} tokens/s.</p>
      ) : null}
      {changed && (
        <div className="flex flex-col gap-2 rounded-lg border border-line bg-surface p-2.5">
          {estimate && (
            <p className={cx("text-[12px]", estimate.verdict === "too_large" ? "text-danger" : estimate.verdict === "tight" ? "text-straw" : "text-muted")}>
              Needs about {bytes(estimate.total)} of the {bytes(estimate.budget.total)} this computer can give it.
            </p>
          )}
          <div className="flex gap-2">
            <Button size="sm" variant="primary" onClick={() => void apply()} disabled={saving}>
              <RotateCcw className="size-3.5" /> {saving ? "Reloading…" : "Reload with these"}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setForm(offer.settings)} disabled={saving}>
              Undo
            </Button>
          </div>
        </div>
      )}
      {error && <p className="text-[12px] text-danger">{error}</p>}
      <p className="text-[12px] text-muted">Saved for this model.</p>
    </div>
  );
}
