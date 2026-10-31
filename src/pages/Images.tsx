import { Dices, Download, FolderOpen, ImageIcon, ImagePlus, Loader2, RotateCcw, Sparkles, Trash2, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Button, Field, Input, PageHeader, ProgressBar, Select, cx } from "../components/ui";
import { api, apiBlob, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { Setup } from "./Audio";

interface Model {
  id: string;
  name: string;
  size: number;
  license: string;
  size_px: number;
  steps: number;
  guidance: number;
  about: string;
  downloaded: boolean;
}

interface Status {
  installed: boolean;
  ready: boolean;
  model: string;
  models: Model[];
}

interface Picture {
  id: string;
  model: string;
  prompt: string;
  negative: string | null;
  steps: number;
  guidance: number;
  width: number | null;
  height: number | null;
  seed: number;
  image: string | null;
  strength: number | null;
  lora: string | null;
  lora_scale: number | null;
  created: number;
  path: string;
}

const SHAPES: [string, number, number][] = [
  ["Square", 1, 1],
  ["Landscape", 3, 2],
  ["Portrait", 2, 3],
  ["Wide", 16, 9],
];

/** A shape at a model's size: its native side for the long edge, in multiples of 64. */
function dimensions(side: number, [, w, h]: [string, number, number]) {
  const round = (n: number) => Math.round(n / 64) * 64;
  return w >= h ? { width: side, height: round((side * h) / w) } : { width: round((side * w) / h), height: side };
}

/** A gallery picture, fetched with the engine's key (an <img> can't send it). */
function Picture({ id, className, alt }: { id: string; className?: string; alt: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let url: string | null = null;
    void apiBlob(`/api/images/gallery/${id}.png`).then((b) => setSrc((url = URL.createObjectURL(b))), () => {});
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [id]);
  return src ? <img src={src} alt={alt} className={className} /> : <div className={cx("animate-pulse bg-surface-2", className)} />;
}

/** Making images on this computer with diffusers (engine images.py): text to image, changing a picture, LoRAs, a gallery. */
export function ImagesPage() {
  const [st, setSt] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => void api<Status>("/api/images").then(setSt, (e: Error) => setError(e.message)), []);
  useEffect(load, [load]);

  const chosen = st?.models.find((m) => m.id === st.model);
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title="Images" description="Make pictures from a prompt, or change one you have. The models run on this computer, on its GPU when it has one." />
      <div className="min-h-0 flex-1 overflow-y-auto px-7 py-6">
        {!st || !chosen ? (
          error && <p className="text-[12px] text-danger">{error}</p>
        ) : st.ready ? (
          <Studio st={st} onChange={setSt} />
        ) : (
          <div className="flex flex-col gap-4">
            {st.installed && <ModelPicker st={st} onChange={setSt} />}
            <Setup
              installed={st.installed}
              size={chosen.size}
              repo={chosen.id}
              install="/api/images/install"
              download={() => api<Job>("/api/images/download", { json: { model: chosen.id } })}
              icon={<ImageIcon className="size-8" />}
              title={st.installed ? `Download ${chosen.name}` : "Set up image generation"}
              about={
                st.installed
                  ? `${bytes(chosen.size)}, ${chosen.license}. ${chosen.about}`
                  : "Image models need PyTorch and diffusers, so they install into their own environment (about 1 GB). Then you download a model."
              }
              onDone={load}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function ModelPicker({ st, onChange }: { st: Status; onChange: (s: Status) => void }) {
  const chosen = st.models.find((m) => m.id === st.model);
  return (
    <Field label="Model" hint={chosen && `${chosen.license}. ${chosen.about}`}>
      <Select value={st.model} onChange={(e) => void api<Status>("/api/images/model", { method: "PUT", json: { model: e.target.value } }).then(onChange)} aria-label="Model" className="max-w-[360px]">
        {st.models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name} · {bytes(m.size)}
            {m.downloaded ? "" : " · not downloaded"}
          </option>
        ))}
      </Select>
    </Field>
  );
}

function Studio({ st, onChange }: { st: Status; onChange: (s: Status) => void }) {
  const { refreshJobs } = useApp();
  const model = st.models.find((m) => m.id === st.model)!;
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [shape, setShape] = useState(0);
  const [count, setCount] = useState(1);
  const [steps, setSteps] = useState<number | "">("");
  const [guidance, setGuidance] = useState<number | "">("");
  const [seed, setSeed] = useState<number | "">("");
  const [lora, setLora] = useState("");
  const [loraScale, setLoraScale] = useState(1);
  const [source, setSource] = useState<{ id?: string; path: string } | null>(null);
  const [strength, setStrength] = useState(0.6);
  const [more, setMore] = useState(false);
  const [pictures, setPictures] = useState<Picture[]>([]);
  const [open, setOpen] = useState<Picture | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;

  const loadGallery = useCallback(() => void api<{ images: Picture[] }>("/api/images/gallery").then((r) => setPictures(r.images), () => {}), []);
  useEffect(loadGallery, [loadGallery]);
  useEffect(() => {
    if (!status || status === "running") return;
    if (status === "failed") setError((events.find((e) => e.event === "status")?.error as string) ?? "It failed.");
    setJobId(null);
    loadGallery();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function generate() {
    setError(null);
    const size = source ? {} : dimensions(model.size_px, SHAPES[shape]);
    try {
      const job = await api<Job>("/api/images/generate", {
        json: {
          prompt,
          negative: negative || null,
          ...size,
          count,
          steps: steps || null,
          guidance: guidance === "" ? null : guidance,
          seed: seed === "" ? null : seed,
          lora: lora.trim() || null,
          lora_scale: loraScale,
          image: source ? (source.id ?? source.path) : null,
          strength,
        },
      });
      setJobId(job.id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function pickSource() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const chosen = await open({ filters: [{ name: "Pictures", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (typeof chosen === "string") setSource({ path: chosen });
  }

  async function pickLora() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const chosen = await open({ filters: [{ name: "LoRA", extensions: ["safetensors"] }] });
    if (typeof chosen === "string") setLora(chosen);
  }

  function reuse(p: Picture) {
    setPrompt(p.prompt);
    setNegative(p.negative ?? "");
    setSteps(p.steps);
    setGuidance(p.guidance);
    setSeed(p.seed);
    setLora(p.lora ?? "");
    if (p.lora_scale) setLoraScale(p.lora_scale);
    if (p.model !== st.model) void api<Status>("/api/images/model", { method: "PUT", json: { model: p.model } }).then(onChange);
    setMore(true);
    setOpen(null);
  }

  async function remove(p: Picture) {
    setPictures((await api<{ images: Picture[] }>(`/api/images/gallery/${p.id}`, { method: "DELETE" })).images);
    if (source?.id === p.id) setSource(null);
    setOpen(null);
  }

  return (
    <div className="grid max-w-[1200px] gap-8 lg:grid-cols-[360px_minmax(0,1fr)]">
      <form
        className="flex flex-col gap-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (prompt.trim() && !running) void generate();
        }}
      >
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) e.currentTarget.form?.requestSubmit();
          }}
          rows={4}
          placeholder={source ? "How to change the picture" : "Describe the picture"}
          aria-label="Prompt"
          className="w-full resize-y rounded-lg border border-line bg-surface px-3 py-2 text-[14px] outline-none focus:border-accent"
        />

        {source ? (
          <div className="flex items-center gap-3 rounded-lg border border-line bg-surface p-2 text-[13px]">
            {source.id ? <Picture id={source.id} alt="Picture to change" className="size-14 rounded object-cover" /> : <ImagePlus className="size-6 text-muted" />}
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{source.id ? "Changing a gallery picture" : source.path.split(/[\\/]/).pop()}</p>
              <label className="mt-1 flex items-center gap-2 text-[12px] text-muted">
                Change
                <input type="range" min={0.1} max={1} step={0.05} value={strength} onChange={(e) => setStrength(+e.target.value)} aria-label="How much to change it" className="flex-1" />
                <span className="w-9 tabular-nums text-text">{Math.round(strength * 100)}%</span>
              </label>
            </div>
            <button type="button" className="text-muted hover:text-text" onClick={() => setSource(null)} aria-label="Start from nothing instead">
              <X className="size-4" />
            </button>
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-1.5">
            {SHAPES.map((s, i) => (
              <Button key={s[0]} type="button" size="sm" variant={shape === i ? "primary" : "ghost"} onClick={() => setShape(i)} aria-pressed={shape === i}>
                {s[0]}
              </Button>
            ))}
            {inTauri && (
              <Button type="button" size="sm" variant="ghost" onClick={() => void pickSource()} title="Start from a picture of yours">
                <ImagePlus className="size-3.5" /> From a picture…
              </Button>
            )}
          </div>
        )}

        <div className="flex items-center gap-2">
          <Select value={count} onChange={(e) => setCount(+e.target.value)} aria-label="How many" className="w-28">
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>
                {n} {n === 1 ? "image" : "images"}
              </option>
            ))}
          </Select>
          <Button variant="primary" type="submit" disabled={!prompt.trim() || running} className="flex-1">
            {running ? <Loader2 className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />} {source ? "Change it" : "Generate"}
          </Button>
        </div>
        {running && <ProgressBar value={progress ?? null} />}
        {error && <p className="text-[12px] text-danger">{error}</p>}

        <button type="button" className="self-start text-[12px] text-muted hover:text-text" onClick={() => setMore(!more)} aria-expanded={more}>
          {more ? "Fewer settings" : "More settings"}
        </button>
        {more && (
          <div className="flex flex-col gap-3">
            <ModelPicker st={st} onChange={onChange} />
            <Field label="Leave out" hint="Things the picture shouldn't have.">
              <Input value={negative} onChange={(e) => setNegative(e.target.value)} placeholder="blurry, text, watermark" />
            </Field>
            <div className="grid grid-cols-3 gap-2">
              <Field label="Steps">
                <Input type="number" min={1} max={100} value={steps} placeholder={String(model.steps)} onChange={(e) => setSteps(e.target.value ? +e.target.value : "")} />
              </Field>
              <Field label="Guidance">
                <Input type="number" min={0} max={30} step={0.5} value={guidance} placeholder={String(model.guidance)} onChange={(e) => setGuidance(e.target.value ? +e.target.value : "")} />
              </Field>
              <Field label="Seed">
                <div className="flex items-center gap-1">
                  <Input type="number" value={seed} placeholder="any" onChange={(e) => setSeed(e.target.value ? +e.target.value : "")} />
                  <button type="button" className="text-muted hover:text-text" onClick={() => setSeed(Math.floor(Math.random() * 2 ** 31))} title="A random seed" aria-label="A random seed">
                    <Dices className="size-4" />
                  </button>
                </div>
              </Field>
            </div>
            <Field label="LoRA" hint="A style or subject you trained, as a .safetensors file for this model's family.">
              <div className="flex items-center gap-2">
                <Input value={lora} onChange={(e) => setLora(e.target.value)} placeholder="Path to a LoRA" />
                {inTauri && (
                  <Button type="button" size="sm" variant="ghost" onClick={() => void pickLora()}>
                    Choose…
                  </Button>
                )}
              </div>
              {lora && (
                <label className="mt-1 flex items-center gap-2 text-[12px] text-muted">
                  Strength
                  <input type="range" min={0} max={2} step={0.05} value={loraScale} onChange={(e) => setLoraScale(+e.target.value)} aria-label="LoRA strength" className="flex-1" />
                  <span className="w-9 tabular-nums text-text">{loraScale.toFixed(2)}</span>
                </label>
              )}
            </Field>
          </div>
        )}
      </form>

      <div className="min-w-0">
        {pictures.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line text-[13px] text-muted">
            <ImageIcon className="size-7" />
            Pictures you make appear here, saved as PNGs in Temper's folder.
          </div>
        ) : (
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
            {pictures.map((p) => (
              <li key={p.id}>
                <button onClick={() => setOpen(p)} className="block w-full overflow-hidden rounded-lg border border-line bg-surface hover:border-accent" aria-label={p.prompt}>
                  <Picture id={p.id} alt={p.prompt} className="aspect-square w-full object-cover" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={() => setOpen(null)} role="dialog" aria-label="Picture">
          <div className="flex max-h-full w-full max-w-[1000px] flex-col overflow-hidden rounded-xl border border-line bg-bg md:flex-row" onClick={(e) => e.stopPropagation()}>
            <Picture id={open.id} alt={open.prompt} className="max-h-[70vh] min-w-0 flex-1 bg-surface-2 object-contain" />
            <div className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto p-5 text-[13px] md:w-[300px]">
              <div className="flex items-start gap-2">
                <p className="flex-1">{open.prompt}</p>
                <button className="text-muted hover:text-text" onClick={() => setOpen(null)} aria-label="Close">
                  <X className="size-4" />
                </button>
              </div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12px]">
                <dt className="text-muted">Model</dt>
                <dd className="truncate">{st.models.find((m) => m.id === open.model)?.name ?? open.model}</dd>
                <dt className="text-muted">Seed</dt>
                <dd className="tabular-nums">{open.seed}</dd>
                <dt className="text-muted">Steps</dt>
                <dd>
                  {open.steps}, guidance {open.guidance}
                </dd>
                {open.negative && (
                  <>
                    <dt className="text-muted">Left out</dt>
                    <dd>{open.negative}</dd>
                  </>
                )}
                {open.image && (
                  <>
                    <dt className="text-muted">Changed</dt>
                    <dd>{Math.round((open.strength ?? 0) * 100)}% of a picture</dd>
                  </>
                )}
                {open.lora && (
                  <>
                    <dt className="text-muted">LoRA</dt>
                    <dd className="truncate" title={open.lora}>
                      {open.lora.split(/[\\/]/).pop()} × {open.lora_scale}
                    </dd>
                  </>
                )}
                <dt className="text-muted">Made</dt>
                <dd>{relativeDate(open.created)}</dd>
              </dl>
              <div className="mt-auto flex flex-wrap gap-1.5">
                <Button size="sm" onClick={() => reuse(open)}>
                  <RotateCcw className="size-3.5" /> Use these settings
                </Button>
                <Button
                  size="sm"
                  onClick={() => {
                    setSource({ id: open.id, path: open.path });
                    setOpen(null);
                  }}
                >
                  <Wand2 className="size-3.5" /> Change it
                </Button>
                {inTauri ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={async () => {
                      const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
                      await revealItemInDir(open.path);
                    }}
                  >
                    <FolderOpen className="size-3.5" /> Show in folder
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={async () => {
                      const a = document.createElement("a");
                      a.href = URL.createObjectURL(await apiBlob(`/api/images/gallery/${open.id}.png`));
                      a.download = `${open.id}.png`;
                      a.click();
                    }}
                  >
                    <Download className="size-3.5" /> Save
                  </Button>
                )}
                <Button size="sm" variant="ghost" onClick={() => void remove(open)} aria-label="Delete picture">
                  <Trash2 className="size-3.5" /> Delete
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
