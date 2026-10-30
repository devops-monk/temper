import { FolderOpen, Play, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { LossChart } from "../components/LossChart";
import { RunHistory } from "../components/RunHistory";
import { MemoryEstimate, type Budget, type Estimate, type Verdict } from "../components/MemoryEstimate";
import { Badge, Button, Field, Input, Notice, PageHeader, ProgressBar, Section, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { duration } from "../lib/format";
import { useApp } from "../lib/store";
import { useJob } from "../lib/useJob";
import type { Job } from "../lib/types";

interface Form {
  name: string;
  base_model: string;
  dataset: string;
  dataset_split: string;
  dataset_config: string;
  dataset_format: "auto" | "messages" | "sharegpt" | "alpaca" | "text";
  max_samples: number;
  method: "qlora" | "lora" | "full";
  train_embeddings: boolean;
  max_seq_length: number;
  lora_r: number;
  lora_alpha: number;
  learning_rate: number;
  epochs: number;
  max_steps: number;
  batch_size: number;
  grad_accum: number;
  warmup_steps: number;
}

const DEFAULTS: Form = {
  name: "",
  base_model: "Qwen/Qwen3-4B-Instruct-2507",
  dataset: "mlabonne/FineTome-100k",
  dataset_split: "train",
  dataset_config: "",
  dataset_format: "auto",
  max_samples: 2000,
  method: "qlora",
  train_embeddings: false,
  max_seq_length: 2048,
  lora_r: 16,
  lora_alpha: 16,
  learning_rate: 0.0002,
  epochs: 1,
  max_steps: 60,
  batch_size: 2,
  grad_accum: 4,
  warmup_steps: 5,
};

/** On Apple Silicon, MLX needs MLX-format weights; these community conversions mirror the defaults. */
const MLX_DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit";

const METHODS = {
  qlora: { label: "QLoRA", note: "4-bit base weights plus small adapters. Uses the least memory." },
  lora: { label: "LoRA", note: "16-bit base weights plus adapters. Slightly better quality, about twice the memory." },
  full: { label: "Full fine-tune", note: "Updates every weight. Needs a large GPU, or on a Mac a 16-bit model and plenty of memory." },
} as const;

function NumberField({ label, value, onChange, step, hint }: { label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <Input type="number" value={value} step={step} onChange={(e) => onChange(Number(e.target.value))} className="tabular-nums" />
    </Field>
  );
}

function RunPanel({ job }: { job: Job }) {
  const { events, status } = useJob(job.id);
  const refreshJobs = useApp((s) => s.refreshJobs);
  const logRef = useRef<HTMLPreElement>(null);

  const { points, last, phase, dataset, logs, totalSteps, val } = useMemo(() => {
    const points: { step: number; loss: number }[] = [];
    let val: { step: number; val_loss: number } | null = null;
    let last: Record<string, number> | null = null;
    let phase = "Starting worker";
    let totalSteps: number | null = null;
    let dataset: Record<string, unknown> | null = null;
    const logs: string[] = [];
    for (const ev of events) {
      if (ev.event === "metrics" && typeof ev.loss === "number") {
        points.push({ step: ev.step as number, loss: ev.loss });
        last = ev as unknown as Record<string, number>;
        totalSteps = (ev.total_steps as number) ?? totalSteps;
      } else if (ev.event === "phase") {
        phase = ev.name as string;
        totalSteps = (ev.total_steps as number) ?? totalSteps;
      } else if (ev.event === "eval" && typeof ev.val_loss === "number") val = { step: ev.step as number, val_loss: ev.val_loss };
      else if (ev.event === "dataset") dataset = ev;
      else if (ev.event === "log") logs.push(ev.line as string);
      else if (ev.event === "error") logs.push(`Error: ${ev.message}`);
    }
    return { points, last, phase, dataset, logs: logs.slice(-400), totalSteps, val };
  }, [events]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  const state = status ?? job.status;
  const progress = totalSteps && last ? last.step / totalSteps : state === "done" ? 1 : null;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate font-semibold">{job.title}</h2>
          <p className="text-[13px] text-muted">
            {state === "running" ? phase : state === "done" ? "Finished. Export it from the Export page." : state === "failed" ? "Training failed" : "Stopped"}
          </p>
        </div>
        {state === "running" ? (
          <Button size="sm" variant="danger" onClick={async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); void refreshJobs(); }}>
            <Square className="size-3 fill-current" /> Stop
          </Button>
        ) : (
          <Badge tone={state === "done" ? "ok" : "neutral"}>{state}</Badge>
        )}
      </div>

      <ProgressBar value={progress} temper />

      <dl className="grid grid-cols-4 gap-3">
        {[
          ["Step", last ? `${last.step} / ${totalSteps ?? "?"}` : "–"],
          ["Loss", last?.loss != null ? last.loss.toFixed(4) : "–"],
          last?.lr != null ? ["Learning rate", last.lr.toExponential(2)] : ["Tokens/s", last?.tokens_per_sec != null ? Math.round(last.tokens_per_sec).toString() : "–"],
          ["Time left", state === "running" ? duration(last?.eta) : "–"],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-surface px-3 py-2">
            <dt className="text-[12px] text-muted">{k}</dt>
            <dd className="mt-0.5 font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="rounded-lg border border-line bg-surface p-3">
        <LossChart points={points} totalSteps={totalSteps} />
        {val && (
          <p className="mt-1 text-[12.5px] text-muted">
            Validation loss {val.val_loss.toFixed(3)} at step {val.step}
          </p>
        )}
      </div>

      {state === "failed" && job.error && <Notice tone="error">{job.error}</Notice>}

      {dataset && (
        <details className="rounded-lg border border-line bg-surface px-3 py-2 text-[13px]">
          <summary className="cursor-pointer">
            {String(dataset.rows)} rows, {String(dataset.format)} format. Show the first sample as the model sees it.
          </summary>
          <pre className="mt-2 max-h-60 overflow-auto font-mono text-[12px] whitespace-pre-wrap text-muted">{String(dataset.sample)}</pre>
        </details>
      )}

      <pre ref={logRef} className="h-56 overflow-auto rounded-lg border border-line bg-surface-2 p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap text-muted">
        {logs.length ? logs.join("\n") : "Worker output appears here."}
      </pre>
    </div>
  );
}

interface TrainEstimate {
  weights: number;
  adapters_and_optimizer: number;
  activations: number;
  overhead: number;
  total: number;
  budget: Budget;
  verdict: Verdict;
  advice: string[];
  measured: boolean;
}

/** Re-estimates training memory as the form changes, after a short pause in typing. */
function useTrainEstimate(form: Form, enabled: boolean) {
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { base_model, method, lora_r, batch_size, max_seq_length } = form;

  useEffect(() => {
    if (!enabled || !base_model.trim()) {
      setEstimate(null);
      return;
    }
    let stale = false;
    const t = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const e = await api<TrainEstimate>("/api/memory/train", { json: { base_model, method, lora_r, batch_size, max_seq_length } });
        if (stale) return;
        setEstimate({
          total: e.total,
          budget: e.budget,
          verdict: e.verdict,
          advice: e.advice,
          measured: e.measured,
          parts: [
            ["Model weights", e.weights],
            ["Activations", e.activations],
            [method === "full" ? "Gradients and optimizer" : "Adapters and optimizer", e.adapters_and_optimizer],
            ["Runtime", e.overhead],
          ],
        });
      } catch (e) {
        if (stale) return;
        setEstimate(null);
        setError((e as Error).message);
      } finally {
        if (!stale) setLoading(false);
      }
    }, 600);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [enabled, base_model, method, lora_r, batch_size, max_seq_length]);

  return { estimate, error, loading };
}

export function TrainPage() {
  const { system, jobs, refreshJobs, pendingDataset, setPendingDataset, pendingBaseModel, setPendingBaseModel, go } = useApp();
  const trainer = system?.capabilities.trainer ?? null;
  const [form, setForm] = useState<Form>(() => ({ ...DEFAULTS, base_model: system?.backend === "metal" ? MLX_DEFAULT_MODEL : DEFAULTS.base_model }));
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const set = <K extends keyof Form>(k: K) => (v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));
  const memory = useTrainEstimate(form, !!trainer);

  // Hardware info can arrive after mount; swap to the MLX default unless the user already edited it.
  useEffect(() => {
    if (system?.backend === "metal") setForm((f) => (f.base_model === DEFAULTS.base_model ? { ...f, base_model: MLX_DEFAULT_MODEL } : f));
  }, [system?.backend]);

  useEffect(() => {
    if (pendingDataset) {
      const { source, split = "train", config = "" } = pendingDataset;
      setForm((f) => ({ ...f, dataset: source, dataset_split: split, dataset_config: config }));
      setPendingDataset(null);
    }
  }, [pendingDataset, setPendingDataset]);

  useEffect(() => {
    if (pendingBaseModel) {
      setForm((f) => ({ ...f, base_model: pendingBaseModel }));
      setPendingBaseModel(null);
    }
  }, [pendingBaseModel, setPendingBaseModel]);

  const trainJobs = jobs.filter((j) => j.kind === "train");
  const current = trainJobs.find((j) => j.id === selected) ?? trainJobs[0] ?? null;
  const running = trainJobs.some((j) => j.status === "running");
  // Past runs are listed again whenever a run ends.
  const finished = trainJobs.filter((j) => j.status !== "running").length;

  async function start() {
    if (memory.estimate?.verdict === "too_large" && !confirm("This run probably needs more memory than this computer has, and may fail or slow your computer to a crawl. Start anyway?")) return;
    setError(null);
    setStarting(true);
    try {
      const job = await api<Job>("/api/train", { json: { ...form, name: form.name || null } });
      setSelected(job.id);
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function browse() {
    if (!inTauri) return;
    const { open } = await import("@tauri-apps/plugin-dialog");
    const file = await open({ filters: [{ name: "Datasets", extensions: ["jsonl", "json", "csv", "parquet"] }] });
    if (typeof file === "string") set("dataset")(file);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Train"
        description="Fine-tune a model on your own data with LoRA adapters."
        actions={
          <Button variant="primary" onClick={start} disabled={!trainer || running || starting}>
            <Play className="size-4" /> {starting ? "Starting…" : "Start training"}
          </Button>
        }
      />
      <div className="flex min-h-0 flex-1">
        <div className="w-[400px] shrink-0 overflow-y-auto border-r border-line px-7 pb-8">
          {!trainer && system && (
            <div className="pt-5">
              <Notice>
                {system.backend === "cpu" ? (
                  <>No supported GPU was found. Training needs an NVIDIA, AMD or Intel GPU, or an Apple Silicon Mac.</>
                ) : (
                  <>
                    Training support isn't installed yet.{" "}
                    <button className="font-medium text-accent hover:underline" onClick={() => go("settings")}>
                      Add it in Settings
                    </button>
                    .
                  </>
                )}
              </Notice>
            </div>
          )}
          <Section title="Model" aside={trainer && <Badge tone="accent">{trainer === "mlx" ? "MLX" : "PyTorch"}</Badge>}>
            <div className="flex flex-col gap-3">
              <Field label="Base model" hint="A Hugging Face repo id or a local folder.">
                <Input value={form.base_model} onChange={(e) => set("base_model")(e.target.value)} spellCheck={false} />
              </Field>
              <Field label="Run name" hint="Optional. Shown in your run list.">
                <Input value={form.name} onChange={(e) => set("name")(e.target.value)} placeholder="support-bot-v1" />
              </Field>
            </div>
          </Section>

          <Section title="Data">
            <div className="flex flex-col gap-3">
              <Field label="Dataset" hint="A Hugging Face dataset id, or a local .jsonl, .json or .csv file.">
                <div className="flex gap-2">
                  <Input value={form.dataset} onChange={(e) => set("dataset")(e.target.value)} spellCheck={false} />
                  {inTauri && (
                    <Button onClick={browse} aria-label="Choose a file" className="px-2.5">
                      <FolderOpen className="size-4" />
                    </Button>
                  )}
                </div>
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Format">
                  <Select value={form.dataset_format} onChange={(e) => set("dataset_format")(e.target.value as Form["dataset_format"])}>
                    <option value="auto">Auto-detect</option>
                    <option value="messages">OpenAI messages</option>
                    <option value="sharegpt">ShareGPT</option>
                    <option value="alpaca">Alpaca</option>
                    <option value="text">Plain text</option>
                  </Select>
                </Field>
                <NumberField label="Max rows" value={form.max_samples} onChange={set("max_samples")} />
              </div>
              {!/^[~/]/.test(form.dataset) && !/^[A-Za-z]:\\/.test(form.dataset) && (
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Subset" hint="Only for datasets with several.">
                    <Input value={form.dataset_config} onChange={(e) => set("dataset_config")(e.target.value)} placeholder="default" spellCheck={false} />
                  </Field>
                  <Field label="Split">
                    <Input value={form.dataset_split} onChange={(e) => set("dataset_split")(e.target.value)} spellCheck={false} />
                  </Field>
                </div>
              )}
            </div>
          </Section>

          <Section title="Method">
            <div className="flex flex-col gap-2" role="radiogroup">
              {(Object.keys(METHODS) as (keyof typeof METHODS)[]).map((m) => (
                <button
                  key={m}
                  role="radio"
                  aria-checked={form.method === m}
                  onClick={() => set("method")(m)}
                  className={cx("rounded-lg border px-3 py-2 text-left transition-colors", form.method === m ? "border-accent bg-accent/6" : "border-line bg-surface hover:bg-surface-2")}
                >
                  <span className="font-medium">{METHODS[m].label}</span>
                  <span className="block text-[12.5px] text-muted">{METHODS[m].note}</span>
                </button>
              ))}
              <Toggle label="Also train the embeddings and output head" checked={form.train_embeddings} onChange={set("train_embeddings")} />
              <p className="-mt-1.5 text-[12px] text-muted">
                For continued pretraining on new words or a new language. Needs a 16-bit model, and more memory.
              </p>
            </div>
          </Section>

          <Section title="Hyperparameters">
            <div className="grid grid-cols-2 gap-3">
              <NumberField label="Max steps" value={form.max_steps} onChange={set("max_steps")} hint="0 = full epochs" />
              <NumberField label="Epochs" value={form.epochs} onChange={set("epochs")} step={0.5} />
              <NumberField label="Learning rate" value={form.learning_rate} onChange={set("learning_rate")} step={0.00005} />
              <NumberField label="Context length" value={form.max_seq_length} onChange={set("max_seq_length")} step={256} />
              <NumberField label="Batch size" value={form.batch_size} onChange={set("batch_size")} />
              <NumberField label="Grad accumulation" value={form.grad_accum} onChange={set("grad_accum")} />
              {form.method !== "full" && (
                <>
                  <NumberField label="LoRA rank" value={form.lora_r} onChange={set("lora_r")} />
                  <NumberField label="LoRA alpha" value={form.lora_alpha} onChange={set("lora_alpha")} />
                </>
              )}
              <NumberField label="Warmup steps" value={form.warmup_steps} onChange={set("warmup_steps")} />
            </div>
          </Section>
          {trainer && (
            <Section title="Memory">
              {memory.error ? <p className="text-[13px] text-muted">{memory.error}</p> : <MemoryEstimate estimate={memory.estimate} loading={memory.loading} />}
            </Section>
          )}
          {error && <Notice tone="error">{error}</Notice>}
        </div>

        <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
          {trainJobs.length > 1 && (
            <div className="mb-4 flex flex-wrap gap-1.5">
              {trainJobs.map((j) => (
                <button
                  key={j.id}
                  onClick={() => setSelected(j.id)}
                  className={cx("rounded-full border px-2.5 py-0.5 text-[12.5px]", j.id === current?.id ? "border-accent text-text" : "border-line text-muted hover:text-text")}
                >
                  {j.title}
                </button>
              ))}
            </div>
          )}
          {current ? (
            <RunPanel key={current.id} job={current} />
          ) : (
            <div className="flex min-h-[50%] flex-col items-center justify-center text-center">
              <div className="bg-temper mb-5 h-1.5 w-40 rounded-full opacity-80" />
              <p className="font-semibold">No training runs yet</p>
              <p className="mt-1 max-w-[46ch] text-muted">
                The defaults train a small instruct model for 60 steps, a few minutes on most GPUs. Change anything on the left, then start training.
              </p>
            </div>
          )}
          <div className="mt-6">
            <RunHistory tick={finished} />
          </div>
        </div>
      </div>
    </div>
  );
}
