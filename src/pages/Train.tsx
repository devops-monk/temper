import { FolderOpen, Play, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { LossChart } from "../components/LossChart";
import { RunHistory } from "../components/RunHistory";
import { SampleReplies, type Sample } from "../components/SampleReplies";
import { TokenPreview } from "../components/TokenPreview";
import { MemoryEstimate, type Budget, type Estimate, type Verdict } from "../components/MemoryEstimate";
import { Badge, Button, Field, Input, Notice, PageHeader, ProgressBar, Section, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { duration } from "../lib/format";
import { useApp } from "../lib/store";
import { useJob } from "../lib/useJob";
import type { Job } from "../lib/types";
import { Classifiers } from "./Classifiers";
import { Embedders } from "./Embedders";

interface Form {
  name: string;
  base_model: string;
  dataset: string;
  dataset_split: string;
  dataset_config: string;
  dataset_format: "auto" | "messages" | "sharegpt" | "alpaca" | "text";
  max_samples: number;
  method: "qlora" | "lora" | "full";
  objective: keyof typeof OBJECTIVES;
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
  early_stopping: boolean;
  sample_replies: boolean;
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
  objective: "sft",
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
  early_stopping: false,
  sample_replies: true,
};

/** On Apple Silicon, MLX needs MLX-format weights; these community conversions mirror the defaults. */
const MLX_DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit";

/** What a run learns (engine preference.py for all but the first). */
const OBJECTIVES = {
  sft: { label: "Supervised fine-tuning", note: "Learns to write the replies in the data." },
  dpo: { label: "DPO", note: "Learns which of two replies is better, from rows with a prompt, a chosen and a rejected reply. Stays close to the starting model." },
  orpo: { label: "ORPO", note: "Fine-tunes on the chosen replies while pushing away from the rejected ones, in one step. Same rows as DPO." },
  simpo: { label: "SimPO", note: "Compares the two replies' average likelihood, with a margin. Same rows as DPO; uses less memory." },
  kto: { label: "KTO", note: "Learns from single replies marked good or bad (rows with a prompt, a completion and a true or false label), so no pairs are needed. Pairs work too." },
} as const;
const SFT_LR = 0.0002;
const PREFERENCE_LR = 0.00005; // preference objectives move further per step

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

/** The evaluation whose adapter a run kept, when it stopped early (engine train_worker, "best"). */
interface Best {
  step: number;
  val_loss: number;
  stopped_at: number | null;
}

function RunPanel({ job }: { job: Job }) {
  const { events, status } = useJob(job.id);
  const refreshJobs = useApp((s) => s.refreshJobs);
  const logRef = useRef<HTMLPreElement>(null);

  const { points, last, phase, dataset, logs, totalSteps, val, samples, best } = useMemo(() => {
    const points: { step: number; loss: number }[] = [];
    let val: { step: number; val_loss: number; val_accuracy?: number } | null = null;
    let last: Record<string, number> | null = null;
    let phase = "Starting worker";
    let totalSteps: number | null = null;
    let dataset: Record<string, unknown> | null = null;
    let best: Best | null = null;
    const samples: Sample[] = [];
    const logs: string[] = [];
    for (const ev of events) {
      if (ev.event === "metrics" && typeof ev.loss === "number") {
        points.push({ step: ev.step as number, loss: ev.loss });
        last = ev as unknown as Record<string, number>;
        totalSteps = (ev.total_steps as number) ?? totalSteps;
      } else if (ev.event === "phase") {
        phase = ev.name as string;
        totalSteps = (ev.total_steps as number) ?? totalSteps;
      } else if (ev.event === "eval" && typeof ev.val_loss === "number") val = { step: ev.step as number, val_loss: ev.val_loss, val_accuracy: ev.val_accuracy as number | undefined };
      else if (ev.event === "dataset") dataset = ev;
      else if (ev.event === "sample") samples.push(ev as unknown as Sample);
      else if (ev.event === "best") best = ev as unknown as Best;
      else if (ev.event === "log") logs.push(ev.line as string);
      else if (ev.event === "error") logs.push(`Error: ${ev.message}`);
    }
    return { points, last, phase, dataset, logs: logs.slice(-400), totalSteps, val, samples, best };
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
          last?.reward_accuracy != null
            ? ["Prefers chosen", `${Math.round(last.reward_accuracy * 100)}%`]
            : last?.lr != null
              ? ["Learning rate", last.lr.toExponential(2)]
              : ["Tokens/s", last?.tokens_per_sec != null ? Math.round(last.tokens_per_sec).toString() : "–"],
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
            {val.val_accuracy != null && `; prefers the chosen reply in ${Math.round(val.val_accuracy * 100)}% of held-out pairs`}
          </p>
        )}
        {best && (
          <p className="mt-1 text-[12.5px]">
            {best.stopped_at ? `Stopped early at step ${best.stopped_at}: validation loss had stopped improving. ` : ""}
            Kept the adapter from step {best.step} (validation loss {best.val_loss.toFixed(3)}).
          </p>
        )}
      </div>

      {samples.length > 0 && <SampleReplies samples={samples} />}

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
  const { base_model, method, lora_r, max_seq_length } = form;
  // A preference pair is two sequences, chosen and rejected.
  const batch_size = form.batch_size * (form.objective === "sft" ? 1 : 2);

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

type FitKey = "max_seq_length" | "lora_r" | "lora_alpha" | "batch_size" | "grad_accum";

/** Context length, LoRA rank and batch size that fit this computer (engine memory.training_fit), applied as the model or
 * method changes until they're edited by hand. */
function useFit(base_model: string, method: Form["method"], objective: Form["objective"], enabled: boolean, apply: (fit: Pick<Form, FitKey>) => void) {
  const [autoFit, setAutoFit] = useState(true);
  const [fitted, setFitted] = useState<string | null>(null);
  const applyRef = useRef(apply);
  applyRef.current = apply;
  useEffect(() => {
    if (!enabled || !autoFit || !base_model.trim()) return;
    let stale = false;
    const t = setTimeout(async () => {
      const fit = await api<Pick<Form, FitKey> & { fits: boolean }>("/api/memory/train/fit", { json: { base_model, method, objective } }).catch(() => null);
      if (stale || !fit) return;
      const { max_seq_length, lora_r, lora_alpha, batch_size, grad_accum } = fit;
      applyRef.current({ max_seq_length, lora_r, lora_alpha, batch_size, grad_accum });
      setFitted(`${base_model} ${method} ${objective}`);
    }, 600);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [enabled, autoFit, base_model, method, objective]);
  return { autoFit, setAutoFit, fitted: fitted === `${base_model} ${method} ${objective}` };
}

type Mode = "lm" | "classifier" | "embedder";

/** What the Train page trains: a language model (LoRA and the rest), a classifier (Classifiers.tsx) or an embedding model (Embedders.tsx). */
function ModeSwitch({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <div className="flex rounded-md border border-line p-0.5 text-[12.5px]" role="tablist" aria-label="What to train">
      {(
        [
          ["lm", "Language model"],
          ["classifier", "Classifier"],
          ["embedder", "Embeddings"],
        ] as const
      ).map(([key, label]) => (
        <button
          key={key}
          role="tab"
          aria-selected={mode === key}
          onClick={() => onChange(key)}
          className={cx("rounded px-2.5 py-1 font-medium", mode === key ? "bg-surface text-text shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:text-text")}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

export function TrainPage() {
  const [mode, setMode] = useState<Mode>("lm");
  const { system, jobs, refreshJobs, pendingDataset, setPendingDataset, pendingBaseModel, setPendingBaseModel, go } = useApp();
  const trainer = system?.capabilities.trainer ?? null;
  const [form, setForm] = useState<Form>(() => ({ ...DEFAULTS, base_model: system?.backend === "metal" ? MLX_DEFAULT_MODEL : DEFAULTS.base_model }));
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [tokens, setTokens] = useState(false);
  const set = <K extends keyof Form>(k: K) => (v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));
  const memory = useTrainEstimate(form, !!trainer);
  const { autoFit, setAutoFit, fitted } = useFit(form.base_model, form.method, form.objective, !!trainer, (fit) => setForm((f) => ({ ...f, ...fit })));
  // Editing a fitted value by hand stops Temper fitting it again.
  const setByHand = <K extends FitKey>(k: K) => (v: Form[K]) => {
    setAutoFit(false);
    set(k)(v);
  };

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

  if (mode === "classifier")
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title="Train" description="Teach a small model to sort text into your labels, from examples. It answers in milliseconds." actions={<ModeSwitch mode={mode} onChange={setMode} />} />
        <Classifiers />
      </div>
    );
  if (mode === "embedder")
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title="Train" description="Tune an embedding model on pairs from your data, so search finds what you mean." actions={<ModeSwitch mode={mode} onChange={setMode} />} />
        <Embedders />
      </div>
    );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title="Train"
        description="Fine-tune a model on your own data with LoRA adapters."
        actions={
          <div className="flex items-center gap-3">
            <ModeSwitch mode={mode} onChange={setMode} />
            <Button variant="primary" onClick={start} disabled={!trainer || running || starting}>
              <Play className="size-4" /> {starting ? "Starting…" : "Start training"}
            </Button>
          </div>
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
              <Field label="Dataset" hint="A Hugging Face dataset id, a .jsonl, .json or .csv file, or a web address of one (https://, or a public s3:// or gs:// file).">
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
              {!/^[~/]/.test(form.dataset) && !/^[A-Za-z]:\\/.test(form.dataset) && !/^(https?|s3|gs):\/\//i.test(form.dataset) && (
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Subset" hint="Only for datasets with several.">
                    <Input value={form.dataset_config} onChange={(e) => set("dataset_config")(e.target.value)} placeholder="default" spellCheck={false} />
                  </Field>
                  <Field label="Split">
                    <Input value={form.dataset_split} onChange={(e) => set("dataset_split")(e.target.value)} spellCheck={false} />
                  </Field>
                </div>
              )}
              {trainer && form.dataset.trim() && (
                <button className="self-start text-[12.5px] text-muted hover:text-text" onClick={() => setTokens(true)}>
                  See a row as the model sees it, token by token
                </button>
              )}
            </div>
          </Section>

          <Section title="Method">
            <div className="flex flex-col gap-2" role="radiogroup">
              <Field label="Objective" hint={OBJECTIVES[form.objective].note}>
                <Select
                  value={form.objective}
                  onChange={(e) => {
                    const objective = e.target.value as Form["objective"];
                    // Moves the learning rate to the objective's usual one, unless it was changed.
                    setForm((f) => ({
                      ...f,
                      objective,
                      learning_rate: objective === "sft" ? (f.learning_rate === PREFERENCE_LR ? SFT_LR : f.learning_rate) : f.learning_rate === SFT_LR ? PREFERENCE_LR : f.learning_rate,
                    }));
                  }}
                >
                  {(Object.keys(OBJECTIVES) as Form["objective"][]).map((o) => (
                    <option key={o} value={o}>
                      {o === "sft" ? OBJECTIVES[o].label : `Preference: ${OBJECTIVES[o].label}`}
                    </option>
                  ))}
                </Select>
              </Field>
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
              {form.objective === "sft" && (
                <>
                  <Toggle label="Also train the embeddings and output head" checked={form.train_embeddings} onChange={set("train_embeddings")} />
                  <p className="-mt-1.5 text-[12px] text-muted">
                    For continued pretraining on new words or a new language. Needs a 16-bit model, and more memory.
                  </p>
                </>
              )}
              {form.objective !== "sft" && form.method === "full" && <Notice>Preference training trains adapters: choose QLoRA or LoRA.</Notice>}
            </div>
          </Section>

          <Section
            title="Hyperparameters"
            aside={
              trainer &&
              (autoFit ? (
                fitted && <span className="text-[12px] text-muted" title="Context length, LoRA rank and batch size are chosen to fit this computer's memory">Fitted to this computer</span>
              ) : (
                <button className="text-[12px] text-muted hover:text-text" onClick={() => setAutoFit(true)}>
                  Fit to this computer
                </button>
              ))
            }
          >
            <div className="grid grid-cols-2 gap-3">
              <NumberField label="Max steps" value={form.max_steps} onChange={set("max_steps")} hint="0 = full epochs" />
              <NumberField label="Epochs" value={form.epochs} onChange={set("epochs")} step={0.5} />
              <NumberField label="Learning rate" value={form.learning_rate} onChange={set("learning_rate")} step={0.00005} />
              <NumberField label="Context length" value={form.max_seq_length} onChange={setByHand("max_seq_length")} step={256} />
              <NumberField label="Batch size" value={form.batch_size} onChange={setByHand("batch_size")} />
              <NumberField label="Grad accumulation" value={form.grad_accum} onChange={setByHand("grad_accum")} />
              {form.method !== "full" && (
                <>
                  <NumberField label="LoRA rank" value={form.lora_r} onChange={setByHand("lora_r")} />
                  <NumberField label="LoRA alpha" value={form.lora_alpha} onChange={setByHand("lora_alpha")} />
                </>
              )}
              <NumberField label="Warmup steps" value={form.warmup_steps} onChange={set("warmup_steps")} />
            </div>
          </Section>

          {form.objective === "sft" && (
          <Section title="Evaluation">
            <div className="flex flex-col gap-2.5">
              <Toggle label="Show sample replies while training" checked={form.sample_replies} onChange={set("sample_replies")} />
              <p className="-mt-1.5 text-[12px] text-muted">At each evaluation the model answers two held-out prompts, so you can watch its replies change.</p>
              <Toggle label="Stop early when validation loss stops improving" checked={form.early_stopping} onChange={set("early_stopping")} />
              <p className="-mt-1.5 text-[12px] text-muted">Stops after three evaluations without improvement, and keeps the adapter from the best one.</p>
            </div>
          </Section>
          )}
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
          {tokens && (
            <div className="mb-5">
              <TokenPreview source={form} onClose={() => setTokens(false)} />
            </div>
          )}
          {current ? (
            <RunPanel key={current.id} job={current} />
          ) : (
            <div className="flex min-h-[50%] flex-col items-center justify-center text-center">
              <div className="bg-temper mb-5 h-1.5 w-40 rounded-full opacity-80" />
              <p className="font-semibold">Ready to train</p>
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
