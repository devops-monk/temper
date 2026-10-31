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
import { type Key, useT, withNodes } from "../lib/i18n";
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
  /** "" for the model's own, a name from /api/train/templates, or Jinja of your own. */
  chat_template: string;
  rewards: string[];
  reward_pattern: string;
  reward_code: string;
  generations: number;
  max_completion_length: number;
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
  chat_template: "",
  rewards: ["answer"],
  reward_pattern: "",
  reward_code: "",
  generations: 4,
  max_completion_length: 256,
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
const OBJECTIVES: Record<"sft" | "dpo" | "orpo" | "simpo" | "grpo" | "dr_grpo" | "dapo" | "kto", { label: string; note: Key }> = {
  sft: { label: "SFT", note: "train.o_sft_note" },
  dpo: { label: "DPO", note: "train.o_dpo_note" },
  orpo: { label: "ORPO", note: "train.o_orpo_note" },
  simpo: { label: "SimPO", note: "train.o_simpo_note" },
  grpo: { label: "GRPO", note: "train.o_grpo_note" },
  dr_grpo: { label: "Dr. GRPO", note: "train.o_dr_grpo_note" },
  dapo: { label: "DAPO", note: "train.o_dapo_note" },
  kto: { label: "KTO", note: "train.o_kto_note" },
};
const STATE: Record<string, Key> = { done: "job.done", failed: "job.failed", cancelled: "job.cancelled" };
const RL: Form["objective"][] = ["grpo", "dr_grpo", "dapo"];
const SFT_LR = 0.0002;
const PREFERENCE_LR = 0.00005; // preference objectives move further per step
const RL_LR = 0.00002; // reinforcement learning: smaller steps keep it from collapsing onto one kind of reply
const usualLr = (o: Form["objective"]) => (o === "sft" ? SFT_LR : RL.includes(o) ? RL_LR : PREFERENCE_LR);

/** Labels are names, except the full fine-tune's. */
const METHODS: Record<Form["method"], { label?: string; note: Key }> = {
  qlora: { label: "QLoRA", note: "train.m_qlora_note" },
  lora: { label: "LoRA", note: "train.m_lora_note" },
  full: { note: "train.m_full_note" },
};

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
  const t = useT();

  const { points, last, phase, dataset, logs, totalSteps, val, valReward, samples, best } = useMemo(() => {
    const points: { step: number; loss: number }[] = [];
    let val: { step: number; val_loss: number; val_accuracy?: number } | null = null;
    let valReward: { step: number; reward: number } | null = null;
    let last: Record<string, number> | null = null;
    let phase = t("train.starting_worker");
    let totalSteps: number | null = null;
    let dataset: Record<string, unknown> | null = null;
    let best: Best | null = null;
    const samples: Sample[] = [];
    const logs: string[] = [];
    for (const ev of events) {
      if (ev.event === "metrics" && typeof ev.loss === "number") {
        // Reinforcement learning's loss is near zero by design; its reward is what shows progress.
        points.push({ step: ev.step as number, loss: typeof ev.reward === "number" ? ev.reward : ev.loss });
        last = ev as unknown as Record<string, number>;
        totalSteps = (ev.total_steps as number) ?? totalSteps;
      } else if (ev.event === "phase") {
        phase = ev.name as string;
        totalSteps = (ev.total_steps as number) ?? totalSteps;
      } else if (ev.event === "eval" && typeof ev.val_loss === "number") val = { step: ev.step as number, val_loss: ev.val_loss, val_accuracy: ev.val_accuracy as number | undefined };
      else if (ev.event === "eval" && typeof ev.val_reward === "number") valReward = { step: ev.step as number, reward: ev.val_reward };
      else if (ev.event === "dataset") dataset = ev;
      else if (ev.event === "sample") samples.push(ev as unknown as Sample);
      else if (ev.event === "best") best = ev as unknown as Best;
      else if (ev.event === "log") logs.push(ev.line as string);
      else if (ev.event === "error") logs.push(t("train.error", { message: String(ev.message) }));
    }
    return { points, last, phase, dataset, logs: logs.slice(-400), totalSteps, val, valReward, samples, best };
  }, [events, t]);

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
            {state === "running" ? phase : t(state === "done" ? "train.finished" : state === "failed" ? "train.failed" : "train.stopped")}
          </p>
        </div>
        {state === "running" ? (
          <Button size="sm" variant="danger" onClick={async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); void refreshJobs(); }}>
            <Square className="size-3 fill-current" /> {t("train.stop")}
          </Button>
        ) : (
          <Badge tone={state === "done" ? "ok" : "neutral"}>{state in STATE ? t(STATE[state]) : state}</Badge>
        )}
      </div>

      <ProgressBar value={progress} temper />

      <dl className="grid grid-cols-4 gap-3">
        {[
          [t("train.step"), last ? `${last.step} / ${totalSteps ?? "?"}` : "–"],
          last?.reward != null ? [t("train.reward"), last.reward.toFixed(2)] : [t("train.loss"), last?.loss != null ? last.loss.toFixed(4) : "–"],
          last?.completion_length != null
            ? [t("train.reply_length"), t("train.tokens_n", { n: Math.round(last.completion_length) })]
            : last?.reward_accuracy != null
            ? [t("train.prefers_chosen"), `${Math.round(last.reward_accuracy * 100)}%`]
            : last?.lr != null
              ? [t("train.learning_rate"), last.lr.toExponential(2)]
              : [t("train.tokens_per_sec"), last?.tokens_per_sec != null ? Math.round(last.tokens_per_sec).toString() : "–"],
          [t("train.time_left"), state === "running" ? duration(last?.eta) : "–"],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-surface px-3 py-2">
            <dt className="text-[12px] text-muted">{k}</dt>
            <dd className="mt-0.5 font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="rounded-lg border border-line bg-surface p-3">
        <LossChart points={points} totalSteps={totalSteps} what={last?.reward != null ? "reward" : "loss"} />
        {valReward && (
          <p className="mt-1 text-[12.5px] text-muted">
            {t("train.val_reward", { reward: valReward.reward.toFixed(2), step: valReward.step })}
          </p>
        )}
        {val && (
          <p className="mt-1 text-[12.5px] text-muted">
            {t("train.val_loss", { loss: val.val_loss.toFixed(3), step: val.step })}
            {val.val_accuracy != null && t("train.val_accuracy", { percent: Math.round(val.val_accuracy * 100) })}
          </p>
        )}
        {best && (
          <p className="mt-1 text-[12.5px]">
            {best.stopped_at ? `${t("train.stopped_early", { step: best.stopped_at })} ` : ""}
            {t("train.kept_best", { step: best.step, loss: best.val_loss.toFixed(3) })}
          </p>
        )}
      </div>

      {samples.length > 0 && <SampleReplies samples={samples} />}

      {state === "failed" && job.error && <Notice tone="error">{job.error}</Notice>}

      {dataset && (
        <details className="rounded-lg border border-line bg-surface px-3 py-2 text-[13px]">
          <summary className="cursor-pointer">
            {t("train.dataset_summary", { rows: String(dataset.rows), format: String(dataset.format) })}
          </summary>
          <pre className="mt-2 max-h-60 overflow-auto font-mono text-[12px] whitespace-pre-wrap text-muted">{String(dataset.sample)}</pre>
        </details>
      )}

      <pre ref={logRef} className="h-56 overflow-auto rounded-lg border border-line bg-surface-2 p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap text-muted">
        {logs.length ? logs.join("\n") : t("train.worker_output")}
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
  const t = useT();
  const { base_model, method, lora_r, max_seq_length } = form;
  // A preference pair is two sequences, chosen and rejected.
  const batch_size = form.batch_size * (form.objective === "sft" ? 1 : 2);

  useEffect(() => {
    if (!enabled || !base_model.trim()) {
      setEstimate(null);
      return;
    }
    let stale = false;
    const timer = setTimeout(async () => {
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
            [t("train.mem_weights"), e.weights],
            [t("train.mem_activations"), e.activations],
            [t(method === "full" ? "train.mem_grads" : "train.mem_adapters"), e.adapters_and_optimizer],
            [t("train.mem_runtime"), e.overhead],
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
      clearTimeout(timer);
    };
  }, [enabled, base_model, method, lora_r, batch_size, max_seq_length, t]);

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
  const t = useT();
  return (
    <div className="flex rounded-md border border-line p-0.5 text-[12.5px]" role="tablist" aria-label={t("train.what")}>
      {(
        [
          ["lm", t("train.mode_lm")],
          ["classifier", t("train.mode_classifier")],
          ["embedder", t("train.mode_embedder")],
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
  const t = useT();
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
    if (memory.estimate?.verdict === "too_large" && !confirm(t("train.too_large_confirm"))) return;
    setError(null);
    setStarting(true);
    try {
      const job = await api<Job>("/api/train", { json: { ...form, name: form.name || null, chat_template: form.chat_template.trim() || null, reward_pattern: form.reward_pattern || null, reward_code: form.reward_code || null } });
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
    const file = await open({ filters: [{ name: t("train.datasets"), extensions: ["jsonl", "json", "csv", "parquet"] }] });
    if (typeof file === "string") set("dataset")(file);
  }

  if (mode === "classifier")
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title={t("nav.train")} description={t("train.classifier_description")} actions={<ModeSwitch mode={mode} onChange={setMode} />} />
        <Classifiers />
      </div>
    );
  if (mode === "embedder")
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title={t("nav.train")} description={t("train.embedder_description")} actions={<ModeSwitch mode={mode} onChange={setMode} />} />
        <Embedders />
      </div>
    );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title={t("nav.train")}
        description={t("train.description")}
        actions={
          <div className="flex items-center gap-3">
            <ModeSwitch mode={mode} onChange={setMode} />
            <Button variant="primary" onClick={start} disabled={!trainer || running || starting}>
              <Play className="size-4" /> {starting ? t("train.starting") : t("train.start")}
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
                  t("train.no_gpu")
                ) : (
                  withNodes(t("train.not_installed"), {
                    link: (
                      <button className="font-medium text-accent hover:underline" onClick={() => go("settings")}>
                        {t("train.add_in_settings")}
                      </button>
                    ),
                  })
                )}
              </Notice>
            </div>
          )}
          <Section title={t("train.model")} aside={trainer && <Badge tone="accent">{trainer === "mlx" ? "MLX" : "PyTorch"}</Badge>}>
            <div className="flex flex-col gap-3">
              <Field label={t("train.base_model")} hint={t("train.base_model_hint")}>
                <Input value={form.base_model} onChange={(e) => set("base_model")(e.target.value)} spellCheck={false} />
              </Field>
              <Field label={t("train.run_name")} hint={t("train.run_name_hint")}>
                <Input value={form.name} onChange={(e) => set("name")(e.target.value)} placeholder="support-bot-v1" />
              </Field>
            </div>
          </Section>

          <Section title={t("train.data")}>
            <div className="flex flex-col gap-3">
              <Field label={t("train.dataset")} hint={t("train.dataset_hint")}>
                <div className="flex gap-2">
                  <Input value={form.dataset} onChange={(e) => set("dataset")(e.target.value)} spellCheck={false} />
                  {inTauri && (
                    <Button onClick={browse} aria-label={t("train.choose_file")} className="px-2.5">
                      <FolderOpen className="size-4" />
                    </Button>
                  )}
                </div>
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label={t("train.format")}>
                  <Select value={form.dataset_format} onChange={(e) => set("dataset_format")(e.target.value as Form["dataset_format"])}>
                    <option value="auto">{t("train.auto")}</option>
                    <option value="messages">{t("datasets.f_messages")}</option>
                    <option value="sharegpt">ShareGPT</option>
                    <option value="alpaca">Alpaca</option>
                    <option value="text">{t("datasets.f_text")}</option>
                  </Select>
                </Field>
                <NumberField label={t("train.max_rows")} value={form.max_samples} onChange={set("max_samples")} />
              </div>
              {!/^[~/]/.test(form.dataset) && !/^[A-Za-z]:\\/.test(form.dataset) && !/^(https?|s3|gs):\/\//i.test(form.dataset) && (
                <div className="grid grid-cols-2 gap-3">
                  <Field label={t("datasets.subset")} hint={t("train.subset_hint")}>
                    <Input value={form.dataset_config} onChange={(e) => set("dataset_config")(e.target.value)} placeholder="default" spellCheck={false} />
                  </Field>
                  <Field label={t("datasets.split")}>
                    <Input value={form.dataset_split} onChange={(e) => set("dataset_split")(e.target.value)} spellCheck={false} />
                  </Field>
                </div>
              )}
              {trainer && form.dataset.trim() && (
                <button className="self-start text-[12.5px] text-muted hover:text-text" onClick={() => setTokens(true)}>
                  {t("train.see_tokens")}
                </button>
              )}
            </div>
          </Section>

          <Section title={t("train.method")}>
            <div className="flex flex-col gap-2" role="radiogroup">
              <Field label={t("train.objective")} hint={t(OBJECTIVES[form.objective].note)}>
                <Select
                  value={form.objective}
                  onChange={(e) => {
                    const objective = e.target.value as Form["objective"];
                    // Moves the learning rate to the objective's usual one, unless it was changed.
                    setForm((f) => ({ ...f, objective, learning_rate: f.learning_rate === usualLr(f.objective) ? usualLr(objective) : f.learning_rate }));
                  }}
                >
                  {(Object.keys(OBJECTIVES) as Form["objective"][]).map((o) => (
                    <option key={o} value={o}>
                      {o === "sft" ? t("train.o_sft") : t(RL.includes(o) ? "train.rl_obj" : "train.pref_obj", { name: OBJECTIVES[o].label })}
                    </option>
                  ))}
                </Select>
              </Field>
              {(Object.keys(METHODS) as Form["method"][]).map((m) => (
                <button
                  key={m}
                  role="radio"
                  aria-checked={form.method === m}
                  onClick={() => set("method")(m)}
                  className={cx("rounded-lg border px-3 py-2 text-left transition-colors", form.method === m ? "border-accent bg-accent/6" : "border-line bg-surface hover:bg-surface-2")}
                >
                  <span className="font-medium">{METHODS[m].label ?? t("train.m_full")}</span>
                  <span className="block text-[12.5px] text-muted">{t(METHODS[m].note)}</span>
                </button>
              ))}
              {form.objective === "sft" && (
                <>
                  <Toggle label={t("train.train_embeddings")} checked={form.train_embeddings} onChange={set("train_embeddings")} />
                  <p className="-mt-1.5 text-[12px] text-muted">
                    {t("train.train_embeddings_hint")}
                  </p>
                </>
              )}
              {form.objective !== "sft" && form.method === "full" && <Notice>{t(RL.includes(form.objective) ? "train.rl_adapters" : "train.pref_adapters")}</Notice>}
              {RL.includes(form.objective) && <Rewards form={form} set={set} />}
              <ChatTemplate value={form.chat_template} onChange={set("chat_template")} />
            </div>
          </Section>

          <Section
            title={t("train.hyperparameters")}
            aside={
              trainer &&
              (autoFit ? (
                fitted && <span className="text-[12px] text-muted" title={t("train.fitted_title")}>{t("train.fitted")}</span>
              ) : (
                <button className="text-[12px] text-muted hover:text-text" onClick={() => setAutoFit(true)}>
                  {t("train.fit")}
                </button>
              ))
            }
          >
            <div className="grid grid-cols-2 gap-3">
              <NumberField label={t("train.max_steps")} value={form.max_steps} onChange={set("max_steps")} hint={t("train.max_steps_hint")} />
              <NumberField label={t("train.epochs")} value={form.epochs} onChange={set("epochs")} step={0.5} />
              <NumberField label={t("train.learning_rate")} value={form.learning_rate} onChange={set("learning_rate")} step={0.00005} />
              <NumberField label={t("train.context_length")} value={form.max_seq_length} onChange={setByHand("max_seq_length")} step={256} />
              <NumberField label={t("train.batch_size")} value={form.batch_size} onChange={setByHand("batch_size")} />
              <NumberField label={t("train.grad_accum")} value={form.grad_accum} onChange={setByHand("grad_accum")} />
              {form.method !== "full" && (
                <>
                  <NumberField label={t("train.lora_rank")} value={form.lora_r} onChange={setByHand("lora_r")} />
                  <NumberField label={t("train.lora_alpha")} value={form.lora_alpha} onChange={setByHand("lora_alpha")} />
                </>
              )}
              <NumberField label={t("train.warmup")} value={form.warmup_steps} onChange={set("warmup_steps")} />
            </div>
          </Section>

          {form.objective === "sft" && (
          <Section title={t("train.evaluation")}>
            <div className="flex flex-col gap-2.5">
              <Toggle label={t("train.sample_replies")} checked={form.sample_replies} onChange={set("sample_replies")} />
              <p className="-mt-1.5 text-[12px] text-muted">{t("train.sample_replies_hint")}</p>
              <Toggle label={t("train.early_stopping")} checked={form.early_stopping} onChange={set("early_stopping")} />
              <p className="-mt-1.5 text-[12px] text-muted">{t("train.early_stopping_hint")}</p>
            </div>
          </Section>
          )}
          {trainer && (
            <Section title={t("train.memory")}>
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
              <p className="font-semibold">{t("train.ready")}</p>
              <p className="mt-1 max-w-[46ch] text-muted">
                {t("train.ready_hint")}
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

interface Template {
  id: string;
  label: string;
  about: string;
}

const OWN_TEMPLATE = "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";

/** The chat template a run renders conversations with: the model's own, a named one, or Jinja of your own (engine chat_render.py). */
function ChatTemplate({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const t = useT();
  const [templates, setTemplates] = useState<Template[]>([]);
  useEffect(() => void api<Template[]>("/api/train/templates").then(setTemplates, () => {}), []);
  const named = templates.find((x) => x.id === value);
  const custom = value !== "" && !named && templates.length > 0;
  const hint = named
    ? t("train.template_used", { about: named.about })
    : t(custom ? "train.template_custom" : "train.template_own");
  return (
    <Field label={t("train.chat_template")} hint={hint}>
      <Select value={custom ? "custom" : value} onChange={(e) => onChange(e.target.value === "custom" ? OWN_TEMPLATE : e.target.value)} aria-label={t("train.chat_template")}>
        <option value="">{t("train.model_own")}</option>
        {templates.map((x) => (
          <option key={x.id} value={x.id}>
            {x.label}
          </option>
        ))}
        <option value="custom">{t("train.your_own")}</option>
      </Select>
      {custom && (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          rows={5}
          aria-label={t("train.your_template")}
          className="mt-2 w-full rounded-md border border-line bg-surface px-3 py-2 font-mono text-[12px] text-text [font-variant-ligatures:none] focus:border-accent focus:outline-none"
        />
      )}
    </Field>
  );
}

const REWARDS: { id: string; label: Key; note: Key }[] = [
  { id: "answer", label: "train.r_answer", note: "train.r_answer_note" },
  { id: "format", label: "train.r_format", note: "train.r_format_note" },
  { id: "python", label: "train.r_python", note: "train.r_python_note" },
];

const REWARD_STARTER = `def reward(prompt, completion, row):
    """Return a number: higher for better replies. row is the dataset row."""
    score = 0.0
    if len(completion) < 400:
        score += 0.5
    return score
`;

/** What reinforcement learning rewards, and how many replies it compares (engine rewards.py). */
function Rewards({ form, set }: { form: Form; set: <K extends keyof Form>(k: K) => (v: Form[K]) => void }) {
  const t = useT();
  const toggle = (id: string, on: boolean) => {
    set("rewards")(on ? [...form.rewards, id] : form.rewards.filter((r) => r !== id));
    if (on && id === "python" && !form.reward_code) set("reward_code")(REWARD_STARTER);
  };
  return (
    <div className="mt-1 flex flex-col gap-2 rounded-lg border border-line bg-surface px-3 py-2.5">
      <span className="text-[12.5px] font-medium">{t("train.rewards")}</span>
      {REWARDS.map((r) => (
        <div key={r.id}>
          <label className="flex items-start gap-2 text-[13px]">
            <input type="checkbox" className="mt-1 accent-[var(--accent)]" checked={form.rewards.includes(r.id)} onChange={(e) => toggle(r.id, e.target.checked)} />
            <span>
              {t(r.label)}
              <span className="block text-[12px] text-muted">{t(r.note)}</span>
            </span>
          </label>
          {r.id === "format" && form.rewards.includes("format") && (
            <Input className="mt-1.5 font-mono text-[12px]" value={form.reward_pattern} onChange={(e) => set("reward_pattern")(e.target.value)} placeholder="Answer: *-?\d+" spellCheck={false} aria-label={t("train.format_pattern")} />
          )}
          {r.id === "python" && form.rewards.includes("python") && (
            <textarea
              value={form.reward_code}
              onChange={(e) => set("reward_code")(e.target.value)}
              spellCheck={false}
              rows={7}
              aria-label={t("train.reward_function")}
              className="mt-1.5 w-full rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-[12px] text-text [font-variant-ligatures:none] focus:border-accent focus:outline-none"
            />
          )}
        </div>
      ))}
      <div className="grid grid-cols-2 gap-3">
        <NumberField label={t("train.generations")} value={form.generations} onChange={set("generations")} hint={t("train.generations_hint")} />
        <NumberField label={t("train.reply_limit")} value={form.max_completion_length} onChange={set("max_completion_length")} step={64} hint={t("train.reply_limit_hint")} />
      </div>
    </div>
  );
}
