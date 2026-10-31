import { ArrowDown, ArrowLeft, ArrowUp, ChefHat, FilePlus, FlaskConical, FolderPlus, Play, Plus, Save, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Badge, Button, EmptyState, Field, Input, Notice, PageHeader, ProgressBar, Section, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import type { GeneratedDataset, Job, Recipe, RecipeColumn, RecipeSeed } from "../lib/types";
import { useJob } from "../lib/useJob";

type ColumnType = RecipeColumn["column_type"];

const COLUMN_TYPES: { type: ColumnType; label: Key; note: Key }[] = [
  { type: "llm-text", label: "recipes.t_text", note: "recipes.t_text_note" },
  { type: "llm-structured", label: "recipes.t_structured", note: "recipes.t_structured_note" },
  { type: "llm-code", label: "recipes.t_code", note: "recipes.t_code_note" },
  { type: "llm-judge", label: "recipes.t_judge", note: "recipes.t_judge_note" },
  { type: "sampler", label: "recipes.t_sampler", note: "recipes.t_sampler_note" },
  { type: "expression", label: "recipes.t_expression", note: "recipes.t_expression_note" },
  { type: "validation", label: "recipes.t_validation", note: "recipes.t_validation_note" },
];
const TYPE_LABEL = Object.fromEntries(COLUMN_TYPES.map((x) => [x.type, x.label])) as Record<ColumnType, Key>;
const STATE: Record<string, Key> = { done: "job.done", failed: "job.failed", cancelled: "job.cancelled" };

const CODE_LANGS = ["python", "javascript", "typescript", "go", "rust", "java", "cpp", "csharp", "bash", "sql:sqlite", "sql:postgres", "sql:mysql", "sql:ansi"];
const DOC_EXTENSIONS = ["pdf", "docx", "txt", "md", "html", "htm", "csv", "json", "jsonl"];
const SEED_COLUMNS: Record<RecipeSeed["type"], string[]> = { documents: ["chunk_text", "source_file"], file: [], hf: [] };

function newColumn(type: ColumnType, name: string, alias: string): RecipeColumn {
  switch (type) {
    case "sampler":
      return { column_type: type, name, sampler_type: "category", params: { values: ["first option", "second option"] } };
    case "llm-structured":
      return {
        column_type: type,
        name,
        model_alias: alias,
        prompt: "",
        output_format: { type: "object", additionalProperties: false, required: ["question", "answer"], properties: { question: { type: "string" }, answer: { type: "string" } } },
      };
    case "llm-code":
      return { column_type: type, name, model_alias: alias, prompt: "", code_lang: "python" };
    case "llm-judge":
      return {
        column_type: type,
        name,
        model_alias: alias,
        prompt: "",
        scores: [{ name: "quality", description: "How good is the answer?", options: { "1": "Poor", "2": "Fair", "3": "Good" } }],
      };
    case "expression":
      return { column_type: type, name, expr: "" };
    case "validation":
      return { column_type: type, name, target_columns: [], validator_type: "code", validator_params: { code_lang: "python" } };
    default:
      return { column_type: "llm-text", name, model_alias: alias, prompt: "" };
  }
}

function uniqueName(base: string, taken: string[]): string {
  let n = 1;
  while (taken.includes(`${base}_${n}`)) n++;
  return `${base}_${n}`;
}

// ------------------------------------------------------------------ small editors

function TextArea({ value, onChange, rows = 3, placeholder, mono }: { value: string; onChange: (v: string) => void; rows?: number; placeholder?: string; mono?: boolean }) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      rows={rows}
      placeholder={placeholder}
      spellCheck={!mono}
      className={cx(
        "w-full resize-y rounded-md border border-line bg-surface px-3 py-2 text-text placeholder:text-muted focus:border-accent focus:outline-none",
        mono && "font-mono text-[12px]",
      )}
    />
  );
}

/** Edits a value as JSON; keeps the text while it's invalid and only commits parsed values. */
function JsonField({ value, onChange, rows = 5 }: { value: unknown; onChange: (v: unknown) => void; rows?: number }) {
  const t = useT();
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);
  return (
    <div>
      <TextArea
        value={text}
        rows={rows}
        mono
        onChange={(v) => {
          setText(v);
          try {
            onChange(JSON.parse(v));
            setError(null);
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      />
      {error && <span className="mt-1 block text-xs text-danger">{t("recipes.invalid_json", { error })}</span>}
    </div>
  );
}

function lines(text: string): string[] {
  return text.split(/\n|,/).map((s) => s.trim()).filter(Boolean);
}

// ------------------------------------------------------------------ columns

function ColumnCard({
  column,
  index,
  count,
  aliases,
  codeColumns,
  onChange,
  onMove,
  onRemove,
}: {
  column: RecipeColumn;
  index: number;
  count: number;
  aliases: string[];
  codeColumns: string[];
  onChange: (c: RecipeColumn) => void;
  onMove: (delta: number) => void;
  onRemove: () => void;
}) {
  const t = useT();
  const set = (patch: Record<string, unknown>) => onChange({ ...column, ...patch });
  const type = column.column_type;
  const llm = type.startsWith("llm");
  const schema = column.output_format as { properties?: Record<string, { type?: string }> } | undefined;
  const simpleFields = schema?.properties && Object.values(schema.properties).every((p) => p.type === "string");
  const params = (column.params ?? {}) as { values?: unknown[] };
  const validatorParams = (column.validator_params ?? {}) as { code_lang?: string };

  return (
    <div className="rounded-lg border border-line bg-surface p-3.5">
      <div className="mb-3 flex items-center gap-2">
        <Input value={column.name} onChange={(e) => set({ name: e.target.value.replace(/\s+/g, "_") })} className="h-8 max-w-[200px] font-mono text-[12.5px]" aria-label={t("recipes.column_name")} />
        <Badge tone={llm ? "accent" : "neutral"}>{TYPE_LABEL[type] ? t(TYPE_LABEL[type]) : type}</Badge>
        <span className="ml-auto flex items-center">
          <Button size="sm" variant="ghost" onClick={() => onMove(-1)} disabled={index === 0} aria-label={t("recipes.move_up")}>
            <ArrowUp className="size-3.5" />
          </Button>
          <Button size="sm" variant="ghost" onClick={() => onMove(1)} disabled={index === count - 1} aria-label={t("recipes.move_down")}>
            <ArrowDown className="size-3.5" />
          </Button>
          <Button size="sm" variant="ghost" onClick={onRemove} aria-label={t("recipes.remove_named", { name: column.name })}>
            <Trash2 className="size-3.5" />
          </Button>
        </span>
      </div>
      <div className="flex flex-col gap-3">
        {llm && (
          <>
            <Field label={t("recipes.prompt")} hint={t("recipes.prompt_hint")}>
              <TextArea value={String(column.prompt ?? "")} onChange={(v) => set({ prompt: v })} rows={4} />
            </Field>
            {aliases.length > 1 && (
              <Field label={t("recipes.model")}>
                <Select value={String(column.model_alias)} onChange={(e) => set({ model_alias: e.target.value })}>
                  {aliases.map((a) => (
                    <option key={a}>{a}</option>
                  ))}
                </Select>
              </Field>
            )}
          </>
        )}
        {type === "llm-code" && (
          <Field label={t("recipes.language")}>
            <Select value={String(column.code_lang)} onChange={(e) => set({ code_lang: e.target.value })}>
              {CODE_LANGS.map((l) => (
                <option key={l}>{l}</option>
              ))}
            </Select>
          </Field>
        )}
        {type === "llm-structured" &&
          (simpleFields ? (
            <Field label={t("recipes.fields")} hint={t("recipes.fields_hint", { name: column.name })}>
              <Input
                defaultValue={Object.keys(schema!.properties!).join(", ")}
                onBlur={(e) => {
                  const names = lines(e.target.value);
                  set({
                    output_format: { type: "object", additionalProperties: false, required: names, properties: Object.fromEntries(names.map((n) => [n, { type: "string" }])) },
                  });
                }}
                className="font-mono text-[12.5px]"
              />
            </Field>
          ) : (
            <Field label={t("recipes.schema")}>
              <JsonField value={column.output_format} onChange={(v) => set({ output_format: v })} />
            </Field>
          ))}
        {type === "llm-judge" && (
          <Field label={t("recipes.scores")} hint={t("recipes.scores_hint")}>
            <JsonField value={column.scores} onChange={(v) => set({ scores: v })} rows={7} />
          </Field>
        )}
        {type === "sampler" &&
          (column.sampler_type === "category" ? (
            <Field label={t("recipes.values")} hint={t("recipes.values_hint")}>
              <TextArea value={(params.values ?? []).map(String).join("\n")} onChange={(v) => set({ params: { ...params, values: lines(v), weights: undefined } })} rows={4} />
            </Field>
          ) : (
            <Field label={t("recipes.parameters", { type: String(column.sampler_type) })}>
              <JsonField value={column.params} onChange={(v) => set({ params: v })} />
            </Field>
          ))}
        {type === "expression" && (
          <Field label={t("recipes.template")} hint={t("recipes.template_hint")}>
            <TextArea value={String(column.expr ?? "")} onChange={(v) => set({ expr: v })} rows={3} mono />
          </Field>
        )}
        {type === "validation" && (
          <div className="grid grid-cols-2 gap-3">
            <Field label={t("recipes.column_to_check")}>
              <Select value={String((column.target_columns as string[])[0] ?? "")} onChange={(e) => set({ target_columns: [e.target.value] })}>
                <option value="">{t("recipes.choose")}</option>
                {codeColumns.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </Select>
            </Field>
            <Field label={t("recipes.language")}>
              <Select value={validatorParams.code_lang ?? "python"} onChange={(e) => set({ validator_params: { ...validatorParams, code_lang: e.target.value } })}>
                {CODE_LANGS.map((l) => (
                  <option key={l}>{l}</option>
                ))}
              </Select>
            </Field>
          </div>
        )}
        <Toggle checked={!column.drop} onChange={(keep) => set({ drop: !keep })} label={t("recipes.keep")} />
      </div>
    </div>
  );
}

function AddColumn({ onAdd }: { onAdd: (type: ColumnType) => void }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  if (!open)
    return (
      <Button onClick={() => setOpen(true)} className="w-full border-dashed">
        <Plus className="size-4" /> {t("recipes.add_column")}
      </Button>
    );
  return (
    <div className="rounded-lg border border-line bg-surface-2 p-2">
      <div className="mb-1 flex items-center justify-between px-1.5">
        <span className="text-[13px] font-medium">{t("recipes.add_a_column")}</span>
        <Button size="sm" variant="ghost" onClick={() => setOpen(false)} aria-label={t("recipes.close")}>
          <X className="size-3.5" />
        </Button>
      </div>
      {COLUMN_TYPES.map((x) => (
        <button
          key={x.type}
          onClick={() => {
            onAdd(x.type);
            setOpen(false);
          }}
          className="block w-full rounded-md px-2.5 py-1.5 text-left hover:bg-surface"
        >
          <span className="font-medium">{t(x.label)}</span>
          <span className="block text-[12.5px] text-muted">{t(x.note)}</span>
        </button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ source and model

function SeedEditor({ seed, onChange }: { seed: RecipeSeed | null; onChange: (s: RecipeSeed | null) => void }) {
  const t = useT();
  const kinds: [RecipeSeed["type"] | "none", string][] = [
    ["none", t("recipes.s_none")],
    ["documents", t("recipes.s_documents")],
    ["file", t("recipes.s_file")],
    ["hf", "Hugging Face"],
  ];
  const current = seed?.type ?? "none";
  const sources = seed?.sources ?? [];

  async function addPaths(directory: boolean) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open(directory ? { directory: true, multiple: true } : { multiple: true, filters: [{ name: t("recipes.s_documents"), extensions: DOC_EXTENSIONS }] });
    const paths = Array.isArray(picked) ? picked : picked ? [picked] : [];
    if (paths.length && seed) onChange({ ...seed, sources: [...sources, ...paths] });
  }

  async function browseFile() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const file = await open({ filters: [{ name: t("recipes.datasets"), extensions: ["jsonl", "json", "csv", "parquet"] }] });
    if (typeof file === "string" && seed) onChange({ ...seed, path: file });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex rounded-md border border-line bg-surface p-0.5" role="radiogroup" aria-label={t("recipes.source")}>
        {kinds.map(([kind, label]) => (
          <button
            key={kind}
            role="radio"
            aria-checked={current === kind}
            onClick={() =>
              onChange(
                kind === "none" ? null : kind === "documents" ? { type: kind, sources: [], chunk_size: 1200, chunk_overlap: 200, sampling: "ordered" } : { type: kind, path: "", sampling: "ordered" },
              )
            }
            className={cx("flex-1 rounded px-2 py-1 text-[13px] font-medium whitespace-nowrap", current === kind ? "bg-surface-2 text-text" : "text-muted hover:text-text")}
          >
            {label}
          </button>
        ))}
      </div>
      {!seed && <p className="text-[13px] text-muted">{t("recipes.no_seed")}</p>}
      {seed?.type === "documents" && (
        <>
          <Field label={t("recipes.sources")} hint={t("recipes.sources_hint")}>
            <TextArea value={sources.join("\n")} onChange={(v) => onChange({ ...seed, sources: v.split("\n") })} rows={4} placeholder={"~/Documents/handbook.pdf\nhttps://github.com/owner/repo"} mono />
          </Field>
          {inTauri && (
            <div className="flex gap-2">
              <Button size="sm" onClick={() => void addPaths(false)}>
                <FilePlus className="size-3.5" /> {t("recipes.add_files")}
              </Button>
              <Button size="sm" onClick={() => void addPaths(true)}>
                <FolderPlus className="size-3.5" /> {t("recipes.add_folder")}
              </Button>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <Field label={t("recipes.chunk_size")} hint={t("recipes.chunk_size_hint")}>
              <Input type="number" min={200} step={100} value={seed.chunk_size ?? 1200} onChange={(e) => onChange({ ...seed, chunk_size: Number(e.target.value) })} />
            </Field>
            <Field label={t("recipes.overlap")} hint={t("recipes.overlap_hint")}>
              <Input type="number" min={0} step={50} value={seed.chunk_overlap ?? 200} onChange={(e) => onChange({ ...seed, chunk_overlap: Number(e.target.value) })} />
            </Field>
          </div>
        </>
      )}
      {(seed?.type === "file" || seed?.type === "hf") && (
        <Field
          label={t(seed.type === "file" ? "recipes.dataset_file" : "recipes.dataset")}
          hint={t(seed.type === "file" ? "recipes.file_hint" : "recipes.hf_hint")}
        >
          <div className="flex gap-2">
            <Input value={seed.path ?? ""} onChange={(e) => onChange({ ...seed, path: e.target.value })} spellCheck={false} className="font-mono text-[12.5px]" />
            {seed.type === "file" && inTauri && (
              <Button onClick={() => void browseFile()} aria-label={t("recipes.choose_file")} className="px-2.5">
                <FilePlus className="size-4" />
              </Button>
            )}
          </div>
        </Field>
      )}
      {seed && (
        <Field label={t("recipes.order")}>
          <Select value={seed.sampling ?? "ordered"} onChange={(e) => onChange({ ...seed, sampling: e.target.value as RecipeSeed["sampling"] })}>
            <option value="ordered">{t("recipes.in_order")}</option>
            <option value="shuffle">{t("recipes.shuffled")}</option>
          </Select>
        </Field>
      )}
    </div>
  );
}

function ModelEditor({ recipe, onChange }: { recipe: Recipe; onChange: (r: Recipe) => void }) {
  const t = useT();
  const runtime = useApp((s) => s.runtime);
  const setModel = (i: number, patch: Partial<Recipe["models"][number]>) => onChange({ ...recipe, models: recipe.models.map((m, j) => (j === i ? { ...m, ...patch } : m)) });
  const setProvider = (i: number, patch: Partial<Recipe["providers"][number]>) => onChange({ ...recipe, providers: recipe.providers.map((p, j) => (j === i ? { ...p, ...patch } : p)) });

  return (
    <div className="flex flex-col gap-4">
      {recipe.models.map((m, i) => (
        <div key={i} className="flex flex-col gap-3">
          {recipe.models.length > 1 && <span className="font-mono text-[12.5px] text-muted">{m.alias}</span>}
          <Field
            label={t("recipes.provider")}
            hint={m.provider === "local" ? (runtime?.state === "ready" ? t("recipes.uses_loaded", { model: runtime.model ?? "" }) : t("recipes.uses_chat_model")) : undefined}
          >
            <Select value={m.provider} onChange={(e) => setModel(i, { provider: e.target.value })}>
              <option value="local">{t("recipes.local")}</option>
              {recipe.providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </Select>
          </Field>
          {m.provider !== "local" && (
            <Field label={t("recipes.model_name")} hint={t("recipes.model_name_hint")}>
              <Input value={m.model} onChange={(e) => setModel(i, { model: e.target.value })} spellCheck={false} />
            </Field>
          )}
          <div className="grid grid-cols-2 gap-3">
            <Field label={t("recipes.temperature")}>
              <Input type="number" min={0} max={2} step={0.1} value={m.temperature ?? 0.7} onChange={(e) => setModel(i, { temperature: Number(e.target.value) })} />
            </Field>
            <Field label={t("recipes.max_tokens")}>
              <Input type="number" min={16} step={64} value={m.max_tokens ?? 1024} onChange={(e) => setModel(i, { max_tokens: Number(e.target.value) })} />
            </Field>
          </div>
        </div>
      ))}
      {recipe.providers.map((p, i) => (
        <div key={i} className="rounded-lg border border-line bg-surface p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[13px] font-medium">{t("recipes.cloud_provider")}</span>
            <Button size="sm" variant="ghost" onClick={() => onChange({ ...recipe, providers: recipe.providers.filter((_, j) => j !== i), models: recipe.models.map((m) => (m.provider === p.name ? { ...m, provider: "local" } : m)) })} aria-label={t("recipes.remove_provider")}>
              <Trash2 className="size-3.5" />
            </Button>
          </div>
          <div className="flex flex-col gap-2">
            <Input value={p.name} onChange={(e) => setProvider(i, { name: e.target.value })} placeholder={t("recipes.provider_name_placeholder")} aria-label={t("recipes.provider_name")} />
            <Input value={p.endpoint} onChange={(e) => setProvider(i, { endpoint: e.target.value })} placeholder="https://api.openai.com/v1" aria-label={t("recipes.endpoint")} spellCheck={false} />
            <Input type="password" value={p.api_key ?? ""} onChange={(e) => setProvider(i, { api_key: e.target.value })} placeholder={t("recipes.api_key")} aria-label={t("recipes.api_key")} autoComplete="off" />
          </div>
        </div>
      ))}
      <Button size="sm" onClick={() => onChange({ ...recipe, providers: [...recipe.providers, { name: uniqueName("provider", recipe.providers.map((p) => p.name)), endpoint: "", api_key: "" }] })} className="self-start">
        <Plus className="size-3.5" /> {t("recipes.add_provider")}
      </Button>
    </div>
  );
}

// ------------------------------------------------------------------ runs

function RowsTable({ columns, rows }: { columns: string[]; rows: Record<string, unknown>[] }) {
  const cell = (v: unknown) => {
    const text = typeof v === "string" ? v : JSON.stringify(v);
    return text.length > 400 ? text.slice(0, 400) + "…" : text;
  };
  return (
    <div className="mt-3 overflow-x-auto rounded-md border border-line">
      <table className="w-full text-left text-[12.5px]">
        <thead className="bg-surface-2">
          <tr>
            {columns.map((c) => (
              <th key={c} className="px-3 py-2 font-mono font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-line align-top">
              {columns.map((c) => (
                <td key={c} className="max-w-[380px] px-3 py-2 font-mono text-[11.5px] break-words whitespace-pre-wrap">
                  {cell(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RecipeRun({ job }: { job: Job }) {
  const t = useT();
  const { go, setPendingDataset } = useApp();
  const { events, status } = useJob(job.id);
  const state = status ?? job.status;
  const phase = [...events].reverse().find((e) => e.event === "phase")?.name as string | undefined;
  const progress = ([...events].reverse().find((e) => e.event === "progress")?.value as number | undefined) ?? job.progress;
  const seed = events.find((e) => e.event === "seed");
  const result = events.find((e) => e.event === "result") as { rows?: Record<string, unknown>[]; columns?: string[]; count?: number; output?: string } | undefined;
  const error = (events.find((e) => e.event === "error")?.message as string | undefined) ?? job.error;
  const logs = events.filter((e) => e.event === "log").map((e) => e.line as string);
  const output = result?.output ?? (job.result.output as string | null);

  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="truncate font-medium">{job.title}</span>
        <Badge tone={state === "done" ? "ok" : state === "failed" ? "warn" : "neutral"}>{state === "running" ? (phase ?? t("recipes.starting")) : STATE[state] ? t(STATE[state]) : state}</Badge>
      </div>
      {state === "running" && <ProgressBar value={progress ?? null} temper />}
      {seed && (
        <p className="mt-2 text-[12.5px] text-muted">
          {t("recipes.read_docs", { documents: String(seed.documents), chunks: String(seed.chunks) })}
        </p>
      )}
      {state === "failed" && error && (
        <div className="mt-2">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
      {result?.rows && result.columns && <RowsTable columns={result.columns} rows={result.rows} />}
      {state === "done" && output && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button size="sm" variant="primary" onClick={() => { setPendingDataset({ source: output }); go("train"); }}>
            <FlaskConical className="size-3.5" /> {t("recipes.train_rows", { n: result?.count ?? "" })}
          </Button>
          <span className="truncate font-mono text-[11.5px] text-muted" title={output}>
            {output}
          </span>
        </div>
      )}
      {logs.length > 0 && (
        <details className="mt-2 text-[12.5px]">
          <summary className="cursor-pointer text-muted">{t("recipes.output")}</summary>
          <pre className="mt-2 max-h-48 overflow-auto font-mono text-[11.5px] whitespace-pre-wrap text-muted">{logs.slice(-200).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ gallery

function Gallery({ templates, saved, datasets, onOpen, onDelete }: { templates: Recipe[]; saved: Recipe[]; datasets: GeneratedDataset[]; onOpen: (r: Recipe, fromTemplate: boolean) => void; onDelete: (r: Recipe) => void }) {
  const t = useT();
  const { go, setPendingDataset } = useApp();
  const card = (r: Recipe, fromTemplate: boolean, aside?: ReactNode) => (
    <div key={r.id ?? r.name} className="flex flex-col rounded-lg border border-line bg-surface p-4">
      <button onClick={() => onOpen(r, fromTemplate)} className="text-left">
        <span className="font-semibold">{r.name}</span>
        <span className="mt-1 block text-[13px] text-muted">{r.description || t("recipes.columns_n", { n: r.columns.length })}</span>
      </button>
      {aside && <div className="mt-3 flex items-center justify-between gap-2 text-[12.5px] text-muted">{aside}</div>}
    </div>
  );
  return (
    <div className="mx-auto max-w-[1040px] px-7 py-5">
      {saved.length > 0 && (
        <Section title={t("recipes.yours")}>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
            {saved.map((r) =>
              card(
                r,
                false,
                <>
                  <span>{t("recipes.saved_when", { when: relativeDate(r.updated) })}</span>
                  <Button size="sm" variant="ghost" onClick={() => onDelete(r)} aria-label={t("recipes.delete_named", { name: r.name })}>
                    <Trash2 className="size-3.5" />
                  </Button>
                </>,
              ),
            )}
          </div>
        </Section>
      )}
      <Section title={t("recipes.templates")}>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">{templates.map((x) => card(x, true))}</div>
      </Section>
      {datasets.length > 0 && (
        <Section title={t("recipes.generated")}>
          <div className="rounded-lg border border-line bg-surface">
            {datasets.map((d) => (
              <div key={d.path} className="flex items-center gap-3 border-b border-line px-4 py-2.5 last:border-b-0">
                <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]" title={d.path}>
                  {d.name}
                </span>
                <span className="text-[12.5px] text-muted tabular-nums">
                  {t("recipes.rows_n", { n: d.rows })} · {bytes(d.size)}
                </span>
                <Button size="sm" onClick={() => { setPendingDataset({ source: d.path }); go("train"); }}>
                  <FlaskConical className="size-3.5" /> {t("recipes.train")}
                </Button>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ page

export function RecipesPage() {
  const t = useT();
  const { jobs, refreshJobs } = useApp();
  const [templates, setTemplates] = useState<Recipe[]>([]);
  const [saved, setSaved] = useState<Recipe[]>([]);
  const [datasets, setDatasets] = useState<GeneratedDataset[]>([]);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [rows, setRows] = useState(50);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const load = useCallback(async () => {
    const [tpl, s, d] = await Promise.all([api<Recipe[]>("/api/recipes/templates"), api<Recipe[]>("/api/recipes"), api<GeneratedDataset[]>("/api/recipes/datasets")]);
    setTemplates(tpl);
    setSaved(s);
    setDatasets(d);
  }, []);

  useEffect(() => {
    load().catch((e) => setError((e as Error).message));
  }, [load]);

  const runs = useMemo(() => jobs.filter((j) => j.kind === "recipe"), [jobs]);
  const recipeRuns = recipe ? runs.filter((j) => j.title.includes(recipe.name)) : [];
  const finished = runs.filter((j) => j.status !== "running").length;
  useEffect(() => {
    // New generated datasets appear as runs finish.
    if (finished) void load();
  }, [finished, load]);

  function edit(next: Recipe) {
    setRecipe(next);
    setSavedAt(null);
  }

  async function save(): Promise<Recipe | null> {
    if (!recipe) return null;
    setError(null);
    try {
      const stored = await api<Recipe>("/api/recipes", { method: "PUT", json: recipe });
      setRecipe(stored);
      setSavedAt(Date.now());
      void load();
      return stored;
    } catch (e) {
      setError((e as Error).message);
      return null;
    }
  }

  async function run(mode: "preview" | "create") {
    if (!recipe) return;
    setError(null);
    setBusy(true);
    try {
      await api("/api/recipes/run", { json: { recipe, mode, num_records: mode === "preview" ? 3 : rows } });
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!recipe)
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <PageHeader title={t("nav.recipes")} description={t("recipes.description")} />
        <div className="min-h-0 flex-1 overflow-y-auto">
          {error && (
            <div className="mx-7 mt-5">
              <Notice tone="error">{error}</Notice>
            </div>
          )}
          <Gallery
            templates={templates}
            saved={saved}
            datasets={datasets}
            onOpen={(r, fromTemplate) => setRecipe(fromTemplate ? { ...structuredClone(r), id: null, template: r.id } : structuredClone(r))}
            onDelete={async (r) => {
              await api(`/api/recipes/${encodeURIComponent(r.id ?? "")}`, { method: "DELETE" });
              void load();
            }}
          />
        </div>
      </div>
    );

  const aliases = recipe.models.map((m) => m.alias);
  const names = recipe.columns.map((c) => c.name);
  const available = [...(recipe.seed ? SEED_COLUMNS[recipe.seed.type] : []), ...names];
  const codeColumns = recipe.columns.filter((c) => c.column_type === "llm-code" || c.column_type === "llm-text").map((c) => c.name);
  const setColumn = (i: number, c: RecipeColumn) => edit({ ...recipe, columns: recipe.columns.map((x, j) => (j === i ? c : x)) });
  const move = (i: number, delta: number) => {
    const cols = [...recipe.columns];
    [cols[i], cols[i + delta]] = [cols[i + delta], cols[i]];
    edit({ ...recipe, columns: cols });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title={recipe.name || t("recipes.untitled")}
        description={recipe.description}
        actions={
          <>
            <Button variant="ghost" onClick={() => setRecipe(null)}>
              <ArrowLeft className="size-4" /> {t("recipes.all")}
            </Button>
            <Button onClick={() => void save()}>
              <Save className="size-4" /> {t(savedAt ? "recipes.saved" : "recipes.save")}
            </Button>
            <Button onClick={() => void run("preview")} disabled={busy}>
              <Play className="size-4" /> {t("recipes.preview")}
            </Button>
            <span className="flex items-center gap-1.5">
              <Input type="number" min={1} max={100000} value={rows} onChange={(e) => setRows(Math.max(1, Number(e.target.value)))} className="h-9 w-24" aria-label={t("recipes.rows_to_generate")} />
              <Button variant="primary" onClick={() => void run("create")} disabled={busy}>
                <ChefHat className="size-4" /> {t("recipes.generate")}
              </Button>
            </span>
          </>
        }
      />
      <div className="flex min-h-0 flex-1">
        <div className="w-[440px] shrink-0 overflow-y-auto border-r border-line px-7 pb-8">
          <Section title={t("recipes.recipe")}>
            <div className="flex flex-col gap-3">
              <Field label={t("recipes.name")}>
                <Input value={recipe.name} onChange={(e) => edit({ ...recipe, name: e.target.value })} />
              </Field>
              <Field label={t("recipes.description_field")}>
                <Input value={recipe.description} onChange={(e) => edit({ ...recipe, description: e.target.value })} />
              </Field>
            </div>
          </Section>
          <Section title={t("recipes.source")}>
            <SeedEditor seed={recipe.seed} onChange={(seed) => edit({ ...recipe, seed })} />
          </Section>
          <Section title={t("recipes.model")}>
            <ModelEditor recipe={recipe} onChange={edit} />
          </Section>
          <Section title={t("recipes.columns")} aside={<span className="text-[12.5px] text-muted">{t("recipes.top_to_bottom")}</span>}>
            {available.length > 0 && (
              <p className="mb-3 text-[12.5px] text-muted">
                {t("recipes.available")}{" "}
                {available.map((n) => (
                  <code key={n} className="mr-1 rounded bg-surface-2 px-1 font-mono text-[11.5px]">
                    {`{{ ${n} }}`}
                  </code>
                ))}
              </p>
            )}
            <div className="flex flex-col gap-3">
              {recipe.columns.map((c, i) => (
                <ColumnCard
                  key={i}
                  column={c}
                  index={i}
                  count={recipe.columns.length}
                  aliases={aliases}
                  codeColumns={codeColumns}
                  onChange={(next) => setColumn(i, next)}
                  onMove={(d) => move(i, d)}
                  onRemove={() => edit({ ...recipe, columns: recipe.columns.filter((_, j) => j !== i) })}
                />
              ))}
              <AddColumn
                onAdd={(type) => {
                  const base = type === "sampler" ? "choice" : type === "expression" ? "text" : type === "validation" ? "check" : type.replace("llm-", "");
                  edit({ ...recipe, columns: [...recipe.columns, newColumn(type, uniqueName(base, names), aliases[0] ?? "writer")] });
                }}
              />
            </div>
          </Section>
        </div>
        <div className="min-w-0 flex-1 overflow-y-auto px-7 py-5">
          {error && (
            <div className="mb-4">
              <Notice tone="error">{error}</Notice>
            </div>
          )}
          {recipeRuns.length ? (
            <div className="flex flex-col gap-3">
              {recipeRuns.map((j) => (
                <RecipeRun key={j.id} job={j} />
              ))}
            </div>
          ) : (
            <EmptyState icon={<ChefHat className="size-8" />} title={t("recipes.empty")}>
              {t("recipes.empty_hint")}
            </EmptyState>
          )}
        </div>
      </div>
    </div>
  );
}
