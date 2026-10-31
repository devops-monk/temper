import { ArrowDown, ArrowLeft, ArrowUp, ChefHat, FilePlus, FlaskConical, FolderPlus, Play, Plus, Save, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Badge, Button, EmptyState, Field, Input, Notice, PageHeader, ProgressBar, Section, Select, Toggle, cx } from "../components/ui";
import { api, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { GeneratedDataset, Job, Recipe, RecipeColumn, RecipeSeed } from "../lib/types";
import { useJob } from "../lib/useJob";

type ColumnType = RecipeColumn["column_type"];

const COLUMN_TYPES: { type: ColumnType; label: string; note: string }[] = [
  { type: "llm-text", label: "Generated text", note: "The model writes free text from a prompt." },
  { type: "llm-structured", label: "Generated fields", note: "The model fills named fields, such as question and answer." },
  { type: "llm-code", label: "Generated code", note: "The model writes code in one language." },
  { type: "llm-judge", label: "Grade", note: "The model scores other columns against your criteria." },
  { type: "sampler", label: "Random choice", note: "Picks a value from a list, to vary the rows." },
  { type: "expression", label: "Template", note: "Combines other columns with {{ name }} placeholders." },
  { type: "validation", label: "Code check", note: "Lints a code column and records whether it passes." },
];
const TYPE_LABEL = Object.fromEntries(COLUMN_TYPES.map((t) => [t.type, t.label])) as Record<ColumnType, string>;

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
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);
  return (
    <div>
      <TextArea
        value={text}
        rows={rows}
        mono
        onChange={(t) => {
          setText(t);
          try {
            onChange(JSON.parse(t));
            setError(null);
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      />
      {error && <span className="mt-1 block text-xs text-danger">Not valid JSON yet: {error}</span>}
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
        <Input value={column.name} onChange={(e) => set({ name: e.target.value.replace(/\s+/g, "_") })} className="h-8 max-w-[200px] font-mono text-[12.5px]" aria-label="Column name" />
        <Badge tone={llm ? "accent" : "neutral"}>{TYPE_LABEL[type] ?? type}</Badge>
        <span className="ml-auto flex items-center">
          <Button size="sm" variant="ghost" onClick={() => onMove(-1)} disabled={index === 0} aria-label="Move up">
            <ArrowUp className="size-3.5" />
          </Button>
          <Button size="sm" variant="ghost" onClick={() => onMove(1)} disabled={index === count - 1} aria-label="Move down">
            <ArrowDown className="size-3.5" />
          </Button>
          <Button size="sm" variant="ghost" onClick={onRemove} aria-label={`Remove ${column.name}`}>
            <Trash2 className="size-3.5" />
          </Button>
        </span>
      </div>
      <div className="flex flex-col gap-3">
        {llm && (
          <>
            <Field label="Prompt" hint="Use {{ column_name }} to insert another column's value.">
              <TextArea value={String(column.prompt ?? "")} onChange={(v) => set({ prompt: v })} rows={4} />
            </Field>
            {aliases.length > 1 && (
              <Field label="Model">
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
          <Field label="Language">
            <Select value={String(column.code_lang)} onChange={(e) => set({ code_lang: e.target.value })}>
              {CODE_LANGS.map((l) => (
                <option key={l}>{l}</option>
              ))}
            </Select>
          </Field>
        )}
        {type === "llm-structured" &&
          (simpleFields ? (
            <Field label="Fields" hint={<>Comma-separated. Use them as {"{{ "}{column.name}.field{" }}"} in later columns.</>}>
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
            <Field label="Output schema (JSON Schema)">
              <JsonField value={column.output_format} onChange={(v) => set({ output_format: v })} />
            </Field>
          ))}
        {type === "llm-judge" && (
          <Field label="Scores" hint="Each score has a name, a description and the options the model picks from.">
            <JsonField value={column.scores} onChange={(v) => set({ scores: v })} rows={7} />
          </Field>
        )}
        {type === "sampler" &&
          (column.sampler_type === "category" ? (
            <Field label="Values" hint="One per line. Each row picks one at random.">
              <TextArea value={(params.values ?? []).map(String).join("\n")} onChange={(v) => set({ params: { ...params, values: lines(v), weights: undefined } })} rows={4} />
            </Field>
          ) : (
            <Field label={`Parameters (${String(column.sampler_type)})`}>
              <JsonField value={column.params} onChange={(v) => set({ params: v })} />
            </Field>
          ))}
        {type === "expression" && (
          <Field label="Template" hint="Jinja: {{ column }}, {{ fields.question }}, {% if %} … {% endif %}.">
            <TextArea value={String(column.expr ?? "")} onChange={(v) => set({ expr: v })} rows={3} mono />
          </Field>
        )}
        {type === "validation" && (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Column to check">
              <Select value={String((column.target_columns as string[])[0] ?? "")} onChange={(e) => set({ target_columns: [e.target.value] })}>
                <option value="">Choose…</option>
                {codeColumns.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </Select>
            </Field>
            <Field label="Language">
              <Select value={validatorParams.code_lang ?? "python"} onChange={(e) => set({ validator_params: { ...validatorParams, code_lang: e.target.value } })}>
                {CODE_LANGS.map((l) => (
                  <option key={l}>{l}</option>
                ))}
              </Select>
            </Field>
          </div>
        )}
        <Toggle checked={!column.drop} onChange={(keep) => set({ drop: !keep })} label="Keep this column in the dataset" />
      </div>
    </div>
  );
}

function AddColumn({ onAdd }: { onAdd: (type: ColumnType) => void }) {
  const [open, setOpen] = useState(false);
  if (!open)
    return (
      <Button onClick={() => setOpen(true)} className="w-full border-dashed">
        <Plus className="size-4" /> Add column
      </Button>
    );
  return (
    <div className="rounded-lg border border-line bg-surface-2 p-2">
      <div className="mb-1 flex items-center justify-between px-1.5">
        <span className="text-[13px] font-medium">Add a column</span>
        <Button size="sm" variant="ghost" onClick={() => setOpen(false)} aria-label="Close">
          <X className="size-3.5" />
        </Button>
      </div>
      {COLUMN_TYPES.map((t) => (
        <button
          key={t.type}
          onClick={() => {
            onAdd(t.type);
            setOpen(false);
          }}
          className="block w-full rounded-md px-2.5 py-1.5 text-left hover:bg-surface"
        >
          <span className="font-medium">{t.label}</span>
          <span className="block text-[12.5px] text-muted">{t.note}</span>
        </button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ source and model

function SeedEditor({ seed, onChange }: { seed: RecipeSeed | null; onChange: (s: RecipeSeed | null) => void }) {
  const kinds: [RecipeSeed["type"] | "none", string][] = [
    ["none", "None"],
    ["documents", "Documents"],
    ["file", "File"],
    ["hf", "Hugging Face"],
  ];
  const current = seed?.type ?? "none";
  const sources = seed?.sources ?? [];

  async function addPaths(directory: boolean) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open(directory ? { directory: true, multiple: true } : { multiple: true, filters: [{ name: "Documents", extensions: DOC_EXTENSIONS }] });
    const paths = Array.isArray(picked) ? picked : picked ? [picked] : [];
    if (paths.length && seed) onChange({ ...seed, sources: [...sources, ...paths] });
  }

  async function browseFile() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const file = await open({ filters: [{ name: "Datasets", extensions: ["jsonl", "json", "csv", "parquet"] }] });
    if (typeof file === "string" && seed) onChange({ ...seed, path: file });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex rounded-md border border-line bg-surface p-0.5" role="radiogroup" aria-label="Source">
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
      {!seed && <p className="text-[13px] text-muted">Rows are built only from random choices and the model.</p>}
      {seed?.type === "documents" && (
        <>
          <Field label="Files, folders and links" hint="One per line: PDF, Word, text or Markdown files, folders, web pages, or a GitHub repository URL. Columns can use {{ chunk_text }} and {{ source_file }}.">
            <TextArea value={sources.join("\n")} onChange={(v) => onChange({ ...seed, sources: v.split("\n") })} rows={4} placeholder={"~/Documents/handbook.pdf\nhttps://github.com/owner/repo"} mono />
          </Field>
          {inTauri && (
            <div className="flex gap-2">
              <Button size="sm" onClick={() => void addPaths(false)}>
                <FilePlus className="size-3.5" /> Add files
              </Button>
              <Button size="sm" onClick={() => void addPaths(true)}>
                <FolderPlus className="size-3.5" /> Add folder
              </Button>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Chunk size" hint="Characters per chunk.">
              <Input type="number" min={200} step={100} value={seed.chunk_size ?? 1200} onChange={(e) => onChange({ ...seed, chunk_size: Number(e.target.value) })} />
            </Field>
            <Field label="Overlap" hint="Characters shared by neighbours.">
              <Input type="number" min={0} step={50} value={seed.chunk_overlap ?? 200} onChange={(e) => onChange({ ...seed, chunk_overlap: Number(e.target.value) })} />
            </Field>
          </div>
        </>
      )}
      {(seed?.type === "file" || seed?.type === "hf") && (
        <Field
          label={seed.type === "file" ? "Dataset file" : "Dataset"}
          hint={seed.type === "file" ? "A .jsonl, .json, .csv or .parquet file. Its columns can be used as {{ name }}." : "A Hugging Face dataset id, such as yahma/alpaca-cleaned. Its parquet files are used."}
        >
          <div className="flex gap-2">
            <Input value={seed.path ?? ""} onChange={(e) => onChange({ ...seed, path: e.target.value })} spellCheck={false} className="font-mono text-[12.5px]" />
            {seed.type === "file" && inTauri && (
              <Button onClick={() => void browseFile()} aria-label="Choose a file" className="px-2.5">
                <FilePlus className="size-4" />
              </Button>
            )}
          </div>
        </Field>
      )}
      {seed && (
        <Field label="Order">
          <Select value={seed.sampling ?? "ordered"} onChange={(e) => onChange({ ...seed, sampling: e.target.value as RecipeSeed["sampling"] })}>
            <option value="ordered">In order</option>
            <option value="shuffle">Shuffled</option>
          </Select>
        </Field>
      )}
    </div>
  );
}

function ModelEditor({ recipe, onChange }: { recipe: Recipe; onChange: (r: Recipe) => void }) {
  const runtime = useApp((s) => s.runtime);
  const setModel = (i: number, patch: Partial<Recipe["models"][number]>) => onChange({ ...recipe, models: recipe.models.map((m, j) => (j === i ? { ...m, ...patch } : m)) });
  const setProvider = (i: number, patch: Partial<Recipe["providers"][number]>) => onChange({ ...recipe, providers: recipe.providers.map((p, j) => (j === i ? { ...p, ...patch } : p)) });

  return (
    <div className="flex flex-col gap-4">
      {recipe.models.map((m, i) => (
        <div key={i} className="flex flex-col gap-3">
          {recipe.models.length > 1 && <span className="font-mono text-[12.5px] text-muted">{m.alias}</span>}
          <Field
            label="Provider"
            hint={m.provider === "local" ? (runtime?.state === "ready" ? <>Uses {runtime.model}, the model loaded for chat.</> : "Uses the model loaded for chat. Load one on the Models page first.") : undefined}
          >
            <Select value={m.provider} onChange={(e) => setModel(i, { provider: e.target.value })}>
              <option value="local">Chat model on this computer</option>
              {recipe.providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </Select>
          </Field>
          {m.provider !== "local" && (
            <Field label="Model name" hint="As the provider names it, such as gpt-5-mini.">
              <Input value={m.model} onChange={(e) => setModel(i, { model: e.target.value })} spellCheck={false} />
            </Field>
          )}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Temperature">
              <Input type="number" min={0} max={2} step={0.1} value={m.temperature ?? 0.7} onChange={(e) => setModel(i, { temperature: Number(e.target.value) })} />
            </Field>
            <Field label="Max tokens">
              <Input type="number" min={16} step={64} value={m.max_tokens ?? 1024} onChange={(e) => setModel(i, { max_tokens: Number(e.target.value) })} />
            </Field>
          </div>
        </div>
      ))}
      {recipe.providers.map((p, i) => (
        <div key={i} className="rounded-lg border border-line bg-surface p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[13px] font-medium">Cloud provider</span>
            <Button size="sm" variant="ghost" onClick={() => onChange({ ...recipe, providers: recipe.providers.filter((_, j) => j !== i), models: recipe.models.map((m) => (m.provider === p.name ? { ...m, provider: "local" } : m)) })} aria-label="Remove provider">
              <Trash2 className="size-3.5" />
            </Button>
          </div>
          <div className="flex flex-col gap-2">
            <Input value={p.name} onChange={(e) => setProvider(i, { name: e.target.value })} placeholder="Name, e.g. openai" aria-label="Provider name" />
            <Input value={p.endpoint} onChange={(e) => setProvider(i, { endpoint: e.target.value })} placeholder="https://api.openai.com/v1" aria-label="Endpoint" spellCheck={false} />
            <Input type="password" value={p.api_key ?? ""} onChange={(e) => setProvider(i, { api_key: e.target.value })} placeholder="API key" aria-label="API key" autoComplete="off" />
          </div>
        </div>
      ))}
      <Button size="sm" onClick={() => onChange({ ...recipe, providers: [...recipe.providers, { name: uniqueName("provider", recipe.providers.map((p) => p.name)), endpoint: "", api_key: "" }] })} className="self-start">
        <Plus className="size-3.5" /> Add an OpenAI-compatible provider
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
        <Badge tone={state === "done" ? "ok" : state === "failed" ? "warn" : "neutral"}>{state === "running" ? (phase ?? "Starting") : state}</Badge>
      </div>
      {state === "running" && <ProgressBar value={progress ?? null} temper />}
      {seed && (
        <p className="mt-2 text-[12.5px] text-muted">
          Read {String(seed.documents)} documents into {String(seed.chunks)} chunks.
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
            <FlaskConical className="size-3.5" /> Train on these {result?.count ?? ""} rows
          </Button>
          <span className="truncate font-mono text-[11.5px] text-muted" title={output}>
            {output}
          </span>
        </div>
      )}
      {logs.length > 0 && (
        <details className="mt-2 text-[12.5px]">
          <summary className="cursor-pointer text-muted">Output</summary>
          <pre className="mt-2 max-h-48 overflow-auto font-mono text-[11.5px] whitespace-pre-wrap text-muted">{logs.slice(-200).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ gallery

function Gallery({ templates, saved, datasets, onOpen, onDelete }: { templates: Recipe[]; saved: Recipe[]; datasets: GeneratedDataset[]; onOpen: (r: Recipe, fromTemplate: boolean) => void; onDelete: (r: Recipe) => void }) {
  const { go, setPendingDataset } = useApp();
  const card = (r: Recipe, fromTemplate: boolean, aside?: ReactNode) => (
    <div key={r.id ?? r.name} className="flex flex-col rounded-lg border border-line bg-surface p-4">
      <button onClick={() => onOpen(r, fromTemplate)} className="text-left">
        <span className="font-semibold">{r.name}</span>
        <span className="mt-1 block text-[13px] text-muted">{r.description || `${r.columns.length} columns`}</span>
      </button>
      {aside && <div className="mt-3 flex items-center justify-between gap-2 text-[12.5px] text-muted">{aside}</div>}
    </div>
  );
  return (
    <div className="mx-auto max-w-[1040px] px-7 py-5">
      {saved.length > 0 && (
        <Section title="Your recipes">
          <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
            {saved.map((r) =>
              card(
                r,
                false,
                <>
                  <span>Saved {relativeDate(r.updated)}</span>
                  <Button size="sm" variant="ghost" onClick={() => onDelete(r)} aria-label={`Delete ${r.name}`}>
                    <Trash2 className="size-3.5" />
                  </Button>
                </>,
              ),
            )}
          </div>
        </Section>
      )}
      <Section title="Start from a template">
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">{templates.map((t) => card(t, true))}</div>
      </Section>
      {datasets.length > 0 && (
        <Section title="Generated datasets">
          <div className="rounded-lg border border-line bg-surface">
            {datasets.map((d) => (
              <div key={d.path} className="flex items-center gap-3 border-b border-line px-4 py-2.5 last:border-b-0">
                <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]" title={d.path}>
                  {d.name}
                </span>
                <span className="text-[12.5px] text-muted tabular-nums">
                  {d.rows} rows · {bytes(d.size)}
                </span>
                <Button size="sm" onClick={() => { setPendingDataset({ source: d.path }); go("train"); }}>
                  <FlaskConical className="size-3.5" /> Train
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
    const [t, s, d] = await Promise.all([api<Recipe[]>("/api/recipes/templates"), api<Recipe[]>("/api/recipes"), api<GeneratedDataset[]>("/api/recipes/datasets")]);
    setTemplates(t);
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
        <PageHeader title="Recipes" description="Generate training data with a model: from your documents, from topics you choose, or from data you already have." />
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
        title={recipe.name || "Untitled recipe"}
        description={recipe.description}
        actions={
          <>
            <Button variant="ghost" onClick={() => setRecipe(null)}>
              <ArrowLeft className="size-4" /> All recipes
            </Button>
            <Button onClick={() => void save()}>
              <Save className="size-4" /> {savedAt ? "Saved" : "Save"}
            </Button>
            <Button onClick={() => void run("preview")} disabled={busy}>
              <Play className="size-4" /> Preview 3 rows
            </Button>
            <span className="flex items-center gap-1.5">
              <Input type="number" min={1} max={100000} value={rows} onChange={(e) => setRows(Math.max(1, Number(e.target.value)))} className="h-9 w-24" aria-label="Rows to generate" />
              <Button variant="primary" onClick={() => void run("create")} disabled={busy}>
                <ChefHat className="size-4" /> Generate dataset
              </Button>
            </span>
          </>
        }
      />
      <div className="flex min-h-0 flex-1">
        <div className="w-[440px] shrink-0 overflow-y-auto border-r border-line px-7 pb-8">
          <Section title="Recipe">
            <div className="flex flex-col gap-3">
              <Field label="Name">
                <Input value={recipe.name} onChange={(e) => edit({ ...recipe, name: e.target.value })} />
              </Field>
              <Field label="Description">
                <Input value={recipe.description} onChange={(e) => edit({ ...recipe, description: e.target.value })} />
              </Field>
            </div>
          </Section>
          <Section title="Source">
            <SeedEditor seed={recipe.seed} onChange={(seed) => edit({ ...recipe, seed })} />
          </Section>
          <Section title="Model">
            <ModelEditor recipe={recipe} onChange={edit} />
          </Section>
          <Section title="Columns" aside={<span className="text-[12.5px] text-muted">Built top to bottom</span>}>
            {available.length > 0 && (
              <p className="mb-3 text-[12.5px] text-muted">
                Available in prompts:{" "}
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
            <EmptyState icon={<ChefHat className="size-8" />} title="Preview before you generate">
              Preview builds 3 rows so you can check the prompts. When they look right, generate the full dataset and train on it.
            </EmptyState>
          )}
        </div>
      </div>
    </div>
  );
}
