import { Loader2, Pencil, Play, Save, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useApp } from "../lib/store";
import { Badge, Button, CopyButton, Field, Input, Select } from "./ui";

type Kind = "yes_no" | "choice" | "score" | "labels";

/** A saved decision, as the engine keeps it (engine decisions.py). */
interface Decision {
  name: string;
  question: string;
  kind: Kind;
  choices: string[];
  scale: [number, number];
  instructions: string | null;
  model: string | null;
  classifier: string | null;
}

interface Answer {
  answer: boolean | string | number | string[];
  reason: string;
  seconds: number;
}

const KINDS: [Kind, string][] = [
  ["yes_no", "Yes or no"],
  ["choice", "One of a set"],
  ["labels", "Labels that apply"],
  ["score", "A score"],
];
const KIND_LABEL = Object.fromEntries(KINDS) as Record<Kind, string>;

const BLANK = { name: "", question: "", kind: "yes_no" as Kind, choices: "", low: 1, high: 10, instructions: "", classifier: "" };

function shown(a: Answer["answer"]) {
  return a === true ? "Yes" : a === false ? "No" : Array.isArray(a) ? a.join(", ") || "None" : String(a);
}

/** Connect → Decisions: questions saved by name that apps answer with one request, and a place to try them. */
export function Decisions() {
  const { engine } = useApp();
  const [saved, setSaved] = useState<Decision[]>([]);
  const [form, setForm] = useState(BLANK);
  const [text, setText] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState<"try" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [classifiers, setClassifiers] = useState<{ name: string; labels: string[] }[]>([]);
  const load = useCallback(() => void api<{ decisions: Decision[] }>("/api/decisions").then((r) => setSaved(r.decisions), () => {}), []);
  useEffect(load, [load]);
  useEffect(() => void api<{ classifiers: { name: string; labels: string[] }[] }>("/api/classifiers").then((r) => setClassifiers(r.classifiers), () => {}), []);
  const byClassifier = !!form.classifier;

  const set = (patch: Partial<typeof BLANK>) => setForm((f) => ({ ...f, ...patch }));
  const body = () => ({
    question: form.question,
    kind: form.kind,
    choices: form.kind === "choice" || form.kind === "labels" ? form.choices.split(",").map((c) => c.trim()).filter(Boolean) : [],
    scale: [form.low, form.high],
    instructions: form.instructions.trim() || null,
    classifier: form.classifier || null,
  });

  async function act(what: "try" | "save") {
    setBusy(what);
    setError(null);
    try {
      if (what === "try") setAnswer(await api<Answer>("/v1/decide", { json: { ...body(), text: text || null } }));
      else {
        setSaved((await api<{ decisions: Decision[] }>(`/api/decisions/${form.name}`, { method: "PUT", json: body() })).decisions);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  function edit(d: Decision) {
    setForm({ name: d.name, question: d.question, kind: d.kind, choices: d.choices.join(", "), low: d.scale[0], high: d.scale[1], instructions: d.instructions ?? "", classifier: d.classifier ?? "" });
    setAnswer(null);
  }

  const curl = (name: string) => `curl ${engine?.url ?? ""}/v1/decisions/${name} \\\n  -H "Authorization: Bearer $TEMPER_KEY" -H "Content-Type: application/json" \\\n  -d '{"text": "..."}'`;
  const ready = form.question.trim() && (byClassifier || !["choice", "labels"].includes(form.kind) || form.choices.split(",").filter((c) => c.trim()).length >= 2);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-4 text-[13px]">
        <Field label="Question">
          <Input value={form.question} onChange={(e) => set({ question: e.target.value })} placeholder="Is this email asking for a refund?" />
        </Field>
        <div className="flex flex-wrap items-end gap-3">
          {classifiers.length > 0 && (
            <Field label="Answered by">
              <Select
                value={form.classifier}
                onChange={(e) => set({ classifier: e.target.value, kind: e.target.value && !["yes_no", "choice"].includes(form.kind) ? "choice" : form.kind })}
                aria-label="Answered by"
                className="w-48"
              >
                <option value="">The loaded model</option>
                {classifiers.map((c) => (
                  <option key={c.name} value={c.name}>
                    Classifier {c.name}
                  </option>
                ))}
              </Select>
            </Field>
          )}
          <Field label="Answer">
            <Select value={form.kind} onChange={(e) => set({ kind: e.target.value as Kind })} aria-label="Kind of answer" className="w-44">
              {KINDS.filter(([k]) => !byClassifier || k === "yes_no" || k === "choice").map(([k, label]) => (
                <option key={k} value={k}>
                  {label}
                </option>
              ))}
            </Select>
          </Field>
          {byClassifier && form.kind === "choice" && (
            <p className="pb-2 text-[12px] text-muted">Its labels: {classifiers.find((c) => c.name === form.classifier)?.labels.join(", ")}</p>
          )}
          {!byClassifier && (form.kind === "choice" || form.kind === "labels") && (
            <div className="min-w-[240px] flex-1">
              <Field label={form.kind === "choice" ? "Choices" : "Labels"}>
                <Input value={form.choices} onChange={(e) => set({ choices: e.target.value })} placeholder="low, normal, urgent" aria-label="Comma-separated choices" />
              </Field>
            </div>
          )}
          {form.kind === "score" && (
            <Field label="From – to">
              <div className="flex items-center gap-1.5">
                <Input type="number" value={form.low} onChange={(e) => set({ low: +e.target.value })} className="w-20" aria-label="Lowest score" />
                <span className="text-muted">–</span>
                <Input type="number" value={form.high} onChange={(e) => set({ high: +e.target.value })} className="w-20" aria-label="Highest score" />
              </div>
            </Field>
          )}
        </div>
        {!byClassifier && (
          <Field label="How to decide" hint="Optional: definitions, examples and edge cases, in plain words.">
            <textarea
              value={form.instructions}
              onChange={(e) => set({ instructions: e.target.value })}
              rows={2}
              className="w-full resize-y rounded-md border border-line bg-bg px-3 py-2 outline-none focus:border-accent"
              placeholder="Urgent means customers can't pay or sign in."
            />
          </Field>
        )}
        <Field label="Try it on">
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} className="w-full resize-y rounded-md border border-line bg-bg px-3 py-2 outline-none focus:border-accent" placeholder="Some text to decide about" />
        </Field>
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => void act("try")} disabled={!ready || !!busy}>
            {busy === "try" ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />} Decide
          </Button>
          <span className="flex-1" />
          <div className="w-56">
            <Input value={form.name} onChange={(e) => set({ name: e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, "-") })} placeholder="name, such as refund-request" aria-label="Name to save it as" />
          </div>
          <Button variant="primary" onClick={() => void act("save")} disabled={!ready || !form.name || !!busy}>
            <Save className="size-3.5" /> Save
          </Button>
        </div>
        {answer && (
          <p className="rounded-md bg-bg px-3 py-2">
            <span className="font-semibold">{shown(answer.answer)}</span> <span className="text-muted">· {answer.reason} · {answer.seconds} s</span>
          </p>
        )}
        {error && <p className="text-[12px] text-danger">{error}</p>}
      </div>

      {saved.length > 0 && (
        <ul className="divide-y divide-line rounded-lg border border-line bg-surface text-[13px]">
          {saved.map((d) => (
            <li key={d.name} className="flex items-center gap-3 px-4 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[12.5px] font-medium">{d.name}</span>
                  <Badge>{KIND_LABEL[d.kind]}</Badge>
                </div>
                <p className="truncate text-[12px] text-muted">
                  {d.question}
                  {d.choices.length > 0 && ` (${d.choices.join(", ")})`}
                  {d.kind === "score" && ` (${d.scale[0]}–${d.scale[1]})`}
                  {d.classifier && ` · classifier ${d.classifier}`}
                </p>
              </div>
              <CopyButton text={curl(d.name)} label="curl" />
              <button className="text-muted hover:text-text" onClick={() => edit(d)} title="Edit" aria-label={`Edit ${d.name}`}>
                <Pencil className="size-3.5" />
              </button>
              <button
                className="text-muted hover:text-danger"
                onClick={() => void api<{ decisions: Decision[] }>(`/api/decisions/${d.name}`, { method: "DELETE" }).then((r) => setSaved(r.decisions))}
                title="Delete"
                aria-label={`Delete ${d.name}`}
              >
                <Trash2 className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
