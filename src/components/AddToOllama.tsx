import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useJob } from "../lib/useJob";
import { Button, Input } from "./ui";

let installed: Promise<boolean> | null = null;

/** An Ollama name from a GGUF's file name: "20261024-154903-gsm8k-math-tutor-Q4_K_M.gguf" → "gsm8k-math-tutor:q4_k_m". */
export function ollamaName(path: string): string {
  const stem = path.split(/[\\/]/).pop()!.replace(/\.gguf$/i, "");
  const quant = stem.match(/-((?:UD-)?I?Q\d\w*|F16|BF16|F32)$/i)?.[1];
  const base = (quant ? stem.slice(0, -quant.length - 1) : stem).replace(/^\d{8}-\d{6}-/, "");
  const slug = base.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[^a-z0-9]+|-+$/g, "") || "model";
  return quant ? `${slug}:${quant.toLowerCase()}` : slug;
}

/** Adds a GGUF to Ollama (engine ollama.create), when Ollama is installed: asks for a name, then follows the job. */
export function AddToOllama({ path }: { path: string }) {
  const [show, setShow] = useState(false);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(() => ollamaName(path));
  const [job, setJob] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status } = useJob(job);

  useEffect(() => {
    installed ??= api<{ installed: boolean }>("/api/ollama/status").then((s) => s.installed, () => false);
    void installed.then(setShow);
  }, []);
  if (!show) return null;

  async function add() {
    setError(null);
    try {
      setJob((await api<{ id: string }>("/api/ollama/create", { json: { gguf: path, name } })).id);
      setOpen(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (job && status !== "failed") {
    return <span className="text-[12.5px] text-muted">{status === "done" ? `In Ollama as ${name}` : "Adding to Ollama…"}</span>;
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      {open ? (
        <>
          <span className="w-48">
            <Input value={name} onChange={(e) => setName(e.target.value)} className="h-7 text-[12.5px]" aria-label="Name in Ollama" spellCheck={false} />
          </span>
          <Button size="sm" onClick={() => void add()} disabled={!name.trim()}>
            Add
          </Button>
        </>
      ) : (
        <Button size="sm" onClick={() => setOpen(true)} title="Copy this model into Ollama, so apps that use Ollama can run it">
          Add to Ollama
        </Button>
      )}
      {(error || status === "failed") && <span className="text-[12px] text-danger">{error ?? "Ollama couldn't add it; see the job's output."}</span>}
    </span>
  );
}
