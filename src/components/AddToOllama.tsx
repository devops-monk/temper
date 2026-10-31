import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { ollamaName } from "../lib/format";
import { useJob } from "../lib/useJob";
import { Button, Input } from "./ui";

let installed: Promise<boolean> | null = null;

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
