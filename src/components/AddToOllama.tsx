import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { ollamaName } from "../lib/format";
import { useT } from "../lib/i18n";
import { useJob } from "../lib/useJob";
import { Button, Input } from "./ui";

let installed: Promise<boolean> | null = null;

/** Adds a GGUF to Ollama (engine ollama.create), when Ollama is installed: asks for a name, then follows the job. */
export function AddToOllama({ path }: { path: string }) {
  const t = useT();
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
    return <span className="text-[12.5px] text-muted">{status === "done" ? t("ollama.in_ollama", { name }) : t("ollama.adding")}</span>;
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      {open ? (
        <>
          <span className="w-48">
            <Input value={name} onChange={(e) => setName(e.target.value)} className="h-7 text-[12.5px]" aria-label={t("ollama.name")} spellCheck={false} />
          </span>
          <Button size="sm" onClick={() => void add()} disabled={!name.trim()}>
            {t("ollama.add")}
          </Button>
        </>
      ) : (
        <Button size="sm" onClick={() => setOpen(true)} title={t("ollama.add_title")}>
          {t("ollama.add_to")}
        </Button>
      )}
      {(error || status === "failed") && <span className="text-[12px] text-danger">{error ?? t("ollama.failed")}</span>}
    </span>
  );
}
