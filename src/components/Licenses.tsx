import { X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { openExternal } from "../lib/engine";
import { Input } from "./ui";

interface Package {
  name: string;
  version: string;
  ecosystem: "crate" | "npm";
  url: string;
}
interface Group {
  license: string;
  text: string;
  packages: Package[];
}
interface Licenses {
  groups: Group[];
  declared: Group[];
}

/** Installed on the user's computer by Temper at runtime rather than shipped in the installer. */
const DOWNLOADED = [
  ["uv", "Apache-2.0 or MIT", "https://github.com/astral-sh/uv"],
  ["Python", "PSF-2.0", "https://www.python.org/"],
  ["llama.cpp", "MIT", "https://github.com/ggml-org/llama.cpp"],
  ["MLX and MLX-LM", "MIT", "https://github.com/ml-explore/mlx-lm"],
  ["PyTorch", "BSD-3-Clause", "https://github.com/pytorch/pytorch"],
  ["Transformers, PEFT, TRL and Datasets", "Apache-2.0", "https://github.com/huggingface"],
  ["Hugging Face Hub", "Apache-2.0", "https://github.com/huggingface/huggingface_hub"],
  ["FastAPI and Uvicorn", "MIT and BSD-3-Clause", "https://github.com/fastapi/fastapi"],
  ["NVIDIA DataDesigner", "Apache-2.0", "https://github.com/NVIDIA-NeMo/DataDesigner"],
  ["vLLM (if installed)", "Apache-2.0", "https://github.com/vllm-project/vllm"],
  ["SGLang (if installed)", "Apache-2.0", "https://github.com/sgl-project/sglang"],
  ["pypdf", "BSD-3-Clause", "https://github.com/py-pdf/pypdf"],
  ["python-docx", "MIT", "https://github.com/python-openxml/python-docx"],
] as const;

function PackageLink({ p }: { p: Package }) {
  return (
    <button onClick={() => void openExternal(p.url)} className="hover:text-accent hover:underline" title={p.url}>
      {p.name} <span className="text-muted">{p.version}</span>
    </button>
  );
}

function LicenseGroup({ group }: { group: Group }) {
  return (
    <li className="border-t border-line py-3 first:border-t-0">
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[13px]">
        {group.packages.map((p) => (
          <PackageLink key={`${p.ecosystem}:${p.name}@${p.version}`} p={p} />
        ))}
      </div>
      <details className="mt-1.5">
        <summary className="cursor-pointer text-[12.5px] text-muted hover:text-text">{group.license || "Licence"} text</summary>
        <pre className="mt-2 max-h-80 overflow-auto rounded-md bg-surface-2 p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap">{group.text}</pre>
      </details>
    </li>
  );
}

export function LicensesDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [data, setData] = useState<Licenses | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
      // About 400 KB of text, so it loads only when someone asks for it.
      if (!data) void import("../licenses.json").then((m) => setData(m.default as Licenses));
    } else if (!open && dialog.open) dialog.close();
  }, [open, data]);

  const groups = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    return [...data.groups, ...data.declared]
      .map((g) => (q ? { ...g, packages: g.packages.filter((p) => p.name.toLowerCase().includes(q)) } : g))
      .filter((g) => g.packages.length);
  }, [data, query]);
  const total = data ? [...data.groups, ...data.declared].reduce((n, g) => n + g.packages.length, 0) : 0;

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      aria-labelledby="licenses-title"
      className="m-auto h-[min(760px,88vh)] w-[min(760px,92vw)] overflow-hidden rounded-xl border border-line bg-bg p-0 text-text shadow-2xl backdrop:bg-black/40"
    >
      <div className="flex h-full flex-col">
        <header className="flex items-start justify-between gap-4 border-b border-line px-6 py-4">
          <div>
            <h2 id="licenses-title" className="text-[17px] font-semibold">
              Open-source licences
            </h2>
            <p className="mt-0.5 text-[13px] text-muted">Temper is licensed under Apache-2.0 and includes the open-source software below.</p>
          </div>
          <button onClick={onClose} className="rounded-md p-1 text-muted hover:bg-surface-2 hover:text-text" aria-label="Close">
            <X className="size-4" />
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
          <section className="py-4">
            <h3 className="font-semibold">Downloaded on your computer</h3>
            <p className="mt-1 text-[13px] text-muted">
              First-run setup and the Models page install these from their publishers. They run under their own licences, and Temper does not modify them.
            </p>
            <ul className="mt-2 grid grid-cols-1 gap-x-6 text-[13px] sm:grid-cols-2">
              {DOWNLOADED.map(([name, license, url]) => (
                <li key={name} className="flex justify-between gap-3 border-t border-line py-1.5">
                  <button onClick={() => void openExternal(url)} className="text-left hover:text-accent hover:underline">
                    {name}
                  </button>
                  <span className="text-right text-muted">{license}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="py-4">
            <div className="flex items-center justify-between gap-4">
              <h3 className="font-semibold">Included in Temper{total ? ` (${total} packages)` : ""}</h3>
              <Input className="h-8 max-w-56 text-[13px]" placeholder="Find a package" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Find a package" />
            </div>
            {!data ? (
              <p className="mt-3 text-[13px] text-muted">Loading…</p>
            ) : groups.length ? (
              <ul className="mt-2">
                {groups.map((g, i) => (
                  <LicenseGroup key={i} group={g} />
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-[13px] text-muted">No package matches “{query}”.</p>
            )}
          </section>
        </div>
      </div>
    </dialog>
  );
}
