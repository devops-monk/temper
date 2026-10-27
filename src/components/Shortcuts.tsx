import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useT } from "../lib/i18n";
import { shortcuts } from "../lib/shortcuts";

/** The keyboard shortcuts, opened with Mod+/. */
export function ShortcutsDialog() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDialogElement>(null);
  const t = useT();
  useEffect(() => {
    const toggle = () => setOpen((o) => !o);
    window.addEventListener("temper:shortcuts", toggle);
    return () => window.removeEventListener("temper:shortcuts", toggle);
  }, []);
  useEffect(() => {
    const dialog = ref.current;
    if (open && dialog && !dialog.open) dialog.showModal();
    else if (!open && dialog?.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={() => setOpen(false)}
      onClick={(e) => e.target === ref.current && setOpen(false)}
      aria-labelledby="shortcuts-title"
      className="m-auto w-[min(520px,92vw)] rounded-xl border border-line bg-bg p-0 text-text shadow-2xl backdrop:bg-black/40"
    >
      <header className="flex items-center justify-between border-b border-line px-5 py-3.5">
        <h2 id="shortcuts-title" className="text-[15px] font-semibold">
          {t("keys.title")}
        </h2>
        <button onClick={() => setOpen(false)} className="rounded-md p-1 text-muted hover:bg-surface-2 hover:text-text" aria-label={t("common.close")}>
          <X className="size-4" />
        </button>
      </header>
      <dl className="divide-y divide-line px-5 py-1 text-[13px]">
        {shortcuts().map((s) => (
          <div key={s.keys} className="flex items-baseline justify-between gap-6 py-2.5">
            <dt className="text-muted">{s.what}</dt>
            <dd className="shrink-0 font-mono text-[12px]">{s.keys}</dd>
          </div>
        ))}
      </dl>
    </dialog>
  );
}
