import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { bytes } from "../lib/format";
import { useT } from "../lib/i18n";
import { CopyButton } from "./ui";

/** What an integrated GPU may use (engine igpu.py), and the commands that give it more. */
interface Shared {
  kind: "apple" | "amd_apu" | "none";
  total: number;
  limit?: number;
  default?: number;
  carve_out?: number;
  custom?: boolean;
  suggested?: number;
  commands?: { raise: string; keep?: string; reset?: string; params?: string };
}

function Command({ label, text }: { label: string; text: string }) {
  return (
    <div className="mt-2">
      <p className="text-[12px] text-muted">{label}</p>
      <div className="mt-1 flex items-center gap-1.5 rounded-md border border-line bg-bg px-2 py-1">
        <code className="min-w-0 flex-1 overflow-x-auto font-mono text-[11.5px] whitespace-nowrap">{text}</code>
        <CopyButton text={text} />
      </div>
    </div>
  );
}

/** Settings → This computer → GPU memory: for Macs and AMD APUs, whose GPU shares the computer's memory. */
export function SharedMemory() {
  const t = useT();
  const [shared, setShared] = useState<Shared | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => void api<Shared>("/api/shared-memory").then(setShared, () => {}), []);
  if (!shared || shared.kind === "none" || !shared.commands) return null;
  const amd = shared.kind === "amd_apu";
  return (
    <div className="border-t border-line py-2.5 text-[13px]">
      <div className="flex items-baseline justify-between gap-6">
        <span className="text-muted">{t("shared.title")}</span>
        <span className="text-right">
          {t(shared.custom ? "shared.limit_custom" : "shared.limit_default", { limit: bytes(shared.limit ?? 0), total: bytes(shared.total) })}{" "}
          <button className="text-accent underline-offset-2 hover:underline" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {t(open ? "shared.hide" : "shared.more")}
          </button>
        </span>
      </div>
      {open && (
        <div className="mt-2 text-[12.5px]">
          <p className="text-muted">
            {amd ? t("shared.amd_about", { carve: bytes(shared.carve_out ?? 0), suggested: bytes(shared.suggested ?? 0) }) : t("shared.mac_about", { suggested: bytes(shared.suggested ?? 0) })}
          </p>
          <Command label={t(amd ? "shared.amd_raise" : "shared.mac_raise")} text={shared.commands.raise} />
          {shared.commands.keep && <Command label={t("shared.mac_keep")} text={shared.commands.keep} />}
          {shared.commands.reset && <Command label={t("shared.mac_reset")} text={shared.commands.reset} />}
          {amd && <p className="mt-2 text-muted">{t("shared.amd_bios")}</p>}
        </div>
      )}
    </div>
  );
}
