import { ArrowDownCircle, Cpu, HardDrive } from "lucide-react";
import { bytes } from "../lib/format";
import { useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { useUpdates } from "../lib/updates";
import { LoadedModels } from "./LoadedModels";
import { cx } from "./ui";

const BACKEND_LABEL: Record<string, string> = { cuda: "CUDA", rocm: "ROCm", metal: "Metal", xpu: "Intel XPU", cpu: "CPU only" };

export function StatusBar() {
  const { engine, system, go } = useApp();
  const t = useT();
  const update = useUpdates((u) => (u.status === "available" || u.status === "downloading" ? u.update : null));
  const status = engine?.status ?? "starting";
  const gpu = system?.gpus[0];
  const memUsed = gpu?.memory_used ?? system?.memory.used ?? 0;
  const memTotal = gpu?.memory_total ?? system?.memory.total ?? 0;
  const memShare = memTotal ? memUsed / memTotal : 0;

  return (
    <footer className="flex h-7 shrink-0 items-center gap-5 border-t border-line bg-surface px-3 text-[12px] text-muted">
      <button onClick={() => go("settings")} className="flex items-center gap-1.5 hover:text-text">
        <span
          className={cx(
            "size-2 rounded-full",
            status === "running" && "bg-ok",
            (status === "starting" || status === "installing") && "animate-pulse-dot bg-straw",
            status === "needs_setup" && "bg-straw",
            (status === "crashed" || status === "missing_uv" || status === "stopped" || status === "setup_failed") && "bg-danger",
          )}
        />
        {
          {
            running: t("engine.running"),
            starting: t("engine.starting"),
            installing: t("engine.installing"),
            needs_setup: t("engine.needs_setup"),
            missing_uv: t("engine.needs_setup"),
            setup_failed: t("engine.setup_failed"),
            crashed: t("engine.offline"),
            stopped: t("engine.offline"),
          }[status]
        }
      </button>

      {system && (
        <>
          <span className="flex items-center gap-1.5">
            <Cpu className="size-3.5" />
            {gpu ? gpu.name : system.cpu.name}
            <span className="text-muted/70">{BACKEND_LABEL[system.backend]}</span>
          </span>
          <span className="flex items-center gap-1.5" title={gpu?.unified_memory ? t("status.unified") : t("status.gpu_memory")}>
            <HardDrive className="size-3.5" />
            <span className="relative h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
              <span
                className={cx("absolute inset-y-0 left-0 rounded-full", memShare > 0.9 ? "bg-danger" : memShare > 0.75 ? "bg-straw" : "bg-accent")}
                style={{ width: `${memShare * 100}%` }}
              />
            </span>
            {t("status.of", { used: bytes(memUsed), total: bytes(memTotal) })}
          </span>
        </>
      )}

      {update && (
        <button onClick={() => go("settings")} className="ml-auto flex items-center gap-1.5 text-accent hover:underline">
          <ArrowDownCircle className="size-3.5" /> {t("status.update", { version: update.version })}
        </button>
      )}

      <div className={cx(!update && "ml-auto")}>
        <LoadedModels />
      </div>
    </footer>
  );
}
