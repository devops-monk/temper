import { ArrowDownCircle, Cpu, HardDrive } from "lucide-react";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import { useUpdates } from "../lib/updates";
import { cx } from "./ui";

const BACKEND_LABEL: Record<string, string> = { cuda: "CUDA", rocm: "ROCm", metal: "Metal", xpu: "Intel XPU", cpu: "CPU only" };

export function StatusBar() {
  const { engine, system, runtime, go } = useApp();
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
            running: "Engine ready",
            starting: "Starting engine",
            installing: "Installing engine",
            needs_setup: "Setup needed",
            missing_uv: "Setup needed",
            setup_failed: "Setup failed",
            crashed: "Engine offline",
            stopped: "Engine offline",
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
          <span className="flex items-center gap-1.5" title={gpu?.unified_memory ? "Unified memory" : "GPU memory"}>
            <HardDrive className="size-3.5" />
            <span className="relative h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
              <span
                className={cx("absolute inset-y-0 left-0 rounded-full", memShare > 0.9 ? "bg-danger" : memShare > 0.75 ? "bg-straw" : "bg-accent")}
                style={{ width: `${memShare * 100}%` }}
              />
            </span>
            {bytes(memUsed)} of {bytes(memTotal)}
          </span>
        </>
      )}

      {update && (
        <button onClick={() => go("settings")} className="ml-auto flex items-center gap-1.5 text-accent hover:underline">
          <ArrowDownCircle className="size-3.5" /> Temper {update.version} is available
        </button>
      )}

      <button onClick={() => go(runtime?.state === "ready" ? "chat" : "models")} className={cx(!update && "ml-auto", "flex items-center gap-1.5 hover:text-text")}>
        {runtime?.state === "ready" ? (
          <>
            <span className="size-2 rounded-full bg-accent" /> {runtime.model}
          </>
        ) : runtime?.state === "loading" ? (
          <>
            <span className="animate-pulse-dot size-2 rounded-full bg-straw" /> Loading {runtime.model}
          </>
        ) : (
          "No model loaded"
        )}
      </button>
    </footer>
  );
}
