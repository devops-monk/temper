import { Cpu, HardDrive } from "lucide-react";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import { cx } from "./ui";

const BACKEND_LABEL: Record<string, string> = { cuda: "CUDA", rocm: "ROCm", metal: "Metal", xpu: "Intel XPU", cpu: "CPU only" };

export function StatusBar() {
  const { engine, system, runtime, go } = useApp();
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
            status === "starting" && "animate-pulse-dot bg-straw",
            (status === "crashed" || status === "missing_uv" || status === "stopped") && "bg-danger",
          )}
        />
        {status === "running" ? "Engine ready" : status === "starting" ? "Starting engine" : status === "missing_uv" ? "Setup needed" : "Engine offline"}
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

      <button onClick={() => go(runtime?.state === "ready" ? "chat" : "models")} className="ml-auto flex items-center gap-1.5 hover:text-text">
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
