import { Monitor, Moon, RotateCw, Sun } from "lucide-react";
import type { MouseEvent, ReactNode } from "react";
import { Badge, Button, Notice, PageHeader, Section, cx } from "../components/ui";
import { inTauri, restartEngine } from "../lib/engine";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";

/** In the desktop app, links open in the system browser instead of the app window. */
async function openExternal(e: MouseEvent<HTMLAnchorElement>) {
  if (!inTauri) return;
  e.preventDefault();
  const { openUrl } = await import("@tauri-apps/plugin-opener");
  await openUrl(e.currentTarget.href);
}

const STATUS_LABEL = { running: "Running", starting: "Starting", crashed: "Stopped after errors", missing_uv: "uv not installed", stopped: "Not running" };

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-t border-line py-2.5 text-[13px] first:border-t-0">
      <span className="text-muted">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

export function SettingsPage() {
  const { theme, setTheme, engine, system, refreshEngine } = useApp();

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title="Settings" />
      <div className="min-h-0 flex-1 overflow-y-auto px-7 pb-10">
        <div className="max-w-[640px]">
          <Section title="Appearance">
            <div className="inline-flex rounded-lg border border-line bg-surface p-1" role="radiogroup" aria-label="Theme">
              {(
                [
                  ["system", "Match system", Monitor],
                  ["light", "Light", Sun],
                  ["dark", "Dark", Moon],
                ] as const
              ).map(([value, label, Icon]) => (
                <button
                  key={value}
                  role="radio"
                  aria-checked={theme === value}
                  onClick={() => setTheme(value)}
                  className={cx("flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium", theme === value ? "bg-surface-2 text-text shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:text-text")}
                >
                  <Icon className="size-3.5" /> {label}
                </button>
              ))}
            </div>
          </Section>

          <Section
            title="Engine"
            aside={
              inTauri && (
                <Button size="sm" onClick={async () => { await restartEngine(); await refreshEngine(); }}>
                  <RotateCw className="size-3.5" /> Restart engine
                </Button>
              )
            }
          >
            <div className="rounded-lg border border-line bg-surface px-4">
              <Row label="Status">
                <Badge tone={engine?.status === "running" ? "ok" : "warn"}>{STATUS_LABEL[engine?.status ?? "starting"]}</Badge>
              </Row>
              <Row label="Address">
                <span className="font-mono text-[12.5px]">{engine?.url ?? "–"}</span>
              </Row>
            </div>
            {engine?.detail && (
              <div className="mt-3">
                <Notice>{engine.detail}</Notice>
              </div>
            )}
          </Section>

          {system && (
            <Section title="This computer">
              <div className="rounded-lg border border-line bg-surface px-4">
                <Row label="System">
                  {system.os.name} {system.os.release} ({system.os.machine})
                </Row>
                <Row label="Processor">
                  {system.cpu.name}, {system.cpu.cores} cores
                </Row>
                <Row label="Memory">{bytes(system.memory.total)}</Row>
                {system.gpus.map((g) => (
                  <Row key={g.index} label={`GPU ${g.index}`}>
                    {g.name}, {bytes(g.memory_total)} {g.unified_memory ? "unified" : "VRAM"}
                  </Row>
                ))}
                <Row label="Free disk space">{bytes(system.disk.free)}</Row>
                <Row label="Chat runtime">{system.capabilities.gguf_inference ? <Badge tone="ok">llama.cpp found</Badge> : <Badge tone="warn">llama.cpp not installed</Badge>}</Row>
                <Row label="Trainer">{system.capabilities.trainer ? <Badge tone="ok">{system.capabilities.trainer === "mlx" ? "MLX" : "Unsloth"}</Badge> : <Badge tone="warn">Not installed</Badge>}</Row>
              </div>
            </Section>
          )}

          <Section title="About">
            <div className="rounded-lg border border-line bg-surface px-4">
              <Row label="Version">Temper 0.1.0</Row>
              <Row label="Training">
                Powered by{" "}
                <a className="text-accent hover:underline" href="https://github.com/unslothai/unsloth" target="_blank" rel="noreferrer" onClick={openExternal}>
                  Unsloth
                </a>{" "}
                (Apache-2.0) and MLX
              </Row>
              <Row label="Chat">llama.cpp (MIT)</Row>
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}
