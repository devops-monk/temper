import { Monitor, Moon, RotateCw, Sun } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Diagnostics } from "../components/Diagnostics";
import { Engines } from "../components/Engines";
import { HfAccount } from "../components/HfAccount";
import { HubSource } from "../components/HubSource";
import { Storage } from "../components/Storage";
import { LicensesDialog } from "../components/Licenses";
import { Badge, Button, Notice, PageHeader, Section, cx } from "../components/ui";
import { inTauri, installEngine, restartEngine } from "../lib/engine";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import { useUpdates } from "../lib/updates";

const STATUS_LABEL = {
  running: "Running",
  starting: "Starting",
  crashed: "Stopped after errors",
  missing_uv: "uv not installed",
  stopped: "Not running",
  needs_setup: "Not set up",
  installing: "Installing",
  setup_failed: "Setup failed",
};
const PROFILE_LABEL = { chat: "Chat only", train: "Chat and training" };

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-t border-line py-2.5 text-[13px] first:border-t-0">
      <span className="text-muted">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

function Updates() {
  const { version, status, update, received, total, error, checkedAt, auto, check, install, setAuto } = useUpdates();
  const jobs = useApp((a) => a.jobs);
  const busy = jobs.some((j) => j.status === "running");
  const working = status === "checking" || status === "downloading" || status === "installing";

  async function restart() {
    if (busy && !confirm("A download, training run or export is still running. Updating restarts Temper and stops it. Update anyway?")) return;
    await install();
  }

  return (
    <Section
      title="Updates"
      aside={
        inTauri &&
        !update && (
          <Button size="sm" disabled={working} onClick={() => void check()}>
            <RotateCw className={cx("size-3.5", status === "checking" && "animate-spin")} /> Check for updates
          </Button>
        )
      }
    >
      <div className="rounded-lg border border-line bg-surface px-4">
        <Row label="Version">Temper {version ?? "0.1.0"}</Row>
        <Row label="Status">
          {status === "checking" && "Checking…"}
          {status === "current" && "Temper is up to date"}
          {status === "idle" && (inTauri ? "Not checked yet" : "Updates work in the desktop app")}
          {status === "error" && "Couldn't check"}
          {update && status === "available" && <Badge tone="accent">Temper {update.version} is available</Badge>}
          {status === "downloading" && `Downloading${total ? ` ${Math.round((received / total) * 100)}%` : "…"}`}
          {status === "installing" && "Installing, Temper will restart"}
        </Row>
        {checkedAt && <Row label="Last checked">{new Date(checkedAt).toLocaleString()}</Row>}
      </div>
      {update && (status === "available" || status === "downloading" || status === "installing") && (
        <div className="mt-3 rounded-lg border border-line bg-surface px-4 py-3">
          {update.body && <p className="text-[13px] whitespace-pre-line text-muted">{update.body.split("\n").slice(0, 8).join("\n")}</p>}
          <Button className="mt-3" size="sm" variant="primary" disabled={working} onClick={() => void restart()}>
            Update and restart
          </Button>
        </div>
      )}
      {error && (
        <div className="mt-3">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
      {inTauri && (
        <label className="mt-3 flex cursor-pointer items-center gap-2 text-[13px]">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} className="accent-[var(--accent)]" />
          Check for updates automatically
        </label>
      )}
    </Section>
  );
}

export function SettingsPage() {
  const { theme, setTheme, engine, system, refreshEngine, go } = useApp();
  const setup = engine?.setup;
  const [licenses, setLicenses] = useState(false);

  async function reinstall(profile: "chat" | "train", clean: boolean) {
    if (clean && !confirm("Reinstall the engine? Temper deletes and rebuilds its Python environment. Your models, runs and exports are kept.")) return;
    await installEngine(profile, clean);
    await refreshEngine();
    go("chat");
  }

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
              {setup?.installed && <Row label="Installed">{PROFILE_LABEL[setup.installed]}</Row>}
            </div>
            {setup?.installed && engine?.status !== "installing" && (
              <div className="mt-3 flex flex-wrap gap-2">
                {setup.installed === "chat" && setup.hardware?.can_train && (
                  <Button size="sm" variant="primary" onClick={() => reinstall("train", false)}>
                    Add training support
                  </Button>
                )}
                <Button size="sm" onClick={() => reinstall(setup.installed ?? "chat", true)}>
                  Reinstall engine
                </Button>
              </div>
            )}
            {engine?.detail && (
              <div className="mt-3">
                <Notice>{engine.detail}</Notice>
              </div>
            )}
          </Section>

          {engine?.status === "running" && <Engines />}

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
                <Row label="Trainer">{system.capabilities.trainer ? <Badge tone="ok">{system.capabilities.trainer === "mlx" ? "MLX" : "PyTorch"}</Badge> : <Badge tone="warn">Not installed</Badge>}</Row>
              </div>
            </Section>
          )}

          <HubSource ready={engine?.status === "running"} />

          <HfAccount ready={engine?.status === "running"} />

          <Storage ready={engine?.status === "running"} />

          <Diagnostics ready={engine?.status === "running"} installed={setup?.installed ?? "chat"} onReinstall={(profile, clean) => void reinstall(profile, clean)} />

          <Updates />

          <Section
            title="About"
            aside={
              <Button size="sm" onClick={() => setLicenses(true)}>
                Open-source licences
              </Button>
            }
          >
            <div className="rounded-lg border border-line bg-surface px-4">
              <Row label="Licence">Apache-2.0</Row>
              <Row label="Training">PyTorch, TRL and MLX</Row>
              <Row label="Chat">llama.cpp (MIT)</Row>
            </div>
            <LicensesDialog open={licenses} onClose={() => setLicenses(false)} />
          </Section>
        </div>
      </div>
    </div>
  );
}
