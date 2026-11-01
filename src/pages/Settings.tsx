import { Minus, Monitor, Moon, Plus, RotateCw, Sun } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Accounts } from "../components/Accounts";
import { SharedMemory } from "../components/SharedMemory";
import { Diagnostics } from "../components/Diagnostics";
import { Engines } from "../components/Engines";
import { SpeechSettings } from "../components/SpeechSettings";
import { CodeSandbox } from "../components/Tools";
import { McpServers } from "../components/McpServers";
import { WebSearch } from "../components/WebSearch";
import { KnowledgeBases } from "../components/Knowledge";
import { SkillsSettings } from "../components/Skills";
import { GlobalInstructions } from "../components/Instructions";
import { HfAccount } from "../components/HfAccount";
import { HubSource } from "../components/HubSource";
import { Storage } from "../components/Storage";
import { LicensesDialog } from "../components/Licenses";
import { Badge, Button, Notice, PageHeader, Section, Select, Toggle, cx } from "../components/ui";
import { api, inTauri, installEngine, restartEngine } from "../lib/engine";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import { Privacy } from "../components/Privacy";
import { useTour } from "../components/Tour";
import { StatsSection } from "../components/Stats";
import { type Key, LANGUAGES, type Language, useLanguage, useT, withNodes } from "../lib/i18n";
import { MOD } from "../lib/shortcuts";
import { useUpdates } from "../lib/updates";
import { ZOOMS, savedZoom, setZoom, stepZoom } from "../lib/zoom";

const STATUS_LABEL: Record<string, Key> = {
  running: "settings.s_running",
  starting: "settings.s_starting",
  crashed: "settings.s_crashed",
  missing_uv: "settings.s_missing_uv",
  stopped: "settings.s_stopped",
  needs_setup: "settings.s_needs_setup",
  installing: "settings.s_installing",
  setup_failed: "settings.s_setup_failed",
  locked: "settings.s_locked",
};
const PROFILE_LABEL: Record<"chat" | "train", Key> = { chat: "settings.p_chat", train: "settings.p_train" };

function LanguageControl() {
  const { language, setLanguage } = useLanguage();
  const t = useT();
  return (
    <div className="mt-4 flex items-center justify-between gap-4 text-[13px]">
      <span>
        <span className="font-medium">{t("settings.language")}</span>
        {language !== "en" && <span className="block text-[12px] text-muted">{t("settings.language_hint")}</span>}
      </span>
      <span className="w-40 shrink-0">
        <Select value={language} onChange={(e) => setLanguage(e.target.value as Language)} className="h-8 text-[13px]" aria-label={t("settings.language")}>
          {LANGUAGES.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </Select>
      </span>
    </div>
  );
}

function ZoomControl() {
  const [zoom, setZoomState] = useState(savedZoom);
  const t = useT();
  useEffect(() => {
    const on = (e: Event) => setZoomState((e as CustomEvent<number>).detail);
    window.addEventListener("temper:zoom", on);
    return () => window.removeEventListener("temper:zoom", on);
  }, []);
  return (
    <div className="mt-4 flex items-center justify-between gap-4 text-[13px]">
      <span>
        <span className="font-medium">{t("settings.zoom")}</span>
        <span className="block text-[12px] text-muted">{t("settings.zoom_hint", { mod: MOD })}</span>
      </span>
      <span className="flex items-center gap-1">
        <Button size="sm" variant="ghost" onClick={() => void stepZoom(-1)} disabled={zoom <= ZOOMS[0]} aria-label={t("settings.zoom_out")} className="w-8 px-0">
          <Minus className="size-3.5" />
        </Button>
        <button onClick={() => void setZoom(1)} className="w-12 text-center tabular-nums hover:text-accent" title={t("settings.actual_size")}>
          {Math.round(zoom * 100)}%
        </button>
        <Button size="sm" variant="ghost" onClick={() => void stepZoom(1)} disabled={zoom >= ZOOMS[ZOOMS.length - 1]} aria-label={t("settings.zoom_in")} className="w-8 px-0">
          <Plus className="size-3.5" />
        </Button>
      </span>
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-t border-line py-2.5 text-[13px] first:border-t-0">
      <span className="text-muted">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

function Updates() {
  const t = useT();
  const { version, status, update, received, total, error, checkedAt, auto, check, install, setAuto } = useUpdates();
  const jobs = useApp((a) => a.jobs);
  const busy = jobs.some((j) => j.status === "running");
  const working = status === "checking" || status === "downloading" || status === "installing";

  async function restart() {
    if (busy && !confirm(t("settings.busy_confirm"))) return;
    await install();
  }

  return (
    <Section
      title={t("settings.updates")}
      aside={
        inTauri &&
        !update && (
          <Button size="sm" disabled={working} onClick={() => void check()}>
            <RotateCw className={cx("size-3.5", status === "checking" && "animate-spin")} /> {t("settings.check_updates")}
          </Button>
        )
      }
    >
      <div className="rounded-lg border border-line bg-surface px-4">
        <Row label={t("settings.version")}>Temper {version ?? "0.1.0"}</Row>
        <Row label={t("settings.status")}>
          {status === "checking" && t("settings.checking")}
          {status === "current" && t("settings.up_to_date")}
          {status === "idle" && t(inTauri ? "settings.not_checked" : "settings.desktop_only")}
          {status === "error" && t("settings.couldnt_check")}
          {update && status === "available" && <Badge tone="accent">{t("settings.available", { version: update.version })}</Badge>}
          {status === "downloading" && (total ? t("settings.downloading_pct", { pct: Math.round((received / total) * 100) }) : t("settings.downloading"))}
          {status === "installing" && t("settings.installing_restart")}
        </Row>
        {checkedAt && <Row label={t("settings.last_checked")}>{new Date(checkedAt).toLocaleString()}</Row>}
      </div>
      {update && (status === "available" || status === "downloading" || status === "installing") && (
        <div className="mt-3 rounded-lg border border-line bg-surface px-4 py-3">
          {update.body && <p className="text-[13px] whitespace-pre-line text-muted">{update.body.split("\n").slice(0, 8).join("\n")}</p>}
          <Button className="mt-3" size="sm" variant="primary" disabled={working} onClick={() => void restart()}>
            {t("settings.update_restart")}
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
          {t("settings.auto")}
        </label>
      )}
    </Section>
  );
}

/** Closing the window can leave Temper in the tray, so the API and the loaded model stay up for other apps. */
function KeepRunning() {
  const t = useT();
  const [on, setOn] = useState<boolean | null>(null);
  useEffect(() => {
    void import("@tauri-apps/api/core").then(({ invoke }) => invoke<boolean>("get_background").then(setOn, () => setOn(false)));
  }, []);
  async function change(value: boolean) {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("set_background", { on: value });
    setOn(value);
  }
  if (on === null) return null;
  return (
    <div className="mt-3">
      <Toggle checked={on} onChange={(v) => void change(v)} label={t("settings.keep_running")} />
      <p className="mt-1 text-[12px] text-muted">{t("settings.keep_running_hint")}</p>
    </div>
  );
}

/** The engine as a login item (engine daemon.py), so the API is up before the app is opened. */
function StartAtLogin() {
  const t = useT();
  const [on, setOn] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api<{ login: boolean }>("/api/daemon").then((d) => setOn(d.login), () => {});
  }, []);
  async function change(value: boolean) {
    setError(null);
    try {
      setOn((await api<{ login: boolean }>("/api/daemon/login", { method: "PUT", json: { enabled: value } })).login);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  if (on === null) return null;
  return (
    <div className="mt-3">
      <Toggle checked={on} onChange={(v) => void change(v)} label={t("settings.login")} />
      <p className="mt-1 text-[12px] text-muted">
        {withNodes(t("settings.login_hint"), { cmd: <span className="font-mono">temper daemon install</span> })}
      </p>
      {error && <p className="mt-1 text-[12px] text-danger">{error}</p>}
    </div>
  );
}

export function SettingsPage() {
  const { theme, setTheme, engine, system, refreshEngine, go, me } = useApp();
  const member = !!me && !me.admin; // a member's settings are their appearance and account
  const t = useT();
  const setup = engine?.setup;
  const [licenses, setLicenses] = useState(false);
  const startTour = useTour((s) => s.start);

  async function reinstall(profile: "chat" | "train", clean: boolean) {
    if (clean && !confirm(t("settings.reinstall_confirm"))) return;
    await installEngine(profile, clean);
    await refreshEngine();
    go("chat");
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader
        title={t("settings.title")}
        actions={
          <Button size="sm" variant="ghost" onClick={startTour}>
            {t("settings.tour")}
          </Button>
        }
      />
      <div className="min-h-0 flex-1 overflow-y-auto px-7 pb-10">
        <div className="max-w-[640px]">
          <Section title={t("settings.appearance")}>
            <div className="inline-flex rounded-lg border border-line bg-surface p-1" role="radiogroup" aria-label={t("settings.theme")}>
              {(
                [
                  ["system", t("settings.system"), Monitor],
                  ["light", t("settings.light"), Sun],
                  ["dark", t("settings.dark"), Moon],
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
            <ZoomControl />
            <LanguageControl />
          </Section>

          <Accounts />

          {!member && (
          <>

          <Section
            title={t("settings.engine")}
            aside={
              inTauri && (
                <Button size="sm" onClick={async () => { await restartEngine(); await refreshEngine(); }}>
                  <RotateCw className="size-3.5" /> {t("settings.restart_engine")}
                </Button>
              )
            }
          >
            <div className="rounded-lg border border-line bg-surface px-4">
              <Row label={t("settings.status")}>
                <Badge tone={engine?.status === "running" ? "ok" : "warn"}>{t(STATUS_LABEL[engine?.status ?? "starting"] ?? "settings.s_starting")}</Badge>
              </Row>
              <Row label={t("settings.address")}>
                <span className="font-mono text-[12.5px]">{engine?.url ?? "–"}</span>
              </Row>
              {setup?.installed && <Row label={t("settings.installed")}>{t(PROFILE_LABEL[setup.installed])}</Row>}
            </div>
            {setup?.installed && engine?.status !== "installing" && (
              <div className="mt-3 flex flex-wrap gap-2">
                {setup.installed === "chat" && setup.hardware?.can_train && (
                  <Button size="sm" variant="primary" onClick={() => reinstall("train", false)}>
                    {t("settings.add_training")}
                  </Button>
                )}
                <Button size="sm" onClick={() => reinstall(setup.installed ?? "chat", true)}>
                  {t("settings.reinstall")}
                </Button>
              </div>
            )}
            {engine?.detail && (
              <div className="mt-3">
                <Notice>{engine.detail}</Notice>
              </div>
            )}
            {inTauri && <KeepRunning />}
            {engine?.status === "running" && <StartAtLogin />}
          </Section>

          {engine?.status === "running" && <GlobalInstructions />}

          {engine?.status === "running" && <SkillsSettings />}

          {engine?.status === "running" && <CodeSandbox />}

          {engine?.status === "running" && <KnowledgeBases />}

          {engine?.status === "running" && <WebSearch />}

          {engine?.status === "running" && <McpServers />}

          {engine?.status === "running" && <Engines />}
          {engine?.status === "running" && <SpeechSettings />}

          {engine?.status === "running" && <StatsSection />}

          {engine?.status === "running" && <Privacy />}

          {system && (
            <Section title={t("settings.computer")}>
              <div className="rounded-lg border border-line bg-surface px-4">
                <Row label={t("settings.system_row")}>
                  {system.os.name} {system.os.release} ({system.os.machine})
                </Row>
                <Row label={t("settings.processor")}>
                  {t("settings.cores", { name: system.cpu.name, n: system.cpu.cores })}
                </Row>
                <Row label={t("settings.memory")}>{bytes(system.memory.total)}</Row>
                {system.gpus.map((g) => (
                  <Row key={g.index} label={`GPU ${g.index}`}>
                    {g.name}, {bytes(g.memory_total)} {g.unified_memory ? t("settings.unified") : "VRAM"}
                  </Row>
                ))}
                <SharedMemory />
                <Row label={t("settings.free_disk")}>{bytes(system.disk.free)}</Row>
                <Row label={t("settings.chat_runtime")}>{system.capabilities.gguf_inference ? <Badge tone="ok">{t("settings.llama_found")}</Badge> : <Badge tone="warn">{t("settings.llama_missing")}</Badge>}</Row>
                <Row label={t("settings.trainer")}>{system.capabilities.trainer ? <Badge tone="ok">{system.capabilities.trainer === "mlx" ? "MLX" : "PyTorch"}</Badge> : <Badge tone="warn">{t("settings.not_installed")}</Badge>}</Row>
              </div>
            </Section>
          )}

          <HubSource ready={engine?.status === "running"} />

          <HfAccount ready={engine?.status === "running"} />

          <Storage ready={engine?.status === "running"} />

          <Diagnostics ready={engine?.status === "running"} installed={setup?.installed ?? "chat"} onReinstall={(profile, clean) => void reinstall(profile, clean)} />

          <Updates />
          </>
          )}

          <Section
            title={t("settings.about")}
            aside={
              <Button size="sm" onClick={() => setLicenses(true)}>
                {t("settings.licences")}
              </Button>
            }
          >
            <div className="rounded-lg border border-line bg-surface px-4">
              <Row label={t("settings.licence")}>Apache-2.0</Row>
              <Row label={t("settings.training")}>{t("settings.training_libs")}</Row>
              <Row label={t("settings.chat")}>llama.cpp (MIT)</Row>
            </div>
            <LicensesDialog open={licenses} onClose={() => setLicenses(false)} />
          </Section>
        </div>
      </div>
    </div>
  );
}
