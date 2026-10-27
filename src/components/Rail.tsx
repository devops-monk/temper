import { Boxes, Cable, ChefHat, Database, FlaskConical, Library, MessageSquare, PackageOpen, Settings } from "lucide-react";
import type { ComponentType } from "react";
import { type Key, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import type { Page } from "../lib/types";
import { LogoMark } from "./Logo";
import { cx } from "./ui";

const ITEMS: { page: Page; label: Key; icon: ComponentType<{ className?: string }> }[] = [
  { page: "chat", label: "nav.chat", icon: MessageSquare },
  { page: "models", label: "nav.models", icon: Boxes },
  { page: "library", label: "nav.library", icon: Library },
  { page: "train", label: "nav.train", icon: FlaskConical },
  { page: "datasets", label: "nav.datasets", icon: Database },
  { page: "recipes", label: "nav.recipes", icon: ChefHat },
  { page: "export", label: "nav.export", icon: PackageOpen },
  { page: "connect", label: "nav.connect", icon: Cable },
];

export function Rail() {
  const { page, go, jobs } = useApp();
  const t = useT();
  const running = (kind: string) => jobs.some((j) => j.kind === kind && j.status === "running");
  const busy: Partial<Record<Page, boolean>> = { models: running("download") || running("convert"), train: running("train"), export: running("export"), recipes: running("recipe"), datasets: running("prepare") };

  const item = (p: Page, label: string, Icon: ComponentType<{ className?: string }>) => (
    <button
      key={p}
      onClick={() => go(p)}
      aria-current={page === p ? "page" : undefined}
      className={cx(
        "group relative flex h-9 w-full items-center gap-2.5 rounded-md px-2.5 text-left transition-colors",
        page === p ? "bg-surface text-text shadow-[0_0_0_1px_var(--line)]" : "text-muted hover:bg-surface/60 hover:text-text",
      )}
    >
      <Icon className="size-[17px] shrink-0" />
      <span className="font-medium">{label}</span>
      {busy[p] && <span className="bg-temper animate-pulse-dot ml-auto size-2 rounded-full" title={t("nav.running")} />}
    </button>
  );

  return (
    <nav className="flex w-[196px] shrink-0 flex-col border-r border-line px-2.5 pt-4 pb-3" aria-label="Main">
      <div className="mb-5 flex items-center gap-2 px-2">
        <LogoMark className="size-[22px]" />
        <span className="text-[17px] font-semibold tracking-[-0.01em]">Temper</span>
      </div>
      <div className="flex flex-col gap-0.5">{ITEMS.map((i) => item(i.page, t(i.label), i.icon))}</div>
      <div className="mt-auto">{item("settings", t("nav.settings"), Settings)}</div>
    </nav>
  );
}
