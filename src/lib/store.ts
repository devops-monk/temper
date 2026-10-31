import { create } from "zustand";
import { api, fetchEngineInfo } from "./engine";
import type { EngineInfo, Job, Page, RuntimeStatus, SystemInfo } from "./types";

type Theme = "system" | "light" | "dark";

/** A Hugging Face id or local path, with the subset and split to train on when it has several. */
export interface PendingDataset {
  source: string;
  split?: string;
  config?: string;
}

const PAGES: Page[] = ["chat", "research", "models", "library", "train", "datasets", "recipes", "export", "images", "audio", "connect", "settings"];

/** `#train` opens the Train page; used by deep links and screenshots. */
function initialPage(): Page {
  const hash = location.hash.slice(1) as Page;
  return PAGES.includes(hash) ? hash : "chat";
}

function readTheme(): Theme {
  try {
    return (localStorage.getItem("temper.theme") as Theme) || "system";
  } catch {
    return "system";
  }
}

export function applyTheme(theme: Theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
}

interface AppState {
  page: Page;
  engine: EngineInfo | null;
  system: SystemInfo | null;
  runtime: RuntimeStatus | null;
  jobs: Job[];
  theme: Theme;
  /** Dataset picked on the Datasets page, handed to the training form. */
  pendingDataset: PendingDataset | null;
  /** Model picked in the catalog, handed to the training form. */
  pendingBaseModel: string | null;
  go: (page: Page) => void;
  setTheme: (theme: Theme) => void;
  setPendingDataset: (dataset: PendingDataset | null) => void;
  setPendingBaseModel: (model: string | null) => void;
  refreshEngine: () => Promise<void>;
  refreshSystem: () => Promise<void>;
  refreshRuntime: () => Promise<void>;
  refreshJobs: () => Promise<void>;
}

export const useApp = create<AppState>((set, get) => ({
  page: initialPage(),
  engine: null,
  system: null,
  runtime: null,
  jobs: [],
  theme: readTheme(),
  pendingDataset: null,
  pendingBaseModel: null,
  go: (page) => set({ page }),
  setTheme: (theme) => {
    try {
      localStorage.setItem("temper.theme", theme);
    } catch {
      /* preference just won't persist */
    }
    applyTheme(theme);
    set({ theme });
  },
  setPendingDataset: (pendingDataset) => set({ pendingDataset }),
  setPendingBaseModel: (pendingBaseModel) => set({ pendingBaseModel }),
  refreshEngine: async () => {
    const was = get().engine?.status;
    const engine = await fetchEngineInfo().catch(() => null);
    set({ engine });
    if (engine?.status === "running" && was !== "running") {
      void get().refreshSystem();
      void get().refreshRuntime();
      void get().refreshJobs();
    }
  },
  refreshSystem: async () => {
    const system = await api<SystemInfo>("/api/system").catch(() => null);
    if (system) set({ system });
  },
  refreshRuntime: async () => {
    const runtime = await api<RuntimeStatus>("/api/runtime").catch(() => null);
    if (runtime) set({ runtime });
  },
  refreshJobs: async () => {
    const jobs = await api<Job[]>("/api/jobs").catch(() => null);
    if (jobs) set({ jobs });
  },
}));
