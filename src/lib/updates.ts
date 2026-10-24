import { getVersion } from "@tauri-apps/api/app";
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { create } from "zustand";
import { inTauri } from "./engine";

/** App updates from GitHub Releases. Each release's `latest.json` is signed in CI; the updater
 * checks the signature against the public key in tauri.conf.json before installing anything. */
type Status = "idle" | "checking" | "current" | "available" | "downloading" | "installing" | "error";

const AUTO_KEY = "temper.autoUpdate";
const RECHECK_MS = 6 * 60 * 60 * 1000;

function readAuto(): boolean {
  try {
    return localStorage.getItem(AUTO_KEY) !== "off";
  } catch {
    return true;
  }
}

interface UpdateState {
  version: string | null;
  status: Status;
  update: Update | null;
  received: number;
  total: number | null;
  error: string | null;
  checkedAt: number | null;
  auto: boolean;
  init: () => void;
  check: () => Promise<void>;
  install: () => Promise<void>;
  setAuto: (auto: boolean) => void;
}

let timer: ReturnType<typeof setInterval> | undefined;

export const useUpdates = create<UpdateState>((set, get) => ({
  version: null,
  status: "idle",
  update: null,
  received: 0,
  total: null,
  error: null,
  checkedAt: null,
  auto: readAuto(),
  init: () => {
    if (!inTauri || timer) return;
    void getVersion().then((version) => set({ version }));
    const tick = () => {
      if (get().auto && get().status !== "downloading" && get().status !== "installing") void get().check();
    };
    setTimeout(tick, 5_000);
    timer = setInterval(tick, RECHECK_MS);
  },
  check: async () => {
    if (!inTauri) return;
    set({ status: "checking", error: null });
    try {
      const update = await check();
      set({ update, status: update ? "available" : "current", checkedAt: Date.now() });
    } catch (e) {
      set({ status: "error", error: `Temper couldn't check for updates. ${String(e)}`, checkedAt: Date.now() });
    }
  },
  install: async () => {
    const update = get().update;
    if (!update) return;
    set({ status: "downloading", received: 0, total: null, error: null });
    try {
      await update.download((event) => {
        if (event.event === "Started") set({ total: event.data.contentLength ?? null });
        else if (event.event === "Progress") set({ received: get().received + event.data.chunkLength });
      });
      set({ status: "installing" });
      // On Windows the installer closes Temper itself; elsewhere the new version starts on relaunch.
      await update.install();
      await relaunch();
    } catch (e) {
      set({ status: "error", error: `The update didn't install. ${String(e)} Your current version still works.` });
    }
  },
  setAuto: (auto) => {
    try {
      localStorage.setItem(AUTO_KEY, auto ? "on" : "off");
    } catch {
      /* preference just won't persist */
    }
    set({ auto });
  },
}));
