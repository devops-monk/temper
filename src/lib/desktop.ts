import { useEffect } from "react";
import { create } from "zustand";
import { inTauri } from "./engine";
import { useApp } from "./store";
import type { Page } from "./types";
import { useFileDrop } from "./useFileDrop";

/** What the desktop shell was asked to open (desktop.rs): a `temper://` link, a model file or a dataset file. */
export type Open = { kind: "page"; page: Page; query: string | null } | { kind: "model"; path: string } | { kind: "dataset"; path: string };

const MODEL = /\.gguf$/i;
const DATASET = /\.(jsonl|json|csv|parquet)$/i;

/** Hand-offs to the pages: each page takes its part and clears it. */
interface Opened {
  models: string[];
  search: string | null;
  dataset: string | null;
  set: (o: Partial<Pick<Opened, "models" | "search" | "dataset">>) => void;
}

export const useOpened = create<Opened>((set) => ({ models: [], search: null, dataset: null, set }));

/** Files dropped anywhere become opens too: models go to the library, data files to Datasets. */
export function routePaths(paths: string[]): Open[] {
  return paths.flatMap((path): Open[] => (MODEL.test(path) ? [{ kind: "model", path }] : DATASET.test(path) ? [{ kind: "dataset", path }] : []));
}

export function applyOpens(opens: Open[]) {
  const { go } = useApp.getState();
  const { set, models } = useOpened.getState();
  let page: Page | null = null;
  const files = opens.flatMap((o) => (o.kind === "model" ? [o.path] : []));
  if (files.length) {
    set({ models: [...models, ...files] });
    page = "models";
  }
  for (const o of opens) {
    if (o.kind === "dataset") {
      set({ dataset: o.path }); // one at a time: the page previews a single source
      page = "datasets";
    } else if (o.kind === "page") {
      if (o.page === "models" && o.query) set({ search: o.query });
      page = o.page;
    }
  }
  if (page) go(page);
}

/** Pages with a drop target of their own. */
const DROPS: Page[] = ["models", "library", "chat"];

/**
 * Takes what the shell queued (files opened with Temper, `temper://` links) once the engine is up, and
 * whenever a later one arrives. Returns whether files are being dragged over a page without its own drop target.
 */
export function useDesktopOpens(ready: boolean): boolean {
  useEffect(() => {
    if (!inTauri || !ready) return;
    let unlisten: (() => void) | undefined;
    let gone = false;
    const take = async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      applyOpens(await invoke<Open[]>("take_opens"));
    };
    void take();
    void import("@tauri-apps/api/event").then(({ listen }) => listen("temper-open", () => void take()).then((fn) => (gone ? fn() : (unlisten = fn))));
    return () => {
      gone = true;
      unlisten?.();
    };
  }, [ready]);

  const page = useApp((s) => s.page);
  const own = DROPS.includes(page);
  const over = useFileDrop((paths) => {
    if (ready && !DROPS.includes(useApp.getState().page)) applyOpens(routePaths(paths));
  });
  return over && ready && !own;
}
