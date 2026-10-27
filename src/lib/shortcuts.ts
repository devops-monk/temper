import { useEffect } from "react";
import { t } from "./i18n";
import { useApp } from "./store";
import type { Page } from "./types";
import { savedZoom, setZoom, stepZoom } from "./zoom";

export const MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
export const MOD = MAC ? "⌘" : "Ctrl";

/** Pages in the order the rail shows them, for Mod+1…8. */
const PAGES: Page[] = ["chat", "models", "library", "train", "datasets", "recipes", "export", "connect"];

/** Every shortcut, for the list that Mod+/ opens, in the current language. */
export const shortcuts = (): { keys: string; what: string }[] => [
  { keys: `${MOD} N`, what: t("keys.new_chat") },
  { keys: `${MOD} F`, what: t("keys.find") },
  { keys: `${MOD} L`, what: t("keys.focus") },
  { keys: "Esc", what: t("keys.stop") },
  { keys: `${MOD} 1 … ${MOD} 8`, what: t("keys.pages") },
  { keys: `${MOD} ,`, what: t("keys.settings") },
  { keys: `${MOD} +  /  ${MOD} −`, what: t("keys.zoom") },
  { keys: `${MOD} 0`, what: t("keys.actual") },
  { keys: `${MOD} /`, what: t("keys.list") },
];

/** Events for the page that handles a shortcut (the chat page listens for these). */
export type ShortcutEvent = "temper:new-chat" | "temper:find" | "temper:focus-input" | "temper:shortcuts";
export const emit = (name: ShortcutEvent) => window.dispatchEvent(new Event(name));

/** The app-wide keyboard shortcuts. */
export function useShortcuts() {
  const go = useApp((s) => s.go);
  useEffect(() => {
    void (savedZoom() !== 1 && setZoom(savedZoom()));
    const onKey = (e: KeyboardEvent) => {
      if (!(MAC ? e.metaKey : e.ctrlKey) || e.altKey) return;
      const key = e.key.toLowerCase();
      const page = Number(key);
      let handled = true;
      if (key === "n" && !e.shiftKey) {
        go("chat");
        // after the chat page mounts, if it wasn't showing
        setTimeout(() => emit("temper:new-chat"));
      } else if (key === "f" && !e.shiftKey && useApp.getState().page === "chat") emit("temper:find");
      else if (key === "l" && useApp.getState().page === "chat") emit("temper:focus-input");
      else if (key === ",") go("settings");
      else if (key === "/") emit("temper:shortcuts");
      else if (key === "=" || key === "+") void stepZoom(1);
      else if (key === "-" || key === "_") void stepZoom(-1);
      else if (key === "0") void setZoom(1);
      else if (page >= 1 && page <= PAGES.length && !e.shiftKey) go(PAGES[page - 1]);
      else handled = false;
      if (handled) e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);
}
