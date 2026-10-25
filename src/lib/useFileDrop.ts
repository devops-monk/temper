import { useEffect, useRef, useState } from "react";
import { inTauri } from "./engine";

/** Files dragged onto the window, from the desktop shell (the webview's own drop events carry no paths). */
export function useFileDrop(onDrop: (paths: string[]) => void) {
  const [over, setOver] = useState(false);
  const latest = useRef(onDrop);
  latest.current = onDrop;
  useEffect(() => {
    if (!inTauri) return;
    let unlisten: (() => void) | undefined;
    let gone = false;
    void import("@tauri-apps/api/webview").then(({ getCurrentWebview }) =>
      getCurrentWebview()
        .onDragDropEvent(({ payload }) => {
          if (payload.type === "enter" || payload.type === "over") setOver(true);
          else setOver(false);
          if (payload.type === "drop" && payload.paths.length) latest.current(payload.paths);
        })
        .then((fn) => (gone ? fn() : (unlisten = fn))),
    );
    return () => {
      gone = true;
      unlisten?.();
    };
  }, []);
  return over;
}
