import { inTauri } from "./engine";

/** Interface zoom steps, like a browser's. */
export const ZOOMS = [0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];
const KEY = "temper.zoom";

export function savedZoom(): number {
  try {
    const z = Number(localStorage.getItem(KEY));
    return ZOOMS.includes(z) ? z : 1;
  } catch {
    return 1;
  }
}

/** Zooms the whole window: the webview's own zoom in the app, CSS zoom in a browser. */
export async function setZoom(z: number) {
  try {
    localStorage.setItem(KEY, String(z));
  } catch {
    // private mode: the zoom lasts until the window closes
  }
  if (inTauri) {
    const { getCurrentWebview } = await import("@tauri-apps/api/webview");
    await getCurrentWebview().setZoom(z);
  } else {
    document.documentElement.style.zoom = String(z);
  }
  window.dispatchEvent(new CustomEvent("temper:zoom", { detail: z }));
}

export function stepZoom(direction: 1 | -1) {
  const i = ZOOMS.indexOf(savedZoom());
  return setZoom(ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, i + direction))]);
}
