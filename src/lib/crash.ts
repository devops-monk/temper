import { api, EngineError } from "./engine";

/** A saved crash report (engine crashes.py): scrubbed of home folders and keys, kept on this computer. */
export interface Crash {
  id: string;
  time: number;
  where: "engine" | "interface";
  error: string;
  trace: string;
  context: string;
  version: string;
  system: string;
  report_url: string;
}

const seen = new Set<string>();

/** Saves an interface error as a crash report (once per message a session) and tells CrashNotice. */
export async function reportCrash(error: unknown, context: string): Promise<Crash | null> {
  if (!(error instanceof Error) || error instanceof EngineError) return null; // the engine's answers aren't crashes
  if (error instanceof TypeError && /fetch|network|load failed/i.test(error.message)) return null; // the engine is restarting
  const message = `${error.name}: ${error.message}`;
  if (seen.has(message)) return null;
  seen.add(message);
  try {
    const crash = await api<Crash>("/api/crashes", { json: { error: message.slice(0, 2000), trace: (error.stack ?? "").slice(0, 20000), context: context.slice(0, 500) } });
    window.dispatchEvent(new CustomEvent<Crash>("temper:crash", { detail: crash }));
    return crash;
  } catch {
    return null;
  }
}

/** Saves errors nothing else caught. */
export function watchCrashes() {
  window.addEventListener("error", (e) => void reportCrash(e.error, "window"));
  window.addEventListener("unhandledrejection", (e) => void reportCrash(e.reason, "promise"));
}
