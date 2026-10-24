import type { EngineInfo } from "./types";

/** True inside the desktop shell; false when the UI runs in a plain browser (`npm run dev`). */
export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

let current: EngineInfo | null = null;

/** Ask the shell where the engine is. In a browser, use VITE_ENGINE_URL / VITE_ENGINE_TOKEN. */
export async function fetchEngineInfo(): Promise<EngineInfo> {
  if (inTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    current = await invoke<EngineInfo>("engine_info");
  } else {
    const url = import.meta.env.VITE_ENGINE_URL ?? "http://127.0.0.1:8765";
    const token = import.meta.env.VITE_ENGINE_TOKEN ?? "dev";
    let status: EngineInfo["status"] = "starting";
    try {
      status = (await fetch(`${url}/api/health`)).ok ? "running" : "crashed";
    } catch {
      status = "stopped";
    }
    current = {
      url,
      token,
      status,
      detail: status === "stopped" ? "Start it with: cd engine && uv run temper-engine --token dev" : null,
    };
  }
  return current;
}

export async function restartEngine(): Promise<void> {
  if (!inTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  current = await invoke<EngineInfo>("engine_restart");
}

export class EngineError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

function connection(): EngineInfo {
  if (!current || current.status !== "running") throw new EngineError("The engine is not running yet.", 0);
  return current;
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { url, token } = connection();
  const { json, ...rest } = init;
  const res = await fetch(url + path, {
    ...rest,
    method: rest.method ?? (json !== undefined ? "POST" : "GET"),
    headers: {
      Authorization: `Bearer ${token}`,
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
      ...rest.headers,
    },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : (body.error?.message ?? JSON.stringify(body.detail ?? body));
    } catch {
      /* keep the status text */
    }
    throw new EngineError(message, res.status);
  }
  return res.json() as Promise<T>;
}

/**
 * Reads a server-sent-event stream with fetch (EventSource cannot send the auth header).
 * Calls `onData` with each `data:` payload until the stream ends or `signal` aborts.
 */
export async function stream(
  path: string,
  onData: (data: string) => void,
  init: RequestInit & { json?: unknown; signal?: AbortSignal } = {},
): Promise<void> {
  const { url, token } = connection();
  const { json, ...rest } = init;
  const res = await fetch(url + path, {
    ...rest,
    method: json !== undefined ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "text/event-stream",
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    body: json !== undefined ? JSON.stringify(json) : undefined,
  });
  if (!res.ok || !res.body) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      message = body.detail ?? body.error?.message ?? message;
    } catch {
      /* keep the status text */
    }
    throw new EngineError(message, res.status);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data:")) onData(line.slice(5).trimStart());
      }
    }
  }
}
