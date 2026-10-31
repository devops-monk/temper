import { Loader2, Square, Volume2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { connection } from "../lib/engine";
import { Button } from "./ui";

/** Text to speech as the engine reports it (engine tts.py). */
export interface TtsStatus {
  installed: boolean;
  downloaded: boolean;
  ready: boolean;
  voice: string;
  repo: string;
  size: number;
  running: boolean;
}

let cache: Promise<TtsStatus | null> | null = null;

/** Whether text to speech can run, asked once per page load; `refresh` after installing or downloading. */
export function useTts(): { tts: TtsStatus | null; refresh: () => void } {
  const [tts, setTts] = useState<TtsStatus | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    cache ??= fetchJson<TtsStatus>("/api/tts").catch(() => null);
    let stale = false;
    void cache.then((s) => !stale && setTts(s));
    return () => {
      stale = true;
    };
  }, [tick]);
  return {
    tts,
    refresh: () => {
      cache = null;
      setTick((t) => t + 1);
    },
  };
}

async function fetchJson<T>(path: string): Promise<T> {
  const { url, token } = connection();
  const r = await fetch(url + path, { headers: { Authorization: `Bearer ${token}` } });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json() as Promise<T>;
}

/** The text as it reads aloud: Markdown's marks, code blocks and links' addresses left out. */
export function speakable(markdown: string): string {
  return markdown
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, "")
    .replace(/(\*\*|__|\*|_|~~)(?=\S)([^*_~]+?)\1/g, "$2")
    .replace(/\|/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

/** Audio of `text` from the engine, as a URL an <audio> can play (it can't send the auth header itself). */
export async function speakUrl(text: string, voice?: string, speed = 1): Promise<string> {
  const { url, token } = connection();
  const r = await fetch(`${url}/api/tts/speak`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice: voice ?? null, speed }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `${r.status} ${r.statusText}`);
  }
  return URL.createObjectURL(await r.blob());
}

/** A reply's "Read aloud" button. */
export function ReadAloud({ text }: { text: string }) {
  const { tts } = useTts();
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const [error, setError] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => () => audio.current?.pause(), []);
  if (!tts?.ready) return null;

  async function play() {
    setError(null);
    setState("loading");
    try {
      const src = await speakUrl(speakable(text));
      const a = new Audio(src);
      audio.current = a;
      a.onended = a.onpause = () => {
        setState("idle");
        URL.revokeObjectURL(src);
      };
      await a.play();
      setState("playing");
    } catch (e) {
      setError((e as Error).message);
      setState("idle");
    }
  }

  return (
    <>
      <Button size="sm" variant="ghost" onClick={() => (state === "playing" ? audio.current?.pause() : state === "idle" && void play())} title={error ?? undefined}>
        {state === "loading" ? <Loader2 className="size-3.5 animate-spin" /> : state === "playing" ? <Square className="size-3 fill-current" /> : <Volume2 className="size-3.5" />}
        {state === "playing" ? "Stop" : "Read aloud"}
      </Button>
      {error && <span className="self-center text-[12px] text-danger">{error}</span>}
    </>
  );
}
