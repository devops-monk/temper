import { Loader2, Mic, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/engine";
import { Button, cx } from "./ui";

/** Speech to text as the engine reports it (engine speech.py). */
export interface SpeechStatus {
  backend: string;
  installed: boolean;
  model: string;
  ready: boolean;
  models: { id: string; name: string; size: number; about: string; downloaded: boolean }[];
  running: boolean;
}

let statusCache: Promise<SpeechStatus | null> | null = null;

/** Whether dictation can run, asked once per page load; `refresh` after installing or downloading. */
export function useSpeech(): { speech: SpeechStatus | null; refresh: () => void } {
  const [speech, setSpeech] = useState<SpeechStatus | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    statusCache ??= api<SpeechStatus>("/api/speech").catch(() => null);
    let stale = false;
    void statusCache.then((s) => !stale && setSpeech(s));
    return () => {
      stale = true;
    };
  }, [tick]);
  return {
    speech,
    refresh: () => {
      statusCache = null;
      setTick((t) => t + 1);
    },
  };
}

/** The microphone button in Chat's composer: click to record, click again to put what was said into the message. */
export function Dictate({ onText, disabled }: { onText: (text: string) => void; disabled?: boolean }) {
  const { speech } = useSpeech();
  const [state, setState] = useState<"idle" | "recording" | "working">("idle");
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);

  useEffect(() => () => recorder.current?.stream.getTracks().forEach((t) => t.stop()), []);
  if (!speech?.ready) return null;

  async function start() {
    setError(null);
    let media: MediaStream;
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Temper can't use the microphone. Allow it in System Settings → Privacy & Security → Microphone.");
      return;
    }
    void api("/api/speech/warm", { method: "POST" }).catch(() => {}); // loads the model while you speak
    const rec = new MediaRecorder(media);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      media.getTracks().forEach((t) => t.stop());
      const type = rec.mimeType || "audio/webm";
      const suffix = type.includes("mp4") ? ".m4a" : type.includes("ogg") ? ".ogg" : ".webm";
      setState("working");
      try {
        const r = await api<{ text: string }>(`/api/speech/transcribe?suffix=${suffix}`, { method: "POST", body: new Blob(chunks, { type }), headers: { "Content-Type": type } });
        if (r.text.trim()) onText(r.text.trim());
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setState("idle");
      }
    };
    recorder.current = rec;
    rec.start();
    setState("recording");
  }

  return (
    <span className="relative">
      <Button
        variant="ghost"
        onClick={() => (state === "recording" ? recorder.current?.stop() : state === "idle" && void start())}
        disabled={disabled || state === "working"}
        aria-label={state === "recording" ? "Stop and transcribe" : "Dictate"}
        title={state === "recording" ? "Stop and put it in the message" : `Dictate (${speech.backend}, on this computer)`}
        className={cx("size-9 rounded-xl px-0 text-muted", state === "recording" && "text-danger")}
      >
        {state === "working" ? <Loader2 className="size-[18px] animate-spin" /> : state === "recording" ? <Square className="size-3.5 animate-pulse fill-current" /> : <Mic className="size-[18px]" />}
      </Button>
      {error && (
        <span role="alert" className="absolute bottom-11 left-0 z-10 w-72 rounded-lg border border-line bg-surface px-3 py-2 text-[12px] text-danger shadow" onClick={() => setError(null)}>
          {error}
        </span>
      )}
    </span>
  );
}
