import { Download, FolderOpen, Loader2, Play, Save, Square, Trash2, Volume2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { speakUrl, useTts } from "../components/ReadAloud";
import { Button, EmptyState, PageHeader, ProgressBar, Select } from "../components/ui";
import { api, apiBlob, inTauri } from "../lib/engine";
import { bytes, relativeDate } from "../lib/format";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";

interface Voice {
  id: string;
  name: string;
  language: string;
  lang: string;
  gender: string;
}

interface Clip {
  id: string;
  text: string;
  voice: string;
  speed: number;
  duration: number | null;
  created: number;
  path: string;
}

// What a voice's preview says, in its language.
const SAMPLE: Record<string, string> = {
  "en-us": "Hello! This is how I sound reading your text aloud.",
  "en-gb": "Hello! This is how I sound reading your text aloud.",
  es: "¡Hola! Así sueno cuando leo tu texto en voz alta.",
  "fr-fr": "Bonjour ! Voici ma voix quand je lis votre texte à voix haute.",
  hi: "नमस्ते! जब मैं आपका पाठ पढ़ती हूँ तो मेरी आवाज़ ऐसी लगती है।",
  it: "Ciao! Ecco come suono quando leggo il tuo testo ad alta voce.",
  ja: "こんにちは。これがあなたの文章を読み上げる私の声です。",
  "pt-br": "Olá! É assim que eu soo lendo o seu texto em voz alta.",
  cmn: "你好！这就是我朗读你的文字时的声音。",
};

/** One <audio> at a time across the page: previews, the composer and the gallery. */
function usePlayer() {
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const stop = useCallback(() => {
    audio.current?.pause();
    audio.current = null;
    setPlaying(null);
  }, []);
  const play = useCallback(
    async (key: string, src: string) => {
      stop();
      const a = new Audio(src);
      audio.current = a;
      a.onended = () => setPlaying((p) => (p === key ? null : p));
      setPlaying(key);
      await a.play();
    },
    [stop],
  );
  useEffect(() => stop, [stop]);
  return { playing, play, stop };
}

/** Text to speech with Kokoro (engine tts.py): a voice library, reading text aloud, and saved clips. */
export function VoicePage() {
  const { tts, refresh } = useTts();
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title="Voice" description="Read text aloud in 54 voices and nine languages, on this computer. Chat replies get a Read aloud button too." />
      <div className="min-h-0 flex-1 overflow-y-auto px-7 py-6">{!tts ? null : tts.ready ? <Studio /> : <Setup installed={tts.installed} size={tts.size} repo={tts.repo} onDone={refresh} />}</div>
    </div>
  );
}

function Setup({ installed, size, repo, onDone }: { installed: boolean; size: number; repo: string; onDone: () => void }) {
  const { refreshJobs } = useApp();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;

  useEffect(() => {
    if (!status || status === "running") return;
    if (status === "done") onDone();
    else setError((events.find((e) => e.event === "status")?.error as string) ?? `It ended ${status}.`);
    setJobId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function start() {
    setError(null);
    try {
      const job = installed ? await api<Job>("/api/models/download", { json: { repo_id: repo } }) : await api<Job>("/api/tts/install", { method: "POST" });
      setJobId(job.id);
      void refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="max-w-[560px]">
      <EmptyState icon={<Volume2 className="size-8" />} title={installed ? "Download Kokoro's voices" : "Set up text to speech"}>
        <p className="text-[13px] text-muted">
          {installed
            ? `One download, ${bytes(size)}: the Kokoro model and all its voices.`
            : "Kokoro is a small, natural-sounding speech model. It installs into its own environment and runs on the CPU, on any computer."}
        </p>
        <div className="mt-4 flex flex-col items-center gap-2">
          <Button variant="primary" onClick={() => void start()} disabled={running}>
            <Download className="size-3.5" /> {installed ? "Download voices" : "Install"}
          </Button>
          {running && <ProgressBar value={progress ?? null} />}
          {error && <p className="text-[12px] text-danger">{error}</p>}
        </div>
      </EmptyState>
    </div>
  );
}

function Studio() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voice, setVoice] = useState("");
  const [language, setLanguage] = useState("");
  const [speed, setSpeed] = useState(1);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState<"read" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const player = usePlayer();

  const loadClips = useCallback(() => void api<{ clips: Clip[] }>("/api/tts/gallery").then((r) => setClips(r.clips), () => {}), []);
  useEffect(() => {
    void api<{ voices: Voice[]; voice: string }>("/api/tts/voices").then(
      (r) => {
        setVoices(r.voices);
        setVoice(r.voice);
        setLanguage(r.voices.find((v) => v.id === r.voice)?.language ?? "");
      },
      (e: Error) => setError(e.message),
    );
    loadClips();
  }, [loadClips]);

  const languages = useMemo(() => [...new Set(voices.map((v) => v.language))], [voices]);
  const shown = voices.filter((v) => !language || v.language === language);

  async function pick(id: string) {
    setVoice(id);
    await api("/api/tts/voice", { method: "PUT", json: { voice: id } }).catch(() => {}); // Chat's Read aloud uses it too
  }

  async function preview(v: Voice) {
    if (player.playing === `voice:${v.id}`) return player.stop();
    setError(null);
    try {
      await player.play(`voice:${v.id}`, await speakUrl(SAMPLE[v.lang] ?? SAMPLE["en-us"], v.id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function read() {
    setBusy("read");
    setError(null);
    try {
      await player.play("text", await speakUrl(text, voice, speed));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function save() {
    setBusy("save");
    setError(null);
    try {
      await api("/api/tts/speak", { json: { text, voice, speed, save: true } });
      loadClips();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function playClip(c: Clip) {
    if (player.playing === `clip:${c.id}`) return player.stop();
    await player.play(`clip:${c.id}`, URL.createObjectURL(await apiBlob(`/api/tts/gallery/${c.id}/audio`)));
  }

  return (
    <div className="grid max-w-[1100px] gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="min-w-0">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={7}
          placeholder="Type or paste what to read aloud"
          aria-label="Text to read"
          className="w-full resize-y rounded-lg border border-line bg-surface px-3 py-2 text-[14px] outline-none focus:border-accent"
        />
        <div className="mt-2 flex flex-wrap items-center gap-3 text-[13px]">
          <label className="flex items-center gap-2 text-muted">
            Speed
            <input type="range" min={0.5} max={2} step={0.1} value={speed} onChange={(e) => setSpeed(+e.target.value)} aria-label="Speed" />
            <span className="w-8 tabular-nums text-text">{speed.toFixed(1)}×</span>
          </label>
          <span className="flex-1" />
          {player.playing === "text" ? (
            <Button onClick={player.stop}>
              <Square className="size-3 fill-current" /> Stop
            </Button>
          ) : (
            <Button onClick={() => void read()} disabled={!text.trim() || !!busy}>
              {busy === "read" ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />} Read aloud
            </Button>
          )}
          <Button variant="primary" onClick={() => void save()} disabled={!text.trim() || !!busy}>
            {busy === "save" ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />} Save to gallery
          </Button>
        </div>
        {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}

        <h2 className="mt-8 mb-2 text-[13px] font-semibold">Gallery</h2>
        {clips.length === 0 ? (
          <p className="text-[13px] text-muted">Clips you save appear here, as WAV files in Temper's folder.</p>
        ) : (
          <ul className="divide-y divide-line rounded-lg border border-line bg-surface">
            {clips.map((c) => (
              <li key={c.id} className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
                <button className="text-muted hover:text-text" onClick={() => void playClip(c)} aria-label={player.playing === `clip:${c.id}` ? "Stop" : "Play"}>
                  {player.playing === `clip:${c.id}` ? <Square className="size-3.5 fill-current" /> : <Play className="size-4" />}
                </button>
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-2">{c.text}</span>
                  <span className="block text-[12px] text-muted">
                    {voices.find((v) => v.id === c.voice)?.name ?? c.voice} · {c.duration ? `${c.duration.toFixed(1)} s` : ""} · {relativeDate(c.created)}
                  </span>
                </span>
                {inTauri && (
                  <button
                    className="text-muted hover:text-text"
                    title="Show in folder"
                    aria-label="Show in folder"
                    onClick={async () => {
                      const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
                      await revealItemInDir(c.path);
                    }}
                  >
                    <FolderOpen className="size-3.5" />
                  </button>
                )}
                <button className="text-muted hover:text-danger" title="Delete" aria-label="Delete clip" onClick={() => void api<{ clips: Clip[] }>(`/api/tts/gallery/${c.id}`, { method: "DELETE" }).then((r) => setClips(r.clips))}>
                  <Trash2 className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <aside>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-[13px] font-semibold">Voices</h2>
          <Select value={language} onChange={(e) => setLanguage(e.target.value)} aria-label="Language" className="w-44">
            <option value="">All languages</option>
            {languages.map((l) => (
              <option key={l}>{l}</option>
            ))}
          </Select>
        </div>
        <ul className="max-h-[560px] divide-y divide-line overflow-y-auto rounded-lg border border-line bg-surface">
          {shown.map((v) => (
            <li key={v.id} className="flex items-center gap-2 px-3 py-2 text-[13px]">
              <label className="flex min-w-0 flex-1 items-center gap-2">
                <input type="radio" name="voice" checked={voice === v.id} onChange={() => void pick(v.id)} />
                <span className="font-medium">{v.name}</span>
                <span className="truncate text-[12px] text-muted">
                  {v.gender} · {v.language}
                </span>
              </label>
              <button className="text-muted hover:text-text" onClick={() => void preview(v)} aria-label={`Hear ${v.name}`} title={`Hear ${v.name}`}>
                {player.playing === `voice:${v.id}` ? <Square className="size-3 fill-current" /> : <Volume2 className="size-3.5" />}
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
