import { AudioLines, Download, FolderOpen, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api, inTauri } from "../lib/engine";
import { bytes } from "../lib/format";
import { useApp } from "../lib/store";
import type { Job } from "../lib/types";
import { useJob } from "../lib/useJob";
import { useSpeech } from "./Dictate";
import { Badge, Button, CopyButton, Input, ProgressBar, Section } from "./ui";

const AUDIO = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "webm", "aiff", "mp4", "mov", "mkv"];

/** Settings → Speech to text (engine speech.py): installing Whisper, its model, and transcribing a file. */
export function SpeechSettings() {
  const { speech, refresh } = useSpeech();
  const { jobs, refreshJobs } = useApp();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, events } = useJob(jobId);
  const running = jobId !== null && (status === null || status === "running");
  const progress = [...events].reverse().find((e) => e.event === "progress")?.value as number | undefined;
  const lastLine = [...events].reverse().find((e) => e.event === "log")?.line as string | undefined;
  // A model download started here or from Models, which the list waits on.
  const downloading = jobs.filter((j) => j.kind === "download" && j.status === "running").map((j) => j.title);

  useEffect(() => {
    if (status && status !== "running") {
      refresh();
      void refreshJobs();
      if (status === "failed") setError((events.find((e) => e.event === "status")?.error as string) ?? "It failed; the log says why.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);
  useEffect(() => {
    if (!downloading.length) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [downloading.length]);

  async function run(path: string, init: Parameters<typeof api>[1] = {}) {
    setError(null);
    try {
      const job = await api<Job>(path, { method: "POST", ...init });
      if (job?.id) {
        setJobId(job.id);
        void refreshJobs();
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function download(repo: string) {
    setError(null);
    await api("/api/models/download", { json: { repo_id: repo } }).catch((e: Error) => setError(e.message));
    void refreshJobs(); // the list shows it downloading, and the Models page its progress
  }

  async function choose(model: string) {
    await api("/api/speech/model", { method: "PUT", json: { model } }).catch((e: Error) => setError(e.message));
    refresh();
  }

  async function remove() {
    if (!confirm("Remove speech to text? Its environment is deleted; downloaded models stay.")) return;
    await api("/api/speech", { method: "DELETE" });
    refresh();
  }

  if (!speech) return null;
  return (
    <Section title="Speech to text">
      <div className="flex items-center gap-2 text-[13px]">
        <span className="font-medium">{speech.backend}</span>
        {speech.installed ? <Badge tone="ok">Installed</Badge> : <Badge>Not installed</Badge>}
        <span className="flex-1" />
        {speech.installed ? (
          <button className="text-muted hover:text-danger" onClick={() => void remove()} title="Remove speech to text" aria-label="Remove speech to text">
            <Trash2 className="size-3.5" />
          </button>
        ) : (
          <Button size="sm" onClick={() => void run("/api/speech/install")} disabled={running}>
            <Download className="size-3.5" /> Install
          </Button>
        )}
      </div>
      <p className="mt-1 text-[12px] text-muted">
        Dictate in Chat with the microphone button, and transcribe recordings into text and subtitles. Whisper runs on this computer; nothing is sent anywhere. Apps can use it at{" "}
        <span className="font-mono">/v1/audio/transcriptions</span>.
      </p>

      {speech.installed && (
        <ul className="mt-3 divide-y divide-line rounded-lg border border-line bg-surface">
          {speech.models.map((m) => {
            const fetching = downloading.some((title) => title.startsWith(m.id));
            return (
              <li key={m.id} className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
                <label className="flex min-w-0 flex-1 items-center gap-2.5">
                  <input type="radio" name="speech-model" checked={speech.model === m.id} disabled={!m.downloaded} onChange={() => void choose(m.id)} />
                  <span className="min-w-0">
                    <span className="font-medium">{m.name}</span> <span className="text-[12px] text-muted">{bytes(m.size)}</span>
                    <span className="block text-[12px] text-muted">{m.about}</span>
                  </span>
                </label>
                {!m.downloaded &&
                  (fetching ? (
                    <span className="text-[12px] text-muted">Downloading…</span>
                  ) : (
                    <Button size="sm" variant="ghost" onClick={() => void download(m.id)}>
                      <Download className="size-3.5" /> Download
                    </Button>
                  ))}
              </li>
            );
          })}
        </ul>
      )}

      {speech.ready && <TranscribeFile onStart={(id) => setJobId(id)} busy={running} />}

      {running && (
        <div className="mt-3 flex flex-col gap-1.5">
          <ProgressBar value={progress ?? null} />
          <p className="truncate font-mono text-[11.5px] text-muted" title={lastLine}>
            {lastLine ?? "Starting…"}
          </p>
        </div>
      )}
      {status === "done" && <Transcript events={events} />}
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}
    </Section>
  );
}

function TranscribeFile({ onStart, busy }: { onStart: (id: string) => void; busy: boolean }) {
  const [path, setPath] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function start(file: string) {
    setError(null);
    try {
      onStart((await api<Job>("/api/speech/jobs", { json: { path: file } })).id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function pick() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const chosen = await open({ filters: [{ name: "Audio and video", extensions: AUDIO }] });
    if (typeof chosen === "string") void start(chosen);
  }

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px]">
      {inTauri ? (
        <Button size="sm" onClick={() => void pick()} disabled={busy}>
          <AudioLines className="size-3.5" /> Transcribe a file…
        </Button>
      ) : (
        <form
          className="flex flex-1 items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (path.trim()) void start(path.trim());
          }}
        >
          <Input value={path} onChange={(e) => setPath(e.target.value)} placeholder="Path to an audio or video file" aria-label="File to transcribe" />
          <Button size="sm" type="submit" disabled={busy || !path.trim()}>
            <AudioLines className="size-3.5" /> Transcribe
          </Button>
        </form>
      )}
      {error && <span className="text-[12px] text-danger">{error}</span>}
    </div>
  );
}

function Transcript({ events }: { events: { event: string; [k: string]: unknown }[] }) {
  const result = events.find((e) => e.event === "status")?.result as { text?: string; files?: Record<string, string>; language?: string } | undefined;
  if (!result?.files) return null;
  const { files } = result;
  return (
    <div className="mt-3 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
      <p className="max-h-40 overflow-y-auto whitespace-pre-wrap">{result.text}</p>
      <div className="mt-2 flex items-center gap-2 text-[12px] text-muted">
        <CopyButton text={result.text ?? ""} />
        <span className="flex-1">Saved as text, SRT and WebVTT subtitles, and JSON{result.language ? ` (${result.language})` : ""}.</span>
        {inTauri && (
          <Button
            size="sm"
            variant="ghost"
            onClick={async () => {
              const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
              await revealItemInDir(files.txt);
            }}
          >
            <FolderOpen className="size-3.5" /> Show
          </Button>
        )}
      </div>
    </div>
  );
}
