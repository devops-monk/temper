import { KeyRound, Trash2 } from "lucide-react";
import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../lib/engine";
import { Button, CopyButton, Input, Notice, Toggle, cx } from "./ui";

/** An API key as the engine lists it (api_keys.py): the whole key is only in the reply that makes it. */
interface ApiKey {
  id: string;
  name: string;
  hint: string;
  created: number;
  last_used: number | null;
  requests: number;
  input_tokens: number;
  output_tokens: number;
}

interface LogEntry {
  id: string;
  time: number;
  method: string;
  path: string;
  key: string;
  status: number | null;
  ms: number | null;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  tokens_per_second: number | null;
  first_ms: number | null;
  done: boolean;
  // Only while the monitor asks for them (engine api_keys.py).
  prompt?: string;
  reply?: string;
}

function ago(t: number | null): string {
  if (!t) return "never";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
}

const count = (n: number) => (n >= 10_000 ? `${(n / 1000).toFixed(n >= 100_000 ? 0 : 1)}K` : String(n));

/** Named keys for /v1, made, copied once and revoked here. */
export function ApiKeys() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [name, setName] = useState("");
  const [made, setMade] = useState<{ name: string; key: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => void api<ApiKey[]>("/api/keys").then(setKeys, () => {}), []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  async function create() {
    setError(null);
    try {
      const k = await api<ApiKey & { key: string }>("/api/keys", { json: { name: name.trim() } });
      setMade({ name: k.name, key: k.key });
      setName("");
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function revoke(k: ApiKey) {
    if (!confirm(`Revoke "${k.name}"? Apps using it will stop working.`)) return;
    await api(`/api/keys/${k.id}`, { method: "DELETE" });
    if (made?.name === k.name) setMade(null);
    load();
  }

  return (
    <div>
      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) void create();
        }}
      >
        <span className="w-64">
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name it after the app or device" aria-label="New key's name" maxLength={60} />
        </span>
        <Button type="submit" disabled={!name.trim()}>
          <KeyRound className="size-3.5" /> Create key
        </Button>
        {error && <span className="text-[12px] text-danger">{error}</span>}
      </form>

      {made && (
        <div className="mt-3">
          <Notice tone="info">
            <div className="flex flex-wrap items-center gap-2">
              <span>Copy the key for {made.name} now. Temper keeps only a fingerprint of it, so it can't show it again.</span>
              <span className="font-mono text-[12.5px] break-all text-text">{made.key}</span>
              <CopyButton text={made.key} />
            </div>
          </Notice>
        </div>
      )}

      {keys.length > 0 && (
        <table className="mt-3 w-full text-[13px]">
          <thead className="text-left text-[12px] text-muted">
            <tr>
              <th className="py-1.5 font-medium">Name</th>
              <th className="font-medium">Key</th>
              <th className="font-medium">Last used</th>
              <th className="text-right font-medium">Requests</th>
              <th className="text-right font-medium">Tokens in / out</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k.id} className="border-t border-line">
                <td className="py-2">{k.name}</td>
                <td className="font-mono text-[12px] text-muted">{k.hint}</td>
                <td className="text-muted">{ago(k.last_used)}</td>
                <td className="text-right tabular-nums">{count(k.requests)}</td>
                <td className="text-right tabular-nums">
                  {count(k.input_tokens)} / {count(k.output_tokens)}
                </td>
                <td className="w-8 text-right">
                  <button className="text-muted hover:text-danger" onClick={() => void revoke(k)} title="Revoke" aria-label={`Revoke ${k.name}`}>
                    <Trash2 className="size-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** The last requests to /v1, from any key or Temper itself, refreshed every few seconds. */
export function ApiMonitor() {
  const [log, setLog] = useState<LogEntry[]>([]);
  const [content, setContent] = useState(false);

  useEffect(() => {
    // With content on, each poll keeps the engine recording prompts and replies; off, it stops and drops them.
    const load = () => void api<LogEntry[]>(`/api/keys/log?limit=50&content=${content}`).then(setLog, () => {});
    load();
    const t = setInterval(load, content ? 1500 : 3000);
    return () => clearInterval(t);
  }, [content]);

  return (
    <div>
      <div className="mb-2 max-w-sm">
        <Toggle checked={content} onChange={setContent} label="Show prompts and replies" />
      </div>
      {content && <p className="mb-2 text-[12px] text-muted">Recorded from now on while this is open, in memory only. Turning it off or leaving the page forgets them.</p>}
      {!log.length ? (
        <p className="text-[13px] text-muted">No requests yet. They show up here as apps use the API.</p>
      ) : (
        <div className="max-h-96 overflow-y-auto rounded-lg border border-line">
          <table className="w-full text-[12.5px]">
            <thead className="sticky top-0 bg-surface text-left text-[12px] text-muted">
              <tr>
                <th className="px-3 py-1.5 font-medium">Time</th>
                <th className="font-medium">From</th>
                <th className="font-medium">Request</th>
                <th className="font-medium">Model</th>
                <th className="text-right font-medium">Tokens</th>
                <th className="text-right font-medium">Speed</th>
                <th className="px-3 text-right font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {log.map((e) => (
                <Fragment key={e.id}>
                  <tr className="border-t border-line">
                    <td className="px-3 py-1.5 text-muted tabular-nums">{new Date(e.time * 1000).toLocaleTimeString()}</td>
                    <td>{e.key}</td>
                    <td className="font-mono text-[12px]">
                      {e.method} {e.path}
                    </td>
                    <td className="max-w-48 truncate text-muted" title={e.model ?? ""}>
                      {e.model ?? ""}
                    </td>
                    <td className="text-right tabular-nums">{e.input_tokens != null ? `${e.input_tokens} → ${e.output_tokens ?? 0}` : ""}</td>
                    <td className="text-right text-muted tabular-nums" title={e.first_ms != null ? `First token after ${(e.first_ms / 1000).toFixed(1)} s` : undefined}>
                      {e.tokens_per_second != null ? `${e.tokens_per_second}/s` : ""}
                    </td>
                    <td className={cx("px-3 text-right tabular-nums", e.status == null ? "text-muted" : e.status >= 400 ? "text-danger" : "")}>
                      {e.status == null || !e.done ? "…" : `${e.status}`}
                      {e.ms != null && <span className="ml-1.5 text-muted">{e.ms < 1000 ? `${e.ms} ms` : `${(e.ms / 1000).toFixed(1)} s`}</span>}
                    </td>
                  </tr>
                  {content && (e.prompt || e.reply) && (
                    <tr>
                      <td colSpan={7} className="px-3 pb-2 text-[12.5px]">
                        {e.prompt && <p className="line-clamp-3 whitespace-pre-wrap text-muted">{e.prompt.trim()}</p>}
                        {e.reply && <p className="mt-1 line-clamp-6 whitespace-pre-wrap">{e.reply.trim()}</p>}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
