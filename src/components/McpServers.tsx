import { ChevronRight, Plug, RotateCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT } from "../lib/i18n";
import { Badge, Button, Field, Input, Section, Select, Toggle, cx } from "./ui";

/** An MCP server as the engine lists it (engine mcp.py); env and header values come back masked. */
export interface McpServer {
  name: string;
  transport: "stdio" | "http" | "sse";
  config: { command?: string; args?: string[]; env?: Record<string, string>; cwd?: string; url?: string; type?: string; headers?: Record<string, string> };
  enabled: boolean;
  state: "connected" | "idle" | "error" | "off";
  error: string | null;
  tools: { name: string; description: string; mode: "off" | "ask" | "auto" }[];
  server: { name?: string; version?: string } | null;
}

/** KEY=value or Name: value lines, as a map. */
const pairs = (text: string, sep: "=" | ":") =>
  Object.fromEntries(
    text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.includes(sep))
      .map((line) => [line.slice(0, line.indexOf(sep)).trim(), line.slice(line.indexOf(sep) + 1).trim()]),
  );

const textarea = "w-full resize-y rounded-md border border-line bg-surface px-3 py-2 font-mono text-[12.5px] placeholder:text-muted/70 focus:border-accent focus:outline-none";

/** Settings: the MCP servers chat models can use, each with its tools and their approval. */
export function McpServers() {
  const t = useT();
  const [servers, setServers] = useState<McpServer[] | null>(null);
  const [adding, setAdding] = useState<"form" | "paste" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = () => void api<{ servers: McpServer[] }>("/api/mcp").then((r) => setServers(r.servers), () => {});
  useEffect(refresh, []);
  const replace = (s: McpServer) => setServers((list) => list?.map((x) => (x.name === s.name ? s : x)) ?? null);

  return (
    <Section
      title={t("mcp.title")}
      aside={
        !adding && (
          <span className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => setAdding("paste")}>
              {t("mcp.paste")}
            </Button>
            <Button size="sm" onClick={() => setAdding("form")}>
              {t("mcp.add")}
            </Button>
          </span>
        )
      }
    >
      <p className="mb-3 text-[12.5px] text-muted">{t("mcp.about")}</p>
      {adding === "form" && (
        <AddServer
          onDone={(s) => {
            setAdding(null);
            if (s) setServers((list) => [...(list ?? []).filter((x) => x.name !== s.name), s]);
          }}
        />
      )}
      {adding === "paste" && (
        <PasteServers
          onDone={(list) => {
            setAdding(null);
            if (list) setServers(list);
          }}
        />
      )}
      {servers?.length === 0 && !adding && <p className="text-[13px] text-muted">{t("mcp.empty")}</p>}
      <ul className="flex flex-col gap-2">
        {servers?.map((s) => (
          <ServerRow
            key={s.name}
            server={s}
            onChange={replace}
            onRemove={async () => {
              if (!confirm(t("mcp.remove_confirm", { name: s.name }))) return;
              setError(null);
              await api<{ servers: McpServer[] }>(`/api/mcp/${encodeURIComponent(s.name)}`, { method: "DELETE" }).then((r) => setServers(r.servers), (e: Error) => setError(e.message));
            }}
          />
        ))}
      </ul>
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}
    </Section>
  );
}

function ServerRow({ server: s, onChange, onRemove }: { server: McpServer; onChange: (s: McpServer) => void; onRemove: () => void }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const path = `/api/mcp/${encodeURIComponent(s.name)}`;
  const where = s.transport === "stdio" ? [s.config.command, ...(s.config.args ?? [])].join(" ") : s.config.url;
  const state = { connected: ["ok", t("mcp.connected", { count: s.tools.length })], idle: ["neutral", t("mcp.idle")], error: ["warn", t("mcp.error")], off: ["neutral", t("mcp.off")] } as const;

  async function connect() {
    setBusy(true);
    await api<McpServer>(`${path}/connect`, { method: "POST" }).then(onChange, () => {});
    setBusy(false);
  }

  return (
    <li className="rounded-lg border border-line bg-surface">
      <div className="flex items-center gap-2 px-3 py-2">
        <button onClick={() => setOpen(!open)} aria-expanded={open} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          <ChevronRight className={cx("size-3.5 shrink-0 text-muted transition-transform", open && "rotate-90")} />
          <Plug className="size-3.5 shrink-0 text-muted" />
          <span className="font-medium">{s.name}</span>
          <Badge tone={state[s.state][0]}>{state[s.state][1]}</Badge>
          <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-muted">{where}</span>
        </button>
        <Button size="sm" variant="ghost" onClick={() => void connect()} disabled={busy || !s.enabled} title={t("mcp.reconnect")} aria-label={t("mcp.reconnect")}>
          <RotateCw className={cx("size-3.5", busy && "animate-spin")} />
        </Button>
        <Button size="sm" variant="ghost" onClick={onRemove} title={t("mcp.remove")} aria-label={t("mcp.remove")}>
          <Trash2 className="size-3.5" />
        </Button>
      </div>
      {s.error && s.state === "error" && <p className="border-t border-line px-3 py-2 text-[12px] text-danger">{s.error}</p>}
      {open && (
        <div className="flex flex-col gap-2 border-t border-line px-3 py-2">
          <Toggle label={t("mcp.enabled")} checked={s.enabled} onChange={(enabled) => void api<McpServer>(path, { method: "PUT", json: { enabled } }).then(onChange, () => {})} />
          {s.tools.length === 0 && s.enabled && (
            <p className="text-[12px] text-muted">
              {s.state === "connected" ? t("mcp.no_tools") : t("mcp.connect_hint")}{" "}
              {s.state !== "connected" && (
                <button className="text-accent hover:underline" onClick={() => void connect()}>
                  {t("mcp.connect")}
                </button>
              )}
            </p>
          )}
          {s.tools.map((tool) => (
            <div key={tool.name} className="flex items-start justify-between gap-3">
              <span className="min-w-0">
                <span className="block font-mono text-[12.5px]">{tool.name}</span>
                <span className="block text-[12px] text-muted">{tool.description}</span>
              </span>
              <Select
                value={tool.mode}
                onChange={(e) => void api<McpServer>(path, { method: "PUT", json: { tools: { [tool.name]: e.target.value } } }).then(onChange, () => {})}
                aria-label={tool.name}
                className="h-7 w-24 shrink-0 text-[12.5px]"
              >
                <option value="off">{t("tools.off")}</option>
                <option value="ask">{t("tools.ask")}</option>
                <option value="auto">{t("tools.auto")}</option>
              </Select>
            </div>
          ))}
        </div>
      )}
    </li>
  );
}

function AddServer({ onDone }: { onDone: (s: McpServer | null) => void }) {
  const t = useT();
  const [name, setName] = useState("");
  const [kind, setKind] = useState<"stdio" | "http" | "sse">("stdio");
  const [command, setCommand] = useState("");
  const [env, setEnv] = useState("");
  const [url, setUrl] = useState("");
  const [headers, setHeaders] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSaving(true);
    const body = kind === "stdio" ? { name, command, env: pairs(env, "=") } : { name, url, type: kind, headers: pairs(headers, ":") };
    try {
      onDone(await api<McpServer>("/api/mcp", { json: body }));
    } catch (err) {
      setError((err as Error).message);
    }
    setSaving(false);
  }

  return (
    <form onSubmit={(e) => void save(e)} className="mb-3 flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
      <div className="grid grid-cols-[1fr_auto] gap-3">
        <Field label={t("mcp.name")}>
          <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="github" maxLength={64} />
        </Field>
        <Field label={t("mcp.transport")}>
          <Select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)} className="w-40">
            <option value="stdio">{t("mcp.stdio")}</option>
            <option value="http">{t("mcp.http")}</option>
            <option value="sse">{t("mcp.sse")}</option>
          </Select>
        </Field>
      </div>
      {kind === "stdio" ? (
        <>
          <Field label={t("mcp.command")} hint={t("mcp.command_hint")}>
            <Input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx -y @modelcontextprotocol/server-filesystem ~/Documents" className="font-mono text-[12.5px]" />
          </Field>
          <Field label={t("mcp.env")} hint={t("mcp.env_hint")}>
            <textarea value={env} onChange={(e) => setEnv(e.target.value)} rows={2} placeholder="API_KEY=…" className={textarea} />
          </Field>
        </>
      ) : (
        <>
          <Field label="URL">
            <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://example.com/mcp" className="font-mono text-[12.5px]" />
          </Field>
          <Field label={t("mcp.headers")} hint={t("mcp.headers_hint")}>
            <textarea value={headers} onChange={(e) => setHeaders(e.target.value)} rows={2} placeholder="Authorization: Bearer …" className={textarea} />
          </Field>
        </>
      )}
      {error && <p className="text-[12px] text-danger">{error}</p>}
      <div className="flex gap-2">
        <Button size="sm" variant="primary" type="submit" disabled={saving || !name.trim() || !(kind === "stdio" ? command.trim() : url.trim())}>
          {saving ? t("mcp.connecting") : t("mcp.save")}
        </Button>
        <Button size="sm" variant="ghost" type="button" onClick={() => onDone(null)}>
          {t("prompts.cancel")}
        </Button>
      </div>
    </form>
  );
}

function PasteServers({ onDone }: { onDone: (list: McpServer[] | null) => void }) {
  const t = useT();
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  async function save() {
    setError(null);
    try {
      onDone((await api<{ servers: McpServer[] }>("/api/mcp/import", { json: { text } })).servers);
    } catch (err) {
      setError((err as Error).message);
    }
  }
  return (
    <div className="mb-3 flex flex-col gap-2 rounded-lg border border-line bg-surface p-3">
      <p className="text-[12.5px] text-muted">{t("mcp.paste_hint")}</p>
      <textarea autoFocus value={text} onChange={(e) => setText(e.target.value)} rows={8} placeholder={'{\n  "mcpServers": {\n    "time": { "command": "uvx", "args": ["mcp-server-time"] }\n  }\n}'} className={textarea} aria-label={t("mcp.paste")} />
      {error && <p className="text-[12px] text-danger">{error}</p>}
      <div className="flex gap-2">
        <Button size="sm" variant="primary" onClick={() => void save()} disabled={!text.trim()}>
          {t("mcp.import")}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onDone(null)}>
          {t("prompts.cancel")}
        </Button>
      </div>
    </div>
  );
}
