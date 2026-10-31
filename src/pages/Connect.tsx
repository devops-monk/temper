import { useState } from "react";
import { ApiKeys, ApiMonitor } from "../components/ApiKeys";
import { DiscoverySettings } from "../components/Discovery";
import { LanAccess } from "../components/LanAccess";
import { CopyButton, Notice, PageHeader, cx } from "../components/ui";
import { useApp } from "../lib/store";

interface Recipe {
  id: string;
  name: string;
  where: string;
  body: (url: string, token: string, model: string) => string;
}

const RECIPES: Recipe[] = [
  {
    id: "claude",
    name: "Claude Code",
    where: "terminal",
    body: (url, token, model) => `export ANTHROPIC_BASE_URL=${url}
export ANTHROPIC_AUTH_TOKEN=${token}
export ANTHROPIC_MODEL=${model}
export ANTHROPIC_DEFAULT_HAIKU_MODEL=${model}
claude`,
  },
  {
    id: "codex",
    name: "Codex CLI",
    where: "~/.codex/config.toml",
    body: (url, _t, model) => `model = "${model}"
model_provider = "temper"

[model_providers.temper]
name = "Temper"
base_url = "${url}/v1"
env_key = "TEMPER_API_KEY"
wire_api = "chat"

# then run:  TEMPER_API_KEY=<token> codex`,
  },
  {
    id: "opencode",
    name: "OpenCode",
    where: "opencode.json",
    body: (url, token, model) => `{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "temper": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Temper",
      "options": { "baseURL": "${url}/v1", "apiKey": "${token}" },
      "models": { "${model}": { "name": "${model}" } }
    }
  }
}`,
  },
  {
    id: "continue",
    name: "VS Code (Continue)",
    where: "~/.continue/config.yaml",
    body: (url, token, model) => `models:
  - name: Temper
    provider: openai
    model: ${model}
    apiBase: ${url}/v1
    apiKey: ${token}
    roles: [chat, edit, apply]`,
  },
  {
    id: "aider",
    name: "Aider",
    where: "terminal",
    body: (url, token, model) => `export OPENAI_API_BASE=${url}/v1
export OPENAI_API_KEY=${token}
aider --model openai/${model}`,
  },
  {
    id: "python",
    name: "Python",
    where: "openai SDK",
    body: (url, token, model) => `from openai import OpenAI

client = OpenAI(base_url="${url}/v1", api_key="${token}")
reply = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)`,
  },
  {
    id: "anthropic",
    name: "Anthropic SDK",
    where: "Python",
    body: (url, token, model) => `from anthropic import Anthropic

client = Anthropic(base_url="${url}", api_key="${token}")
reply = client.messages.create(
    model="${model}",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.content[0].text)`,
  },
  {
    id: "curl",
    name: "curl",
    where: "terminal",
    body: (url, token, model) => `curl ${url}/v1/chat/completions \\
  -H "Authorization: Bearer ${token}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`,
  },
];

/** Configs for agents that use Temper as an MCP server (engine mcp_server.py). */
const MCP_RECIPES: Recipe[] = [
  {
    id: "json",
    name: "Cursor, Claude Desktop, VS Code",
    where: "mcp.json (mcpServers)",
    body: (url, token) => `{
  "mcpServers": {
    "temper": {
      "url": "${url}/mcp",
      "headers": { "Authorization": "Bearer ${token}" }
    }
  }
}`,
  },
  {
    id: "claude",
    name: "Claude Code",
    where: "terminal",
    body: (url, token) => `claude mcp add --transport http temper ${url}/mcp \\
  --header "Authorization: Bearer ${token}"`,
  },
  {
    id: "codex",
    name: "Codex CLI",
    where: "~/.codex/config.toml",
    body: (url) => `[mcp_servers.temper]
url = "${url}/mcp"
bearer_token_env_var = "TEMPER_API_KEY"
# then run:  TEMPER_API_KEY=<token> codex`,
  },
];

function Tabs({ recipes, active, onChange, url, token, model }: { recipes: Recipe[]; active: string; onChange: (id: string) => void; url: string; token: string; model: string }) {
  const recipe = recipes.find((r) => r.id === active) ?? recipes[0];
  const snippet = recipe.body(url, token, model);
  return (
    <>
      <div className="flex flex-wrap gap-1 border-b border-line" role="tablist">
        {recipes.map((r) => (
          <button
            key={r.id}
            role="tab"
            aria-selected={r.id === recipe.id}
            onClick={() => onChange(r.id)}
            className={cx("-mb-px border-b-2 px-3 py-2 text-[13px] font-medium", r.id === recipe.id ? "border-accent" : "border-transparent text-muted hover:text-text")}
          >
            {r.name}
          </button>
        ))}
      </div>
      <div className="relative mt-3 rounded-lg border border-line bg-surface">
        <div className="flex items-center justify-between border-b border-line px-3 py-1.5 text-[12px] text-muted">
          <span className="font-mono">{recipe.where}</span>
          <CopyButton text={snippet} />
        </div>
        <pre className="overflow-x-auto p-4 font-mono text-[12.5px] leading-relaxed">{snippet}</pre>
      </div>
    </>
  );
}

export function ConnectPage() {
  const { engine, runtime } = useApp();
  const [active, setActive] = useState(RECIPES[0].id);
  const [activeMcp, setActiveMcp] = useState(MCP_RECIPES[0].id);
  const [reveal, setReveal] = useState(false);
  const url = engine?.url ?? "";
  const token = engine?.token ?? "";
  const model = runtime?.state === "ready" && runtime.model ? runtime.model : "local-model";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title="Connect" description="Use the model loaded in Temper from coding agents, editors and your own code through OpenAI- and Anthropic-compatible APIs." />
      <div className="min-h-0 flex-1 overflow-y-auto px-7 py-5">
        <dl className="mb-5 grid max-w-[760px] grid-cols-[120px_1fr_auto] items-center gap-x-4 gap-y-2 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
          <dt className="text-muted">Base URL</dt>
          <dd className="font-mono text-[12.5px]">{url}/v1</dd>
          <CopyButton text={`${url}/v1`} />
          <dt className="text-muted">Temper's key</dt>
          <dd className="font-mono text-[12.5px]">
            {reveal ? token : "•".repeat(24)}{" "}
            <button className="ml-2 font-sans text-accent hover:underline" onClick={() => setReveal(!reveal)}>
              {reveal ? "Hide" : "Show"}
            </button>
          </dd>
          <CopyButton text={token} />
          <dt className="text-muted">Model</dt>
          <dd className="font-mono text-[12.5px]">{model}</dd>
          <span />
        </dl>

        {runtime?.state !== "ready" && (
          <div className="mb-5 max-w-[760px]">
            <Notice>Load a model on the Models page first. Requests fail until one is loaded.</Notice>
          </div>
        )}

        <div className="max-w-[760px]">
          <Tabs recipes={RECIPES} active={active} onChange={setActive} url={url} token={token} model={model} />
          <p className="mt-4 text-[13px] text-muted">
            The key stays the same when Temper restarts, so these configs keep working. From a terminal, <span className="font-mono text-[12.5px] text-text">temper start claude</span> (or{" "}
            <span className="font-mono text-[12.5px] text-text">codex</span>, <span className="font-mono text-[12.5px] text-text">opencode</span>, <span className="font-mono text-[12.5px] text-text">aider</span>) starts the agent
            already pointed at the loaded model. Claude Code reads a long system prompt and many tool descriptions, so load the model with a context of 32K or more, and pick one that's good at
            calling tools.
          </p>
        </div>

        <div className="mt-8 max-w-[760px]">
          <h2 className="text-[15px] font-semibold">Other apps</h2>
          <p className="mt-1 mb-3 text-[13px] text-muted">
            Many apps look for a local model at Ollama's or llama.cpp's usual address. Temper can answer there too, on this computer only. Like Ollama, those addresses take no key.
          </p>
          <DiscoverySettings />
        </div>

        <div className="mt-8 max-w-[760px]">
          <h2 className="text-[15px] font-semibold">Other devices</h2>
          <p className="mt-1 mb-3 text-[13px] text-muted">Let phones and other computers on your home network use the loaded model.</p>
          <LanAccess />
        </div>

        <div className="mt-8 max-w-[760px]">
          <h2 className="text-[15px] font-semibold">API keys</h2>
          <p className="mt-1 mb-3 text-[13px] text-muted">
            Give each app or device its own key instead of Temper's, so you can see what each one uses and revoke it on its own. These keys open the model API (
            <span className="font-mono text-[12.5px]">/v1</span>) and nothing else.
          </p>
          <ApiKeys />
        </div>

        <div className="mt-8 max-w-[760px]">
          <h2 className="text-[15px] font-semibold">Requests</h2>
          <p className="mt-1 mb-3 text-[13px] text-muted">The latest calls to the model API, including Temper's own chats.</p>
          <ApiMonitor />
        </div>

        <div className="mt-8 max-w-[760px]">
          <h2 className="text-[15px] font-semibold">MCP server</h2>
          <p className="mt-1 mb-3 text-[13px] text-muted">
            Agents that speak the Model Context Protocol can use Temper as a server at <span className="font-mono text-[12.5px]">{url}/mcp</span>: list and load this computer's models, ask
            them questions, and follow or cancel downloads, training and exports. It uses the same key.
          </p>
          <Tabs recipes={MCP_RECIPES} active={activeMcp} onChange={setActiveMcp} url={url} token={token} model={model} />
          <p className="mt-4 text-[13px] text-muted">
            Or, from a terminal: <span className="font-mono text-[12.5px] text-text">temper mcp add claude</span> (or <span className="font-mono text-[12.5px] text-text">codex</span>, <span className="font-mono text-[12.5px] text-text">cursor</span>,{" "}
            <span className="font-mono text-[12.5px] text-text">vscode</span>) adds it for you, and <span className="font-mono text-[12.5px] text-text">temper doctor --mcp</span> checks it.
          </p>
        </div>
      </div>
    </div>
  );
}
