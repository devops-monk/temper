import { useState } from "react";
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
    id: "curl",
    name: "curl",
    where: "terminal",
    body: (url, token, model) => `curl ${url}/v1/chat/completions \\
  -H "Authorization: Bearer ${token}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`,
  },
];

export function ConnectPage() {
  const { engine, runtime } = useApp();
  const [active, setActive] = useState(RECIPES[0].id);
  const [reveal, setReveal] = useState(false);
  const url = engine?.url ?? "";
  const token = engine?.token ?? "";
  const model = runtime?.state === "ready" && runtime.model ? runtime.model : "local-model";
  const recipe = RECIPES.find((r) => r.id === active)!;
  const snippet = recipe.body(url, token, model);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title="Connect" description="Use the model loaded in Temper from coding agents, editors and your own code through an OpenAI-compatible API." />
      <div className="min-h-0 flex-1 overflow-y-auto px-7 py-5">
        <dl className="mb-5 grid max-w-[760px] grid-cols-[120px_1fr_auto] items-center gap-x-4 gap-y-2 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
          <dt className="text-muted">Base URL</dt>
          <dd className="font-mono text-[12.5px]">{url}/v1</dd>
          <CopyButton text={`${url}/v1`} />
          <dt className="text-muted">API key</dt>
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
          <div className="flex flex-wrap gap-1 border-b border-line" role="tablist">
            {RECIPES.map((r) => (
              <button
                key={r.id}
                role="tab"
                aria-selected={r.id === active}
                onClick={() => setActive(r.id)}
                className={cx("-mb-px border-b-2 px-3 py-2 text-[13px] font-medium", r.id === active ? "border-accent" : "border-transparent text-muted hover:text-text")}
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
          <p className="mt-4 text-[13px] text-muted">
            The key changes every time Temper starts. Claude Code needs an Anthropic-compatible endpoint, which is on the roadmap (PLAN.md, item 10.2).
          </p>
        </div>
      </div>
    </div>
  );
}
