import { ExternalLink, RotateCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, inTauri, openExternal } from "../lib/engine";
import { useApp } from "../lib/store";
import { Badge, Button, Input, Notice, Section, cx } from "./ui";

interface Account {
  signed_in: boolean;
  source: "temper" | "cli" | null;
  name?: string | null;
  orgs?: string[];
  invalid?: boolean;
  offline?: boolean;
}

interface Access {
  access: boolean;
  gated: boolean;
  has_token: boolean;
  url: string;
  message?: string;
}

const TOKENS_URL = "https://huggingface.co/settings/tokens";

async function keychain(command: "hf_token_save" | "hf_token_delete", args?: Record<string, unknown>) {
  if (!inTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke(command, args);
}

/** Settings: the Hugging Face token Temper uses for gated and private models, kept in the system keychain. */
export function HfAccount({ ready }: { ready: boolean }) {
  const [account, setAccount] = useState<Account | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  const load = useCallback(() => {
    api<Account>("/api/hf/account").then(setAccount, () => setAccount(null));
  }, []);
  useEffect(() => {
    if (ready) load();
  }, [ready, load]);

  async function save() {
    setBusy(true);
    setError(null);
    setWarning(null);
    try {
      const next = await api<Account>("/api/hf/token", { method: "POST", json: { token } });
      setAccount(next);
      setToken("");
      // The engine already uses it; the keychain only matters for the next launch.
      await keychain("hf_token_save", { token }).catch((e) => setWarning(String(e)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await keychain("hf_token_delete");
      setAccount(await api<Account>("/api/hf/token", { method: "DELETE" }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const ownToken = account?.source === "temper";
  return (
    <Section title="Hugging Face account">
      <div className="rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
        {!account && <p className="text-muted">{ready ? "Checking…" : "Start the engine to manage your token."}</p>}
        {account?.signed_in && (
          <div className="flex items-center justify-between gap-3">
            <div>
              <span className="font-medium">{account.name ?? "Signed in"}</span>
              {account.offline && <span className="text-muted"> (offline, not checked)</span>}
              <p className="mt-0.5 text-[12px] text-muted">
                {ownToken ? (inTauri ? "Token saved in your system keychain." : "Token set for this engine session.") : "Using the token from huggingface-cli login."}
                {account.orgs && account.orgs.length > 0 && ` Member of ${account.orgs.join(", ")}.`}
              </p>
            </div>
            {ownToken ? (
              <Button size="sm" disabled={busy} onClick={() => void remove()}>
                Remove token
              </Button>
            ) : (
              <Badge>CLI token</Badge>
            )}
          </div>
        )}
        {account && !account.signed_in && (
          <p className="text-muted">
            {account.invalid ? "The saved token no longer works. Add a new one." : "No token. Public models download without one; gated models such as Llama and Gemma need it."}
          </p>
        )}
        {account && (!account.signed_in || !ownToken) && (
          <form
            className="mt-3 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (token.trim()) void save();
            }}
          >
            <Input type="password" autoComplete="off" spellCheck={false} placeholder="hf_…" value={token} onChange={(e) => setToken(e.target.value)} className="flex-1" aria-label="Hugging Face token" />
            <Button type="submit" variant="primary" disabled={busy || !token.trim()}>
              {busy ? "Checking…" : "Save token"}
            </Button>
          </form>
        )}
        {account && !ownToken && (
          <button type="button" className="mt-2 inline-flex items-center gap-1 text-[12px] text-accent hover:underline" onClick={() => void openExternal(TOKENS_URL)}>
            Create a read token on Hugging Face <ExternalLink className="size-3" />
          </button>
        )}
      </div>
      {error && (
        <div className="mt-3">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
      {warning && (
        <div className="mt-3">
          <Notice>{warning}</Notice>
        </div>
      )}
    </Section>
  );
}

/** Models: checks a repo before download and explains what to do when it's gated or private. */
export function RepoAccess({ repoId }: { repoId: string }) {
  const go = useApp((a) => a.go);
  const [access, setAccess] = useState<Access | null>(null);
  const [checking, setChecking] = useState(false);

  const check = useCallback(() => {
    setChecking(true);
    api<Access>(`/api/hf/access?repo_id=${encodeURIComponent(repoId)}`)
      .then(setAccess, () => setAccess(null))
      .finally(() => setChecking(false));
  }, [repoId]);
  useEffect(() => {
    setAccess(null);
    check();
  }, [check]);

  if (!access || access.access) return null;
  return (
    <Notice tone="warn">
      <p>{access.message}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        <Button size="sm" onClick={() => void openExternal(access.url)}>
          <ExternalLink className="size-3.5" /> Open model page
        </Button>
        {!access.has_token && (
          <Button size="sm" onClick={() => go("settings")}>
            Add token in Settings
          </Button>
        )}
        <Button size="sm" variant="ghost" disabled={checking} onClick={check}>
          <RotateCw className={cx("size-3.5", checking && "animate-spin")} /> Check again
        </Button>
      </div>
    </Notice>
  );
}
