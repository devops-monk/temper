import { LogOut, Trash2, UserPlus } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, saveEngineKey } from "../lib/engine";
import { useT, withNodes } from "../lib/i18n";
import { useApp } from "../lib/store";
import { Badge, Button, Input, Section } from "./ui";

/** An account, as the engine lists it (engine accounts.py). */
interface Account {
  name: string;
  admin: boolean;
}

/** Settings → Accounts: your own account, and for admins everyone's. */
export function Accounts() {
  const t = useT();
  const { me, refreshEngine, refreshAccount } = useApp();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [admin, setAdmin] = useState(false);
  const [current, setCurrent] = useState("");
  const [fresh, setFresh] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const enabled = !!me?.enabled;

  const reload = useCallback(() => {
    if (me?.admin) void api<{ accounts: Account[] }>("/api/accounts").then((r) => setAccounts(r.accounts), () => {});
  }, [me?.admin]);
  useEffect(reload, [reload]);
  if (!me) return null;

  async function run(action: () => Promise<unknown>, done?: string) {
    setError(null);
    setNotice(null);
    try {
      await action();
      if (done) setNotice(done);
      reload();
      void refreshAccount();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function signOut() {
    await api("/api/accounts/logout", { method: "POST" }).catch(() => {});
    saveEngineKey("");
    await refreshEngine();
  }

  return (
    <Section title={t("accounts.title")}>
      <p className="-mt-1 mb-3 text-[12.5px] text-muted">{withNodes(t("accounts.hint"), { cmd: <span className="font-mono">temper serve</span> })}</p>

      {me.account && (
        <div className="mb-3 rounded-lg border border-line bg-surface px-4 py-3 text-[13px]">
          <div className="flex items-center gap-2">
            <span>{withNodes(t("accounts.signed_in_as"), { name: <span className="font-medium">{me.account.name}</span> })}</span>
            <Badge>{t(me.account.admin ? "accounts.admin" : "accounts.member")}</Badge>
            <span className="flex-1" />
            <Button size="sm" onClick={() => void signOut()}>
              <LogOut className="size-3.5" /> {t("accounts.sign_out")}
            </Button>
          </div>
          <form
            className="mt-3 flex flex-wrap items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void run(() => api("/api/accounts/me/password", { method: "PUT", json: { current, password: fresh } }).then(signOut));
            }}
          >
            <span className="w-44">
              <Input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} placeholder={t("accounts.current_password")} aria-label={t("accounts.current_password")} autoComplete="current-password" />
            </span>
            <span className="w-44">
              <Input type="password" value={fresh} onChange={(e) => setFresh(e.target.value)} placeholder={t("accounts.new_password")} aria-label={t("accounts.new_password")} autoComplete="new-password" />
            </span>
            <Button type="submit" size="sm" disabled={!current || fresh.length < 8}>
              {t("accounts.change_password")}
            </Button>
          </form>
        </div>
      )}

      {me.admin && (
        <>
          {accounts.length > 0 && (
            <ul className="mb-3 divide-y divide-line rounded-lg border border-line bg-surface text-[13px]">
              {accounts.map((a) => (
                <li key={a.name} className="flex items-center gap-2 px-4 py-2">
                  <span className="font-medium">{a.name}</span>
                  <Badge>{t(a.admin ? "accounts.admin" : "accounts.member")}</Badge>
                  <span className="flex-1" />
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      const next = prompt(t("accounts.reset_prompt", { name: a.name }));
                      if (next) void run(() => api(`/api/accounts/${a.name}/password`, { method: "PUT", json: { password: next } }), t("accounts.reset_done", { name: a.name }));
                    }}
                  >
                    {t("accounts.reset")}
                  </Button>
                  {a.name !== me.account?.name && (
                    <button
                      className="text-muted hover:text-danger"
                      onClick={() => confirm(t("accounts.remove_confirm", { name: a.name })) && void run(() => api(`/api/accounts/${a.name}`, { method: "DELETE" }))}
                      title={t("accounts.remove", { name: a.name })}
                      aria-label={t("accounts.remove", { name: a.name })}
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
          <form
            className="flex flex-wrap items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void run(async () => {
                await api("/api/accounts", { json: { name: name.trim(), password, admin: admin || !enabled } });
                setName("");
                setPassword("");
                setAdmin(false);
              });
            }}
          >
            <span className="w-40">
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={t("accounts.name")} aria-label={t("accounts.name")} autoComplete="off" maxLength={32} />
            </span>
            <span className="w-44">
              <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder={t("accounts.password")} aria-label={t("accounts.password")} autoComplete="new-password" />
            </span>
            <label className="flex items-center gap-1.5 text-[13px]">
              <input type="checkbox" checked={admin || !enabled} disabled={!enabled} onChange={(e) => setAdmin(e.target.checked)} />
              {t("accounts.admin")}
            </label>
            <Button type="submit" size="sm" disabled={!name.trim() || password.length < 8}>
              <UserPlus className="size-3.5" /> {t(enabled ? "accounts.add" : "accounts.add_first")}
            </Button>
          </form>
        </>
      )}
      {notice && <p className="mt-2 text-[12px] text-muted">{notice}</p>}
      {error && <p className="mt-2 text-[12px] text-danger">{error}</p>}
    </Section>
  );
}
