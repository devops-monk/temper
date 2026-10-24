import { Check, Copy, TriangleAlert } from "lucide-react";
import { useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from "react";

const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" }) {
  return (
    <button
      {...props}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-45",
        size === "sm" ? "h-7 px-2.5 text-[13px]" : "h-9 px-3.5",
        variant === "primary" && "bg-accent text-accent-text hover:brightness-110",
        variant === "secondary" && "border border-line bg-surface hover:bg-surface-2",
        variant === "ghost" && "text-muted hover:bg-surface-2 hover:text-text",
        variant === "danger" && "border border-line bg-surface text-danger hover:bg-surface-2",
        className,
      )}
    />
  );
}

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cx(
        "h-9 w-full rounded-md border border-line bg-surface px-3 text-text placeholder:text-muted/70 focus:border-accent focus:outline-none",
        className,
      )}
    />
  );
}

export function Select({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cx("h-9 w-full rounded-md border border-line bg-surface px-2.5 text-text focus:border-accent focus:outline-none", className)}
    >
      {children}
    </select>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[13px] font-medium">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 py-1">
      <span className="text-[13px]">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx("relative h-5 w-9 shrink-0 rounded-full transition-colors", checked ? "bg-accent" : "bg-line")}
      >
        <span className={cx("absolute top-0.5 size-4 rounded-full bg-white shadow transition-all", checked ? "left-[18px]" : "left-0.5")} />
      </button>
    </label>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: ReactNode }) {
  return (
    <header className="flex items-end justify-between gap-4 border-b border-line px-7 pt-6 pb-4">
      <div className="min-w-0">
        <h1 className="text-[22px] leading-tight font-semibold tracking-[-0.01em]">{title}</h1>
        {description && <p className="mt-1 max-w-[70ch] text-muted">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="py-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-semibold">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Notice({ tone = "warn", children }: { tone?: "warn" | "error" | "info"; children: ReactNode }) {
  return (
    <div
      className={cx(
        "flex gap-2.5 rounded-lg border px-3.5 py-3 text-[13px]",
        tone === "error" && "border-danger/40 bg-danger/8 text-text",
        tone === "warn" && "border-straw/50 bg-straw/10 text-text",
        tone === "info" && "border-line bg-surface-2 text-text",
      )}
    >
      {tone !== "info" && <TriangleAlert className={cx("mt-0.5 size-4 shrink-0", tone === "error" ? "text-danger" : "text-straw")} />}
      <div className="min-w-0 [&_code]:rounded [&_code]:bg-surface [&_code]:px-1 [&_code]:font-mono [&_code]:text-[12px]">{children}</div>
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex min-h-full flex-col items-center justify-center px-6 py-16 text-center">
      <div className="mb-3 text-muted">{icon}</div>
      <p className="font-semibold">{title}</p>
      {children && <div className="mt-1 max-w-[52ch] text-muted">{children}</div>}
    </div>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "ok" | "warn" | "accent" }) {
  return (
    <span
      className={cx(
        "inline-flex items-center rounded px-1.5 py-px text-[11.5px] font-medium whitespace-nowrap",
        tone === "neutral" && "bg-surface-2 text-muted",
        tone === "ok" && "bg-ok/12 text-ok",
        tone === "warn" && "bg-straw/15 text-straw",
        tone === "accent" && "bg-accent/12 text-accent",
      )}
    >
      {children}
    </span>
  );
}

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      onClick={() => {
        void navigator.clipboard.writeText(text);
        setDone(true);
        setTimeout(() => setDone(false), 1500);
      }}
    >
      {done ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      {done ? "Copied" : label}
    </Button>
  );
}

export function ProgressBar({ value, temper = false }: { value: number | null; temper?: boolean }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={value == null ? undefined : Math.round(value * 100)}>
      {value == null ? (
        <div className={cx("animate-indeterminate h-full w-1/3 rounded-full", temper ? "bg-temper" : "bg-accent")} />
      ) : (
        <div
          className={cx("h-full rounded-full transition-[width] duration-500", temper ? "bg-temper" : "bg-accent")}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      )}
    </div>
  );
}

export { cx };
