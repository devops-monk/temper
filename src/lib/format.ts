export function bytes(n: number | null | undefined): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1);
  const v = n / 1024 ** i;
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function compact(n: number | null | undefined): string {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n ?? 0);
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function relativeDate(iso: string | number | null | undefined): string {
  if (!iso) return "";
  const date = typeof iso === "number" ? new Date(iso * 1000) : new Date(iso);
  const days = Math.round((date.getTime() - Date.now()) / 86_400_000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (Math.abs(days) < 1) return "today";
  if (Math.abs(days) < 30) return rtf.format(days, "day");
  if (Math.abs(days) < 365) return rtf.format(Math.round(days / 30), "month");
  return rtf.format(Math.round(days / 365), "year");
}

/** Short quant label from a GGUF filename, e.g. "Qwen3-8B-UD-Q4_K_XL.gguf" → "UD-Q4_K_XL". */
export function quantLabel(file: string): string {
  const name = file.split("/").pop()!.replace(/\.gguf$/i, "");
  const m = name.match(/((?:UD-)?(?:I?Q\d[\w]*|BF16|F16|F32|MXFP4\w*))(?:-\d+-of-\d+)?$/i);
  return m ? m[1] : name;
}

/** An Ollama name from a GGUF's file name: "20261024-154903-gsm8k-math-tutor-Q4_K_M.gguf" → "gsm8k-math-tutor:q4_k_m". */
export function ollamaName(path: string): string {
  const stem = path.split(/[\\/]/).pop()!.replace(/\.gguf$/i, "");
  const quant = stem.match(/-((?:UD-)?I?Q\d\w*|F16|BF16|F32)$/i)?.[1];
  const base = (quant ? stem.slice(0, -quant.length - 1) : stem).replace(/^\d{8}-\d{6}-/, "");
  const slug = base.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[^a-z0-9]+|-+$/g, "") || "model";
  return quant ? `${slug}:${quant.toLowerCase()}` : slug;
}
