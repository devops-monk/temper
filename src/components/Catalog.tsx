import { ChevronDown, FlaskConical, Lock, MessageSquare } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "../lib/engine";
import { type Key, t as tr, useT } from "../lib/i18n";
import { useApp } from "../lib/store";
import { Badge, Button, Notice, cx } from "./ui";

export type Sampling = Partial<Record<"temperature" | "top_p" | "top_k" | "min_p" | "repetition_penalty" | "presence_penalty", number>>;

export interface CatalogModel {
  id: string;
  label: string;
  params: number;
  context: number | null;
  license: string | null;
  gated: boolean;
  vision: boolean;
  moe: boolean;
  thinking: "default" | "optional" | null;
  sampling: Sampling;
  gguf: string | null;
  mlx: string | null;
  fit: { total: number; verdict: "fits" | "tight" | "too_large" };
  trainable: boolean;
  train_repo: string;
  downloaded: { gguf: boolean; train: boolean };
}

interface Family {
  id: string;
  name: string;
  publisher: string;
  about: string;
  uses: string[];
  suggested: string | null;
  variants: CatalogModel[];
}

interface Catalog {
  families: Family[];
  trainer: "mlx" | "pytorch" | null;
}

const FIT = {
  fits: { label: "cat.fits", tone: "ok" },
  tight: { label: "cat.tight", tone: "warn" },
  too_large: { label: "cat.too_large", tone: "neutral" },
} as const;

const SAMPLING_LABEL: Record<keyof Sampling, Key | string> = {
  temperature: "cat.temperature",
  top_p: "top P",
  top_k: "top K",
  min_p: "min P",
  repetition_penalty: "cat.repeat",
  presence_penalty: "cat.presence",
};

export function samplingText(s: Sampling): string {
  return (Object.entries(s) as [keyof Sampling, number][]).map(([k, v]) => `${SAMPLING_LABEL[k].startsWith("cat.") ? tr(SAMPLING_LABEL[k] as Key) : SAMPLING_LABEL[k]} ${v}`).join(", ");
}

function contextText(n: number | null) {
  if (!n) return null;
  return tr("cat.context", { n: n >= 1_000_000 ? `${Math.round(n / 1_048_576)}M` : `${Math.round(n / 1024)}K` });
}

function Row({ model, family, suggested, trainer, renderRepo }: { model: CatalogModel; family: Family; suggested: boolean; trainer: Catalog["trainer"]; renderRepo: (id: string) => ReactNode }) {
  const t = useT();
  const { go, setPendingBaseModel } = useApp();
  const [open, setOpen] = useState(false);
  const fit = FIT[model.fit.verdict];
  const trainHint = !trainer ? t("cat.add_training") : model.trainable ? undefined : t("cat.unsupported");

  return (
    <li>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{model.id.split("/")[1]}</span>
            {suggested && <Badge tone="accent">{t("cat.best")}</Badge>}
            {model.gated && <Lock className="size-3.5 text-straw" aria-label={t("cat.gated")} />}
          </div>
          <div className="mt-0.5 flex flex-wrap gap-x-3 text-[12px] text-muted">
            <span className="tabular-nums">{t("cat.params", { n: model.params.toFixed(model.params < 10 ? 1 : 0) })}</span>
            {contextText(model.context) && <span>{contextText(model.context)}</span>}
            {model.moe && <span title={t("cat.moe_title")}>{t("cat.moe")}</span>}
            {model.vision && <span>{t("cat.vision")}</span>}
            {model.thinking && <span>{t("cat.thinks")}</span>}
            {model.license && <span>{model.license}</span>}
          </div>
        </div>
        <span title={t("cat.fit_title", { gb: (model.fit.total / 1e9).toFixed(1) })}>
          <Badge tone={fit.tone}>{t(fit.label)}</Badge>
        </span>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setOpen(!open)} disabled={!model.gguf} aria-expanded={open}>
            <MessageSquare className="size-3.5" /> {t(model.downloaded.gguf ? "cat.downloaded" : "cat.chat")}
            <ChevronDown className={cx("size-3.5 transition-transform", open && "rotate-180")} />
          </Button>
          <Button
            size="sm"
            disabled={!model.trainable}
            title={trainHint}
            onClick={() => {
              setPendingBaseModel(model.train_repo);
              go("train");
            }}
          >
            <FlaskConical className="size-3.5" /> {t("cat.fine_tune")}
          </Button>
        </div>
      </div>
      {open && model.gguf && (
        <div className="space-y-3 bg-surface-2/60 px-4 pt-1 pb-3">
          <p className="text-[12.5px] text-muted">
            {model.gguf} · {Object.keys(model.sampling).length ? t("cat.recommends", { publisher: family.publisher, settings: samplingText(model.sampling) }) : t("cat.no_settings", { publisher: family.publisher })}
          </p>
          {renderRepo(model.gguf)}
        </div>
      )}
    </li>
  );
}

/** Hand-picked models with their publishers' facts and settings, sized against this computer. */
export function CatalogView({ renderRepo }: { renderRepo: (id: string) => ReactNode }) {
  const t = useT();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Catalog>("/api/catalog").then(setCatalog, (e) => setError(e.message));
  }, []);

  if (error) return <div className="px-7 py-5"><Notice tone="error">{error}</Notice></div>;
  if (!catalog) return <p className="px-7 py-5 text-[13px] text-muted">{t("cat.loading")}</p>;
  return (
    <div className="space-y-8 px-7 py-5">
      <p className="max-w-[640px] text-[13px] text-muted">
        {t("cat.about")}
      </p>
      {catalog.families.map((family) => (
        <section key={family.id} aria-labelledby={`family-${family.id}`}>
          <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h2 id={`family-${family.id}`} className="text-[15px] font-semibold">
              {family.name}
            </h2>
            <span className="text-[13px] text-muted">{family.publisher}</span>
            <span className="ml-auto flex flex-wrap gap-1.5">
              {family.uses.map((u) => (
                <Badge key={u}>{u}</Badge>
              ))}
            </span>
          </div>
          <p className="mb-3 max-w-[720px] text-[13px]">{family.about}</p>
          <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
            {family.variants.map((m) => (
              <Row key={m.id} model={m} family={family} suggested={family.suggested === m.id} trainer={catalog.trainer} renderRepo={renderRepo} />
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
