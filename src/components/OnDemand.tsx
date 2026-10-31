import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { useT, withNodes } from "../lib/i18n";
import { Toggle } from "./ui";

/** Connect → whether API requests naming a downloaded model load it (engine on_demand.py). */
export function OnDemand() {
  const t = useT();
  const [on, setOn] = useState<boolean | null>(null);
  useEffect(() => {
    api<{ load_on_demand: boolean }>("/api/runtime/loaded").then((r) => setOn(r.load_on_demand), () => {});
  }, []);
  if (on === null) return null;
  return (
    <div className="mb-5 max-w-[760px]">
      <Toggle
        checked={on}
        onChange={(value) => {
          setOn(value);
          void api("/api/runtime/loaded/on-demand", { method: "PUT", json: { enabled: value } });
        }}
        label={t("ondemand.label")}
      />
      <p className="mt-1 text-[12px] text-muted">
        {withNodes(t("ondemand.hint"), {
          example: <span className="font-mono">Qwen/Qwen3-4B-GGUF</span>,
          models: <span className="font-mono">/v1/models</span>,
          ttl: <span className="font-mono">ttl</span>,
        })}
      </p>
    </div>
  );
}
