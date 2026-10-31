import { useEffect, useState } from "react";
import { api } from "../lib/engine";
import { Toggle } from "./ui";

/** Connect → whether API requests naming a downloaded model load it (engine on_demand.py). */
export function OnDemand() {
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
        label="Load models on demand"
      />
      <p className="mt-1 text-[12px] text-muted">
        A request naming a model you've downloaded, like <span className="font-mono">Qwen/Qwen3-4B-GGUF</span>, loads it alongside the others; <span className="font-mono">/v1/models</span>{" "}
        lists them all. It unloads after the idle time in Settings, or the request's <span className="font-mono">ttl</span> in seconds. Models loaded this way make room for each other;
        ones you load yourself stay.
      </p>
    </div>
  );
}
