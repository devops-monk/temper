import { Save, Star, Trash2 } from "lucide-react";
import { useState } from "react";
import { api } from "../lib/engine";
import { Button, Input, Select, cx } from "./ui";

export interface Preset {
  id: string;
  name: string;
  /** null: for any model. */
  model: string | null;
  settings: Record<string, string | number>;
  thinking: Record<string, unknown> | null;
}

/** The generation settings' source: the model's recommended settings, Temper's defaults, or a saved preset. */
export function PresetBar({
  model,
  presets,
  selected,
  modified,
  defaultId,
  hasRecommended,
  current,
  onSelect,
  onReload,
}: {
  model: string | null;
  presets: Preset[];
  selected: string;
  modified: boolean;
  defaultId: string | null;
  hasRecommended: boolean;
  current: { settings: object; thinking: object };
  onSelect: (id: string) => void;
  onReload: (select?: string) => void;
}) {
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState("");
  const [onlyThis, setOnlyThis] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const preset = presets.find((p) => p.id === selected);

  async function run(call: Promise<unknown>, select?: string) {
    setError(null);
    try {
      await call;
      onReload(select);
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    }
  }

  async function saveNew() {
    const body = { name, model: onlyThis ? model : null, ...current };
    try {
      const created = await api<Preset>("/api/presets", { json: body });
      setSaving(false);
      setName("");
      setError(null);
      onReload(created.id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="mb-4">
      <div className="flex items-center gap-1.5">
        <Select value={selected} onChange={(e) => onSelect(e.target.value)} className="h-8 min-w-0 flex-1 text-[13px]" aria-label="Settings preset">
          {hasRecommended && <option value="recommended">Recommended{modified && selected === "recommended" ? " (changed)" : ""}</option>}
          <option value="defaults">Temper defaults{modified && selected === "defaults" ? " (changed)" : ""}</option>
          {presets.length > 0 && (
            <optgroup label="Your presets">
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                  {p.id === defaultId ? " ★" : ""}
                  {modified && p.id === selected ? " (changed)" : ""}
                </option>
              ))}
            </optgroup>
          )}
        </Select>
        <Button size="sm" variant="ghost" className="px-2" onClick={() => setSaving((s) => !s)} title="Save these settings as a preset" aria-label="Save as preset">
          <Save className="size-3.5" />
        </Button>
      </div>
      {preset && (
        <div className="mt-1.5 flex items-center gap-1 text-[12px]">
          {modified && (
            <button className="rounded px-1.5 py-0.5 text-accent hover:bg-surface-2" onClick={() => void run(api(`/api/presets/${preset.id}`, { method: "PUT", json: { name: preset.name, model: preset.model, ...current } }), preset.id)}>
              Save changes
            </button>
          )}
          {model && (
            <button
              className={cx("flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-surface-2", preset.id === defaultId ? "text-straw" : "text-muted")}
              onClick={() => void run(api("/api/presets", { method: "PUT", json: { model, preset: preset.id === defaultId ? null : preset.id } }), preset.id)}
              title={preset.id === defaultId ? "Stop using this preset when this model loads" : "Use this preset whenever this model loads"}
            >
              <Star className={cx("size-3", preset.id === defaultId && "fill-current")} />
              {preset.id === defaultId ? "Loads with this model" : "Load with this model"}
            </button>
          )}
          <button
            className="ml-auto rounded p-1 text-muted hover:bg-surface-2 hover:text-danger"
            onClick={() => confirm(`Delete the preset ${preset.name}?`) && void run(api(`/api/presets/${preset.id}`, { method: "DELETE" }), hasRecommended ? "recommended" : "defaults")}
            title="Delete preset"
            aria-label={`Delete ${preset.name}`}
          >
            <Trash2 className="size-3" />
          </button>
        </div>
      )}
      {preset?.model === null && <p className="mt-1 text-[11.5px] text-muted">For every model.</p>}
      {saving && (
        <form
          className="mt-2 rounded-lg border border-line bg-surface p-2.5"
          onSubmit={(e) => {
            e.preventDefault();
            void saveNew();
          }}
        >
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Preset name, e.g. Precise" autoFocus className="h-8 text-[13px]" aria-label="Preset name" />
          {model && (
            <label className="mt-2 flex items-center gap-2 text-[12.5px]">
              <input type="checkbox" checked={onlyThis} onChange={(e) => setOnlyThis(e.target.checked)} className="accent-[var(--accent)]" />
              <span className="min-w-0 truncate">Only for {model}</span>
            </label>
          )}
          <p className="mt-1.5 text-[11.5px] text-muted">Saves the system prompt, sampling, max tokens and thinking choices.</p>
          <div className="mt-2 flex gap-2">
            <Button size="sm" variant="primary" type="submit" disabled={!name.trim()}>
              Save preset
            </Button>
            <Button size="sm" variant="ghost" type="button" onClick={() => setSaving(false)}>
              Cancel
            </Button>
          </div>
        </form>
      )}
      {error && <p className="mt-1.5 text-[12px] text-danger">{error}</p>}
    </div>
  );
}
