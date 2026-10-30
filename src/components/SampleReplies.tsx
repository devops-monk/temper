import { useState } from "react";

export interface Sample {
  step: number;
  prompt: string;
  reply: string;
  expected: string | null;
}

/** The model's replies to held-out prompts at each evaluation; a slider steps back through them. */
export function SampleReplies({ samples }: { samples: Sample[] }) {
  const steps = [...new Set(samples.map((s) => s.step))];
  const [picked, setPicked] = useState<number | null>(null);
  const step = picked ?? steps[steps.length - 1];
  return (
    <div className="rounded-lg border border-line bg-surface p-3 text-[13px]">
      <div className="mb-2 flex items-center gap-3">
        <span className="font-medium">Sample replies</span>
        <span className="text-[12px] text-muted">at step {step}</span>
        {steps.length > 1 && (
          <input
            type="range"
            min={0}
            max={steps.length - 1}
            value={steps.indexOf(step)}
            onChange={(e) => setPicked(Number(e.target.value) === steps.length - 1 ? null : steps[Number(e.target.value)])}
            className="ml-auto w-40 accent-[var(--accent)]"
            aria-label="Evaluation step"
          />
        )}
      </div>
      <div className="space-y-3">
        {samples
          .filter((s) => s.step === step)
          .map((s, i) => (
            <div key={i} className="space-y-1">
              <p className="line-clamp-2 text-muted">{s.prompt}</p>
              <p className="max-h-40 overflow-y-auto whitespace-pre-wrap">{s.reply || <span className="text-muted">(no reply)</span>}</p>
              {s.expected && (
                <details className="text-[12px] text-muted">
                  <summary className="cursor-pointer">Reply in the data</summary>
                  <p className="mt-1 whitespace-pre-wrap">{s.expected}</p>
                </details>
              )}
            </div>
          ))}
      </div>
    </div>
  );
}
