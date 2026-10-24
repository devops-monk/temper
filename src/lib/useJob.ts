import { useEffect, useState } from "react";
import { stream } from "./engine";
import { useApp } from "./store";
import type { JobEvent, JobStatus } from "./types";

/**
 * Follows one job's event stream. Returns every event so far plus the status from the stream, which
 * stays null until the stream reports one; callers fall back to the job's listed status meanwhile.
 */
export function useJob(jobId: string | null) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const refreshJobs = useApp((s) => s.refreshJobs);

  useEffect(() => {
    setEvents([]);
    setStatus(null);
    if (!jobId) return;
    const ctrl = new AbortController();
    const buffer: JobEvent[] = [];
    // Batch renders: workers can log hundreds of lines a second. Take the batch outside the
    // updater; React may call updaters twice, and a second splice would find the buffer empty.
    const flushBuffer = () => {
      const batch = buffer.splice(0);
      if (batch.length) setEvents((prev) => prev.concat(batch));
    };
    const flush = setInterval(flushBuffer, 200);

    stream(
      `/api/jobs/${jobId}/events`,
      (data) => {
        const ev = JSON.parse(data) as JobEvent;
        buffer.push(ev);
        if (ev.event === "status") setStatus(ev.status as JobStatus);
      },
      { signal: ctrl.signal },
    )
      .catch(() => {})
      .finally(() => {
        flushBuffer();
        void refreshJobs();
      });

    return () => {
      ctrl.abort();
      clearInterval(flush);
    };
  }, [jobId, refreshJobs]);

  return { events, status };
}
