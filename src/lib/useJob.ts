import { useEffect, useState } from "react";
import { stream } from "./engine";
import { useApp } from "./store";
import type { JobEvent, JobStatus } from "./types";

/** Follows one job's event stream. Returns every event so far plus the latest status. */
export function useJob(jobId: string | null) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const refreshJobs = useApp((s) => s.refreshJobs);

  useEffect(() => {
    setEvents([]);
    setStatus(jobId ? "running" : null);
    if (!jobId) return;
    const ctrl = new AbortController();
    const buffer: JobEvent[] = [];
    // Batch renders: workers can log hundreds of lines a second.
    const flush = setInterval(() => {
      if (buffer.length) setEvents((prev) => prev.concat(buffer.splice(0)));
    }, 200);

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
        setEvents((prev) => prev.concat(buffer.splice(0)));
        void refreshJobs();
      });

    return () => {
      ctrl.abort();
      clearInterval(flush);
    };
  }, [jobId, refreshJobs]);

  return { events, status };
}
