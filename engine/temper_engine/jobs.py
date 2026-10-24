"""Background jobs (downloads, training, export) with an event log the UI streams over SSE."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

TERMINAL = {"done", "failed", "cancelled"}


@dataclass
class Job:
    kind: str
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "running"
    progress: float | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    process: subprocess.Popen | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: str, **data: Any) -> None:
        with self.lock:
            self.events.append({"seq": len(self.events), "event": event, "t": time.time(), **data})

    def log(self, line: str) -> None:
        self.emit("log", line=line)

    def finish(self, status: str, error: str | None = None, **result: Any) -> None:
        if self.status in TERMINAL:
            return
        self.status, self.error = status, error
        self.result.update(result)
        if status == "done":
            self.progress = 1.0
        self.emit("status", status=status, error=error, result=self.result)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "result": self.result,
            "created": self.created,
        }


class Registry:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def create(self, kind: str, title: str) -> Job:
        job = Job(kind=kind, title=title)
        self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise HTTPException(404, f"No job {job_id}")
        return self.jobs[job_id]

    def run_thread(self, job: Job, target: Callable[[Job], None]) -> None:
        def wrapper() -> None:
            try:
                target(job)
                job.finish("cancelled" if job.cancel_requested else "done")
            except Exception as exc:  # surfaced to the UI as the job's error
                job.finish("cancelled" if job.cancel_requested else "failed", str(exc))

        threading.Thread(target=wrapper, daemon=True).start()

    def run_worker(self, job: Job, module: str, config_path: str) -> None:
        """Run `python -m <module> <config>` and translate its JSON lines into job events.

        Workers import torch/unsloth, which are slow to load and can crash the interpreter,
        so they never run inside the engine process.
        """
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", module, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        job.process = proc

        def pump() -> None:
            assert proc.stdout
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line.startswith("@@"):
                    try:
                        msg = json.loads(line[2:])
                    except json.JSONDecodeError:
                        job.log(line)
                        continue
                    event = msg.pop("event")
                    if event == "progress":
                        job.progress = msg.get("value")
                    if event == "result":
                        job.result.update(msg)
                    if event == "error":
                        job.error = msg.get("message")
                    job.emit(event, **msg)
                elif line:
                    job.log(line)
            code = proc.wait()
            if job.cancel_requested:
                job.finish("cancelled")
            elif code == 0:
                job.finish("done")
            else:
                job.finish("failed", job.error or f"Worker exited with code {code}")

        threading.Thread(target=pump, daemon=True).start()

    def cancel(self, job: Job) -> None:
        job.cancel_requested = True
        if job.process and job.process.poll() is None:
            job.process.terminate()

    def cancel_all(self) -> None:
        for job in self.jobs.values():
            if job.status not in TERMINAL:
                self.cancel(job)


registry = Registry()


def emit_from_worker(event: str, **data: Any) -> None:
    """Called inside worker processes; `run_worker` parses these lines."""
    print("@@" + json.dumps({"event": event, **data}, default=str), flush=True)


@router.get("")
def list_jobs(kind: str | None = None):
    jobs = [j.summary() for j in registry.jobs.values() if kind is None or j.kind == kind]
    return sorted(jobs, key=lambda j: j["created"], reverse=True)


@router.get("/{job_id}")
def get_job(job_id: str):
    return registry.get(job_id).summary()


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    job = registry.get(job_id)
    registry.cancel(job)
    return job.summary()


@router.get("/{job_id}/events")
async def stream_events(job_id: str, after: int = -1):
    """Server-sent events: every event after `after`, then new ones until the job ends."""
    job = registry.get(job_id)

    async def gen():
        seq = after
        while True:
            with job.lock:
                fresh = job.events[seq + 1 :]
            for ev in fresh:
                seq = ev["seq"]
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            if job.status in TERMINAL and not fresh:
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
