"""Export API: turn a finished run into LoRA, merged weights or GGUF files."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import home
from .jobs import registry
from .training import runs_dir

router = APIRouter(prefix="/api/export", tags=["export"])

FORMATS = {"lora", "merged_16bit", "merged_4bit", "gguf"}
GGUF_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16", "q3_k_m", "q6_k", "q2_k"]


class ExportRequest(BaseModel):
    run_id: str
    formats: list[str] = ["gguf"]
    gguf_quants: list[str] = ["q4_k_m"]


@router.get("/options")
def options():
    return {"formats": sorted(FORMATS), "gguf_quants": GGUF_QUANTS}


@router.post("")
def start(req: ExportRequest):
    run = runs_dir() / req.run_id
    cfg_file = run / "config.json"
    if not cfg_file.is_file():
        raise HTTPException(404, f"No run {req.run_id}")
    if not (run / "adapter").is_dir():
        raise HTTPException(409, "This run has no saved adapter yet. Wait for training to finish.")
    bad = set(req.formats) - FORMATS
    if bad:
        raise HTTPException(400, f"Unknown formats: {', '.join(sorted(bad))}")

    cfg = json.loads(cfg_file.read_text())
    out = home() / "exports" / f"{req.run_id}-{time.strftime('%H%M%S')}"
    out.mkdir(parents=True)
    job_cfg = {**req.model_dump(), "backend": cfg["backend"], "base_model": cfg["base_model"], "adapter": str(run / "adapter"), "output_dir": str(out)}
    (out / "export.json").write_text(json.dumps(job_cfg, indent=2))

    job = registry.create("export", f"Export {req.run_id} → {', '.join(req.formats)}")
    job.result["output_dir"] = str(out)
    registry.run_worker(job, "temper_engine.export_worker", str(out / "export.json"))
    return job.summary()


@router.get("/list")
def list_exports():
    root = home() / "exports"
    if not root.is_dir():
        return []
    return [
        {"id": d.name, "path": str(d), "files": sorted(str(p.relative_to(d)) for p in d.rglob("*.gguf"))}
        for d in sorted(root.iterdir(), reverse=True)
        if d.is_dir()
    ]

