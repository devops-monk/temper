"""Image generation with diffusers: text to image, editing an image from a prompt, LoRAs, and a gallery.

diffusers and PyTorch install on request into their own environment under engines/image, never touching
the engine's. image_worker.py keeps the pipeline loaded between requests (speech.Worker), on CUDA, Apple's
Metal or the CPU. The models are diffusers folders from Hugging Face, downloaded without the files a
pipeline doesn't load (safety-checker weights, ONNX exports, single-file checkpoints). Images are saved
as PNGs under images/ in Temper's home, each with the settings that made it.

Each generation is a job, so the app shows its progress and the image survives the window closing.
OpenAI's /v1/images/generations answers with base64 PNGs.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import engines, home, prefs, storage
from .jobs import Job, registry
from .speech import Worker

router = APIRouter(tags=["images"])

PACKAGES = [engines.TORCH, "diffusers>=0.32", "transformers>=4.44", "accelerate", "safetensors", "peft", "pillow"]
WORKER = Path(__file__).with_name("image_worker.py")
_CONFIGS = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "feature_extractor/*", "*/config.json"]

MODELS = [
    {"id": "SimianLuo/LCM_Dreamshaper_v7", "name": "LCM Dreamshaper v7", "size": 4_270_000_000, "license": "MIT", "size_px": 512, "steps": 4, "guidance": 8.0,
     "about": "Four steps to a 512-pixel image; quick on a Mac.", "variant": None,
     "patterns": [*_CONFIGS, "text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]},
    {"id": "stabilityai/sd-turbo", "name": "SD-Turbo", "size": 2_580_000_000, "license": "Stability AI Non-Commercial", "size_px": 512, "steps": 1, "guidance": 0.0,
     "about": "One step; the fastest. Its licence allows non-commercial use only.", "variant": "fp16", "patterns": [*_CONFIGS, "*/*.fp16.safetensors"]},
    {"id": "segmind/SSD-1B", "name": "SSD-1B", "size": 4_470_000_000, "license": "Apache-2.0", "size_px": 1024, "steps": 20, "guidance": 7.0,
     "about": "1024-pixel images in the SDXL style; slower, and best with a GPU.", "variant": "fp16", "patterns": [*_CONFIGS, "*/*.fp16.safetensors"]},
]


def venv() -> Path:
    return home() / "engines" / "image"


def python() -> str | None:
    p = engines._venv_python(venv())
    return str(p) if p.is_file() and (venv() / ".installed").is_file() else None


def model_folder(model: str) -> str | None:
    """The downloaded diffusers folder for a model id."""
    snapshots = storage.models_dir() / ("models--" + model.replace("/", "--")) / "snapshots"
    found = sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if snapshots.is_dir() else []
    return next((str(s) for s in found if (s / "model_index.json").is_file() and (s / "unet").is_dir()), None)


def info(model: str) -> dict:
    found = next((m for m in MODELS if m["id"] == model), None)
    if found is None:
        raise HTTPException(400, f"Pick one of: {', '.join(m['id'] for m in MODELS)}.")
    return found


def chosen() -> str:
    picked = prefs.load().get("image_model")
    return picked if any(m["id"] == picked for m in MODELS) else MODELS[0]["id"]


def status() -> dict:
    return {"installed": python() is not None, "model": chosen(), "ready": python() is not None and model_folder(chosen()) is not None, "running": worker.running(),
            "models": [{k: v for k, v in m.items() if k != "patterns"} | {"downloaded": model_folder(m["id"]) is not None} for m in MODELS]}


worker = Worker("image", lambda: [python(), "-I", "-u", str(WORKER)])


# --- The gallery -------------------------------------------------------------------------------------


def gallery_dir() -> Path:
    path = home() / "images"
    path.mkdir(exist_ok=True)
    return path


def gallery() -> list[dict]:
    out = []
    for meta in gallery_dir().glob("*.json"):
        try:
            out.append(json.loads(meta.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda i: i["created"], reverse=True)


# --- Making images ---------------------------------------------------------------------------------


class Generate(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative: str | None = None
    model: str | None = None
    steps: int | None = Field(None, ge=1, le=100)
    guidance: float | None = Field(None, ge=0, le=30)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    seed: int | None = None
    count: int = Field(1, ge=1, le=8)
    image: str | None = None  # a picture to change: a file path, or a gallery image's id
    strength: float = Field(0.6, gt=0, le=1)
    lora: str | None = None
    lora_scale: float = Field(1.0, ge=0, le=2)


def _check(model: str) -> None:
    info(model)
    if python() is None:
        raise HTTPException(409, "Image generation isn't installed. Install it on the Images page, or run `temper image --install`.")
    if model_folder(model) is None:
        raise HTTPException(409, f"{info(model)['name']} isn't downloaded. Download it on the Images page.")


def _source(image: str | None) -> str | None:
    if not image:
        return None
    if re.fullmatch(r"[0-9a-f]{12}", image) and (gallery_dir() / f"{image}.png").is_file():
        return str(gallery_dir() / f"{image}.png")
    if not Path(image).is_file():
        raise HTTPException(404, f"No image at {image}.")
    return image


def _lora(lora: str | None) -> None:
    if lora and not Path(lora).exists():
        raise HTTPException(404, f"No LoRA at {lora}.")


def make(req: Generate, on_progress=None) -> list[dict]:
    """Runs the request in the worker; returns the gallery entries it made."""
    model = req.model or chosen()
    m = info(model)
    _check(model)
    source = _source(req.image)
    _lora(req.lora)
    side = m["size_px"]
    ids = [uuid.uuid4().hex[:12] for _ in range(req.count)]
    settings = {"prompt": req.prompt.strip(), "negative": (req.negative or "").strip() or None, "steps": req.steps or m["steps"],
                "guidance": m["guidance"] if req.guidance is None else req.guidance, "width": req.width or (None if source else side),
                "height": req.height or (None if source else side), "seed": req.seed, "count": req.count, "image": source,
                "strength": req.strength if source else None, "lora": req.lora, "lora_scale": req.lora_scale if req.lora else None}
    payload = {**settings, "model": model_folder(model), "variant": m["variant"], "out": [str(gallery_dir() / f"{i}.png") for i in ids]}
    try:
        done = worker.request(payload, lambda msg: on_progress and "progress" in msg and on_progress(msg["progress"]))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    made = []
    for i, seed in zip(ids, done["seeds"], strict=True):
        entry = {"id": i, "model": model, **{k: v for k, v in settings.items() if k != "count"}, "seed": seed, "created": time.time(), "path": str(gallery_dir() / f"{i}.png")}
        (gallery_dir() / f"{i}.json").write_text(json.dumps(entry, ensure_ascii=False))
        made.append(entry)
    return made


# --- Routes ------------------------------------------------------------------------------------------


@router.get("/api/images")
def images_status():
    return status()


@router.post("/api/images/install")
def install():
    job = registry.create("engine-install", "Install image generation (diffusers)")
    registry.run_thread(job, lambda j: engines.install_venv(j, venv(), PACKAGES, "image generation"))
    return job.summary()


@router.delete("/api/images")
def remove():
    worker.stop()
    shutil.rmtree(venv(), ignore_errors=True)
    return status()


class Choice(BaseModel):
    model: str


@router.post("/api/images/download")
def download(req: Choice):
    """Downloads a model with only the files the pipeline loads."""
    from .hub import DownloadRequest
    from .hub import download as hub_download

    m = info(req.model)
    return hub_download(DownloadRequest(repo_id=m["id"], patterns=m["patterns"]))


@router.put("/api/images/model")
def choose(req: Choice):
    info(req.model)
    prefs.update("image_model", req.model)
    return status()


@router.post("/api/images/generate")
def generate(req: Generate):
    _check(req.model or chosen())
    _source(req.image)
    _lora(req.lora)
    job = registry.create("image", f"Image: {req.prompt.strip()[:60]}")
    job.progress = 0.0

    def run(j: Job) -> None:
        def progress(value: float) -> None:
            j.progress = value
            j.emit("progress", value=value)

        made = make(req, progress)
        j.result.update(images=[m["id"] for m in made])

    registry.run_thread(job, run)
    return job.summary()


@router.get("/api/images/gallery")
def list_gallery():
    return {"images": gallery()}


def _image(image: str) -> Path:
    path = gallery_dir() / f"{image}.png"
    if not re.fullmatch(r"[0-9a-f]{12}", image) or not path.is_file():
        raise HTTPException(404, "No such image.")
    return path


@router.get("/api/images/gallery/{image}.png")
def image_file(image: str):
    return FileResponse(_image(image), media_type="image/png")


@router.delete("/api/images/gallery/{image}")
def delete_image(image: str):
    _image(image).unlink()
    (gallery_dir() / f"{image}.json").unlink(missing_ok=True)
    return {"images": gallery()}


class OpenAIImage(BaseModel):
    prompt: str
    n: int = Field(1, ge=1, le=8)
    size: str | None = None
    model: str | None = None
    response_format: str = "b64_json"


@router.post("/v1/images/generations")
def v1_images(req: OpenAIImage):
    """OpenAI's image API. `size` is WIDTHxHEIGHT; `model` may name one of MODELS, anything else uses the chosen one."""
    width = height = None
    if req.size and (m := re.fullmatch(r"(\d+)x(\d+)", req.size)):
        width, height = int(m.group(1)), int(m.group(2))
    model = req.model if any(x["id"] == req.model for x in MODELS) else None
    made = make(Generate(prompt=req.prompt, count=req.n, width=width, height=height, model=model))
    return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(Path(m["path"]).read_bytes()).decode(), "revised_prompt": m["prompt"]} for m in made]}
