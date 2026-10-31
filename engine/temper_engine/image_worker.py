"""Makes images with diffusers, inside the image environment (images.py), one request at a time.

Run by path with that environment's Python, so it imports nothing from temper_engine. Each line on stdin
is a request: {"model": folder, "variant": "fp16" or null, "prompt", "negative", "steps", "guidance",
"width", "height", "seed", "count", "out": [paths], "image": path or null (image to image), "strength",
"lora": path or null, "lora_scale"}. It answers {"progress": 0..1} after each step, then
{"done": {"seeds": [...]}} or {"error": "..."}. The pipeline stays loaded while the model doesn't change.

It runs on CUDA, Apple's Metal (MPS) or the CPU, in half precision on a GPU.
"""

from __future__ import annotations

import json
import sys


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


_loaded: dict = {"key": None, "text": None, "image": None, "lora": None}


def device_and_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def pipelines(req: dict):
    """The text-to-image pipeline for the request's model, and an image-to-image one sharing its weights."""
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    key = (req["model"], req.get("variant"))
    if _loaded["key"] != key:
        _loaded.update(key=None, text=None, image=None, lora=None)
        device, dtype = device_and_dtype()
        kwargs = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False}
        if req.get("variant"):
            kwargs["variant"] = req["variant"]
        try:
            text = AutoPipelineForText2Image.from_pretrained(req["model"], **kwargs)
        except TypeError:  # pipelines without a safety checker (SDXL) don't take those arguments
            kwargs.pop("safety_checker")
            kwargs.pop("requires_safety_checker")
            text = AutoPipelineForText2Image.from_pretrained(req["model"], **kwargs)
        text.to(device)
        text.set_progress_bar_config(disable=True)
        if device == "mps":
            text.enable_attention_slicing()  # keeps peak memory down in unified memory
        _loaded.update(key=key, text=text, image=AutoPipelineForImage2Image.from_pipe(text))
    lora = req.get("lora")
    if lora != _loaded["lora"]:
        text = _loaded["text"]
        text.unload_lora_weights()
        if lora:
            text.load_lora_weights(lora, adapter_name="temper")
        _loaded["lora"] = lora
    return _loaded["text"], _loaded["image"]


def generate(req: dict) -> dict:
    import torch
    from PIL import Image

    text, image = pipelines(req)
    steps = int(req.get("steps") or 4)
    count = max(1, int(req.get("count") or 1))
    base = req.get("seed")
    base = int(base) if base is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
    seeds = []
    for i in range(count):
        seed = base + i
        seeds.append(seed)
        generator = torch.Generator("cpu").manual_seed(seed)

        def step(pipe, index, timestep, kwargs, i=i):
            send({"progress": round((i + (index + 1) / max(steps, 1)) / count, 3)})
            return kwargs

        args = {"prompt": req["prompt"], "num_inference_steps": steps, "generator": generator, "callback_on_step_end": step}
        if req.get("negative"):
            args["negative_prompt"] = req["negative"]
        if req.get("guidance") is not None:
            args["guidance_scale"] = float(req["guidance"])
        if req.get("lora"):
            args["cross_attention_kwargs"] = {"scale": float(req.get("lora_scale") or 1.0)}
        if req.get("image"):
            source = Image.open(req["image"]).convert("RGB")
            if req.get("width") and req.get("height"):
                source = source.resize((int(req["width"]), int(req["height"])))
            args["image"] = source
            args["strength"] = float(req.get("strength") or 0.6)
            out = image(**args).images[0]
        else:
            args["width"], args["height"] = int(req.get("width") or 512), int(req.get("height") or 512)
            out = text(**args).images[0]
        out.save(req["out"][i])
    return {"seeds": seeds}


def main() -> None:
    send({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            send({"done": generate(json.loads(line))})
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
