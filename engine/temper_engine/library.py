"""Models imported from your own files: a GGUF, a folder of GGUFs, a Hugging Face (safetensors) folder,
an MLX folder or a Mistral-format folder (params.json and consolidated.safetensors).

Temper remembers where they are rather than copying them, since they're often many gigabytes; the list
lives in library.json. Each import is checked when added and again whenever the library is listed, so
a file that has moved shows as missing instead of failing to load.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import engines, home
from .hub import SPLIT, group_gguf
from .memory import read_gguf_metadata

router = APIRouter(prefix="/api/models/import", tags=["models"])

WEIGHTS = ("*.safetensors", "*.bin")
TOKENIZER = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")
MISTRAL_TOKENIZER = ("tekken.json", "tokenizer.model.v3", "tokenizer.model.v7", "tokenizer.model")
# Architectures whose MXFP4 weights llama.cpp's converter repacks; for any other model it has no reader.
MXFP4_ARCHITECTURES = {"GptOssForCausalLM", "DeepseekV4ForCausalLM", "KimiK3ForConditionalGeneration", "MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"}


def _file() -> Path:
    return home() / "library.json"


def _imports() -> list[dict]:
    try:
        return json.loads(_file().read_text()).get("imports", [])
    except (OSError, ValueError):
        return []


def _save(imports: list[dict]) -> None:
    _file().write_text(json.dumps({"imports": imports}, indent=2))


def _gguf_name(path: Path) -> str:
    """The model's own name from the GGUF header ("" if it has none). Raises ValueError for anything else."""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF")
        f.seek(0)
        try:
            meta = read_gguf_metadata(f.read)
        except struct.error as e:  # the header ends early
            raise OSError("the file is cut short") from e
    return meta.get("general.name") or meta.get("general.basename") or ""


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _ggufs(folder: Path, only: Path | None = None) -> list[dict]:
    """The GGUFs in a folder as library entries; with `only`, just that file (and its split parts)."""
    files = [{"name": p.name, "size": p.stat().st_size, "path": str(p)} for p in sorted(folder.glob("*.gguf")) if _is_gguf(p)]
    groups = group_gguf(files)
    if only:
        key = SPLIT.sub(".gguf", only.name)
        groups = [g for g in groups if SPLIT.sub(".gguf", g["name"]) == key]
    return groups


def _name(path: Path) -> str:
    """A folder's or file's name; for something inside the Hugging Face cache, the repo's name."""
    for part in reversed(path.parts):
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return path.stem if path.is_file() else path.name


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def quantization(config: dict, folder: Path) -> dict | None:
    """How a checkpoint was quantized before it was published, and whether llama.cpp's converter can read it.

    Returns {"label", "convertible", "note"?}, or None for plain 16/32-bit weights. The converter
    dequantizes FP8, GPTQ, BitNet, compressed-tensors (one config group, or NVFP4) and ModelOpt
    checkpoints, and keeps MXFP4 only for the few architectures that use it natively.
    """
    q = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config") or {}
    method, algo, fmt, groups = q.get("quant_method"), q.get("quant_algo"), q.get("format"), q.get("config_groups") or {}
    modelopt = _read_json(folder / "hf_quant_config.json")  # NVIDIA ModelOpt writes its settings here
    if modelopt:
        quant = modelopt.get("quantization") or {}
        method = method or ((modelopt.get("producer") or {}).get("name") or "").lower() or None
        algo = quant.get("quant_algo", algo)
    if not method:
        return None
    method = str(method).lower()
    nvfp4 = fmt == "nvfp4-pack-quantized" or (fmt == "mixed-precision" and any(isinstance(g, dict) and g.get("format") == "nvfp4-pack-quantized" for g in groups.values()))
    weights = (next(iter(groups.values()), {}) or {}).get("weights") or {} if groups else {}
    bits = weights.get("num_bits") or q.get("bits")

    def no(label: str, why: str) -> dict:
        return {"label": label, "convertible": False, "note": why}

    if method == "fp8":
        return {"label": "FP8", "convertible": True}
    if method == "modelopt":
        return {"label": f"{'NVFP4' if algo and 'NVFP4' in str(algo) else algo or 'FP8'} (ModelOpt)", "convertible": True}
    if method == "compressed-tensors":
        if nvfp4:
            return {"label": "NVFP4 (compressed-tensors)", "convertible": True}
        kind = {"float-quantized": "FP", "int-quantized": "INT", "naive-quantized": "INT", "pack-quantized": "INT"}.get(fmt)
        label = f"{kind}{bits or ''} (compressed-tensors)" if kind else f"{fmt or 'compressed-tensors'} (compressed-tensors)"
        if not kind:
            return no(label, f"llama.cpp's converter doesn't read the {fmt!r} compressed-tensors format.")
        if len(groups) > 1:
            return no(label, "This checkpoint mixes several quantization schemes, which llama.cpp's converter can't read yet.")
        return {"label": label, "convertible": True}
    if method == "mxfp4":
        if set(config.get("architectures") or []) & MXFP4_ARCHITECTURES:
            return {"label": "MXFP4", "convertible": True}
        return no("MXFP4", "llama.cpp's converter reads MXFP4 weights only for gpt-oss, DeepSeek V4, Kimi K3 and MiMo V2.")
    if method == "gptq":
        if bits == 3:
            return no("GPTQ 3-bit", "llama.cpp's converter can't read 3-bit GPTQ weights.")
        return {"label": f"GPTQ{f' {bits}-bit' if bits else ''}", "convertible": True}
    if method == "bitnet":
        return {"label": "BitNet", "convertible": True}
    label = {"awq": "AWQ", "bitsandbytes": "bitsandbytes", "hqq": "HQQ", "eetq": "EETQ", "torchao": "torchao", "quark": "Quark"}.get(method, method)
    return no(label, f"llama.cpp's converter can't read {label} weights. Use the model's original, unquantized release, or a GGUF of it.")


def checkpoint(folder: Path) -> dict:
    """For a model folder: its pre-quantization, and whether Temper can convert it to a GGUF for chat."""
    config_file = folder / "config.json"
    if not config_file.is_file():
        mistral = (folder / "params.json").is_file() and any(folder.glob("consolidated*.safetensors"))
        return {"quantization": None, "convertible": mistral}
    config = _read_json(config_file)
    if "quantization" in config:  # MLX's own format, which only MLX reads
        return {"quantization": None, "convertible": False}
    quant = quantization(config, folder)
    return {"quantization": quant, "convertible": quant is None or quant["convertible"]}


def inspect(raw: str) -> dict:
    """What kind of model is at `raw`, or a 400 saying why it can't be used."""
    # absolute, not resolve(): files in the Hugging Face cache are symlinks to extensionless blobs.
    path = Path(os.path.abspath(Path(raw).expanduser()))
    if not path.exists():
        raise HTTPException(400, f"Temper can't find {path}.")
    if (path / "snapshots").is_dir() and (path / "refs" / "main").is_file():  # a cache repo folder: use its current snapshot
        path = path / "snapshots" / (path / "refs" / "main").read_text().strip()
    if path.is_file():
        # Ollama's blobs are GGUFs without the extension, so the header decides.
        if path.suffix.lower() != ".gguf" and not _is_gguf(path):
            raise HTTPException(400, "Import a .gguf file, or a folder that holds a model (config.json and .safetensors files).")
        try:
            name = _gguf_name(path)
        except ValueError as e:
            raise HTTPException(400, f"{path.name} isn't a GGUF file, despite its name.") from e
        except OSError as e:
            raise HTTPException(400, f"{path.name} looks damaged: Temper couldn't read its header ({e}).") from e
        if "mmproj" in path.name.lower():
            raise HTTPException(400, f"{path.name} is a vision projector, not a model. Import the model's GGUF; Temper finds the projector beside it.")
        if path.suffix.lower() == ".gguf":
            gguf = _ggufs(path.parent, only=path)
        else:
            size = path.stat().st_size
            gguf = [{"name": path.name, "size": size, "path": str(path), "parts": [path.name], "split": 1, "complete": True}]
        return {"kind": "gguf", "name": name or _name(path), "path": str(path), "gguf": gguf, "size": sum(g["size"] for g in gguf)}

    config_file = path / "config.json"
    if config_file.is_file():
        try:
            config = json.loads(config_file.read_text())
        except ValueError as e:
            raise HTTPException(400, f"{path.name}/config.json isn't valid JSON.") from e
        weights = [p for pattern in WEIGHTS for p in path.glob(pattern)]
        if not weights:
            raise HTTPException(400, f"{path.name} has a config.json but no .safetensors weights.")
        kind = "mlx" if "quantization" in config else "safetensors"
        warnings = [] if any((path / t).is_file() for t in TOKENIZER) else ["There are no tokenizer files in this folder, so training and export will fail."]
        quant = quantization(config, path) if kind == "safetensors" else None
        if quant and not quant["convertible"]:
            warnings.append(quant["note"])
        return {
            "kind": kind, "name": _name(path), "path": str(path), "size": _folder_size(path), "model_type": config.get("model_type"),
            "quantization": quant, "convertible": kind == "safetensors" and (quant is None or quant["convertible"]), "warnings": warnings,
            "chat": engines.chat_model(str(path)),
        }

    gguf = _ggufs(path)
    if gguf:
        return {"kind": "gguf", "name": _name(path), "path": str(path), "gguf": gguf, "size": sum(g["size"] for g in gguf)}
    if (path / "params.json").is_file():
        if not any(path.glob("consolidated*.safetensors")):
            raise HTTPException(400, f"{path.name} has Mistral's params.json but no consolidated.safetensors weights.")
        warnings = [] if any((path / t).is_file() for t in MISTRAL_TOKENIZER) else ["There's no tokenizer (tekken.json or tokenizer.model) in this folder, so it can't be converted."]
        return {"kind": "mistral", "name": _name(path), "path": str(path), "size": _folder_size(path), "quantization": None, "convertible": not warnings, "warnings": warnings}
    raise HTTPException(400, f"{path.name} doesn't hold a model: Temper looks for .gguf files, or config.json with .safetensors weights.")


def _folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def entries() -> list[dict]:
    """Imported models in the shape of /api/models/local, re-checked so moved files show as missing."""
    out = []
    for item in _imports():
        try:
            found = inspect(item["path"])
        except HTTPException as e:
            out.append({"id": item["name"], "path": item["path"], "kind": item["kind"], "imported": True, "missing": e.detail, "size": 0, "last_used": item["added"], "gguf": [], "files": 0, "safetensors": False})
            continue
        gguf = found.get("gguf", [])
        if found["kind"] == "gguf" and not found["path"].lower().endswith(".gguf"):
            gguf = [{**g, "name": item["name"]} for g in gguf]  # a blob named by its hash: show the import's name
        out.append({
            "id": item["name"],
            "path": found["path"],
            "kind": found["kind"],
            "imported": True,
            "size": found["size"],
            "last_used": item["added"],
            "gguf": gguf,
            "files": len(gguf) or 1,
            "safetensors": found["kind"] in ("safetensors", "mlx"),
            "warnings": found.get("warnings", []),
            "quantization": found.get("quantization"),
            "convertible": found.get("convertible", False),
            "source": (item.get("settings") or {}).get("source"),
        })
    return out


class ImportRequest(BaseModel):
    path: str
    # Referenced where it is (the default), or copied or moved into Temper's imports folder first,
    # for a model on a drive that won't stay plugged in or in a folder that gets cleaned out.
    mode: Literal["link", "copy", "move"] = "link"


def imports_dir() -> Path:
    return home() / "imports"


def bring(path: str, mode: str) -> str:
    """Copies or moves a model into the imports folder, with a split GGUF's other parts and any projector
    beside a single GGUF, and returns its new path."""
    from .storage import models_dir

    src = Path(path)
    if mode == "move" and src.is_relative_to(models_dir()):
        raise HTTPException(400, f"{src.name} is in Temper's models folder already; moving it would break the download.")
    if src.is_dir():
        items = [src]
    else:
        key = SPLIT.sub(".gguf", src.name)
        siblings = [p for p in src.parent.glob("*.gguf") if p != src and (SPLIT.sub(".gguf", p.name) == key or "mmproj" in p.name.lower())]
        items = [src, *siblings]
    dest = imports_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / src.name).exists():
        raise HTTPException(409, f"{src.name} is already in Temper's imports folder ({dest}).")
    for item in items:
        if (dest / item.name).exists():
            continue  # a projector another import brought already

        if mode == "move":
            shutil.move(str(item), dest / item.name)
        elif item.is_dir():
            shutil.copytree(item, dest / item.name)
        else:
            shutil.copy2(item, dest / item.name)
    return str(dest / src.name)


def add_entry(path: str, name: str | None = None, settings: dict | None = None) -> dict:
    """Adds (or refreshes) an import. `settings` are chat settings that came with it, e.g. from Ollama."""
    found = inspect(path)
    imports = [i for i in _imports() if i["path"] != found["path"]]
    entry = {"name": name or found["name"], "path": found["path"], "kind": found["kind"], "added": time.time()}
    if settings:
        entry["settings"] = settings
    imports.insert(0, entry)
    _save(imports)
    return {**found, "name": entry["name"]}


def settings_for(path: str) -> tuple[str, dict] | None:
    """(name, settings) for an imported model file that came with chat settings."""
    target = os.path.abspath(path)
    for item in _imports():
        if item.get("settings") and os.path.abspath(item["path"]) == target:
            return item["name"], item["settings"]
    return None


@router.post("")
def add(req: ImportRequest):
    if req.mode == "link":
        return add_entry(req.path)
    found = inspect(req.path)  # checked before anything is copied
    return add_entry(bring(found["path"], req.mode))


@router.delete("")
def remove(path: str):
    """Forgets an import. The files stay where they are."""
    _save([i for i in _imports() if i["path"] != path])
    return {"ok": True}
