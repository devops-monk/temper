"""The model catalog: curated families with each model's facts and its publisher's recommended settings.

catalog.json is built by scripts/build_catalog.py. This adds what depends on the machine: whether a
4-bit GGUF fits in memory, whether the installed trainer supports the architecture, and what's already
downloaded. It also finds the catalog entry for whichever model chat has loaded.
"""

from __future__ import annotations

import importlib.util
import json
import re
from functools import cache
from struct import error as struct_error
from pathlib import Path

from fastapi import APIRouter

from . import memory
from .downloads import repo_cache
from .hardware import probe

router = APIRouter(prefix="/api/catalog", tags=["catalog"])

Q4_BITS = 4.85  # Q4_K_M, the quant most people pick


@cache
def load() -> dict:
    return json.loads((Path(__file__).parent / "catalog.json").read_text())


def _package_dir(name: str) -> Path | None:
    # find_spec on the top-level package finds its folder without importing it (mlx_lm imports mlx).
    spec = importlib.util.find_spec(name)
    return Path(next(iter(spec.submodule_search_locations))) if spec and spec.submodule_search_locations else None


@cache
def _supported(trainer: str, model_type: str) -> bool:
    if trainer == "mlx":
        root = _package_dir("mlx_lm")
        return bool(root and (root / "models" / f"{model_type}.py").is_file())
    root = _package_dir("transformers")
    return bool(root and (root / "models" / model_type).is_dir())


def _downloaded(repo: str | None) -> bool:
    return bool(repo) and (repo_cache(repo) / "snapshots").is_dir()


def chat_fit(params_b: float, budget: dict) -> dict:
    """A rough fit for a Q4_K_M GGUF at an 8K context: weights, plus about 15% and 0.6 GB for the KV cache
    and llama.cpp's buffers. The quant picker reads the real file for an exact answer."""
    total = int(params_b * 1e9 * Q4_BITS / 8 * 1.15 + 0.6e9)
    return {"total": total, "verdict": memory.verdict(total, budget)}


@router.get("")
def catalog():
    budget = memory.memory_budget()
    trainer = probe()["capabilities"]["trainer"]
    families = []
    for family in load()["families"]:
        variants = []
        for v in family["variants"]:
            train_repo = (v["mlx"] or v["id"]) if trainer == "mlx" else v["id"]
            variants.append({
                **v,
                "fit": chat_fit(v["params"], budget),
                "trainable": bool(trainer and v["model_type"] and _supported(trainer, v["model_type"])),
                "train_repo": train_repo,
                "downloaded": {"gguf": _downloaded(v["gguf"]), "train": _downloaded(train_repo)},
            })
        # The largest model that fits comfortably is the one to suggest.
        fitting = [x for x in variants if x["fit"]["verdict"] == "fits"]
        suggested = max(fitting, key=lambda x: x["params"])["id"] if fitting else None
        families.append({**family, "variants": variants, "suggested": suggested})
    return {"families": families, "budget": budget, "trainer": trainer}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower())


def match(model: str) -> dict | None:
    """The catalog entry for a loaded model, from its repo id, cache path or file name.

    The longest matching name wins, so "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" is Qwen3 4B Instruct 2507,
    not Qwen3 4B. A name must end at a separator, so "qwen3.5-4b" doesn't match "qwen3.5-40b".
    """
    haystack = _key(model.replace("models--", "").replace("--", "/"))
    best, best_len = None, 0
    for family in load()["families"]:
        for v in family["variants"]:
            for repo in filter(None, (v["id"], v["gguf"], v["mlx"])):
                name = _key(repo.split("/")[1]).removesuffix("-gguf").removesuffix("-4bit")
                for m in re.finditer(re.escape(name), haystack):
                    end = m.end()
                    if (end == len(haystack) or not haystack[end].isalnum()) and len(name) > best_len:
                        best, best_len = {"family": family["name"], "publisher": family["publisher"], **v}, len(name)
    return best


# Generation settings by the names chat uses (repetition_penalty is mapped onto repeat penalty there).
GGUF_SAMPLING = {"general.sampling.temp": "temperature", "general.sampling.top_p": "top_p", "general.sampling.top_k": "top_k", "general.sampling.min_p": "min_p", "general.sampling.penalty_repeat": "repetition_penalty"}
CONFIG_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
HF_URL = re.compile(r"huggingface\.co/([\w.-]+/[\w.-]+)")


def _tidy(sampling: dict) -> dict:
    # GGUF stores floats as float32, so 0.8 reads back as 0.800000011920929.
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sampling.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _gguf_meta(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return memory.read_gguf_metadata(f.read)
    except (OSError, ValueError, struct_error):
        return {}


def _repos(path: str, meta: dict) -> list[str]:
    """Hugging Face repos a model file may come from: its cache folder, then the base models it names."""
    repos = [part.removeprefix("models--").replace("--", "/") for part in Path(path).parts if part.startswith("models--")]
    i = 0
    while f"general.base_model.{i}.repo_url" in meta:
        if m := HF_URL.search(str(meta[f"general.base_model.{i}.repo_url"])):
            repos.append(m.group(1))
        i += 1
    return list(dict.fromkeys(repos))


@cache
def _hub_sampling(repo: str) -> tuple[str, dict] | None:
    """(repo, sampling) from generation_config.json of `repo`, or of the base model its card names."""
    import httpx
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    from . import sources

    if sources.modelscope():
        return None
    candidates = [repo]
    try:
        base = (HfApi().model_info(repo, timeout=8).card_data or {}).get("base_model")
        candidates += [base] if isinstance(base, str) else list(base or [])
    except Exception:  # offline, gated or deleted: fall back to the repo itself
        pass
    for candidate in dict.fromkeys(candidates):
        try:
            # Fetched rather than downloaded: hf_hub_download would leave a metadata-only "model" in the cache.
            r = httpx.get(hf_hub_url(candidate, "generation_config.json"), headers=build_hf_headers(), timeout=8, follow_redirects=True)
            r.raise_for_status()
            config = r.json()
        except Exception:
            continue
        sampling = _tidy({k: config[k] for k in CONFIG_SAMPLING if k in config})
        if {"temperature", "top_p", "top_k"} & sampling.keys():
            return candidate, sampling
    return None


def recommended_for(path: str) -> dict | None:
    """Settings for a model the catalog doesn't know: those its converter stored in the GGUF (from the
    original generation_config.json), else generation_config.json from Hugging Face."""
    meta = _gguf_meta(path) if path.lower().endswith(".gguf") or Path(path).is_file() else {}
    name = meta.get("general.name") or Path(path).name
    if sampling := _tidy({ours: meta[key] for key, ours in GGUF_SAMPLING.items() if key in meta}):
        return {"id": name, "label": name, "publisher": None, "source": "the settings stored in this GGUF", "sampling": sampling}
    for repo in _repos(path, meta):
        if found := _hub_sampling(repo):
            source_repo, sampling = found
            return {"id": source_repo, "label": source_repo, "publisher": None, "source": f"generation_config.json from {source_repo}", "sampling": sampling}
    return None


@router.get("/match")
def match_route(model: str):
    """Settings for a loaded model: those imported with it (Ollama's Modelfile) first, then the catalog's,
    then the ones it was published with."""
    from .library import settings_for

    if found := settings_for(model):
        name, settings = found
        return {"match": {"id": name, "label": name, "publisher": settings["source"], "sampling": settings.get("sampling") or {}, "system": settings.get("system")}}
    return {"match": match(model) or (recommended_for(model) if Path(model).exists() else None)}
