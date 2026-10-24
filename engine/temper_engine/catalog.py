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


@router.get("/match")
def match_route(model: str):
    return {"match": match(model)}
