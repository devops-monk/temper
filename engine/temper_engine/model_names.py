"""Which local model a name means: a path, a GGUF's file name, or a model's name, as `temper load` and API
requests write them. Works on the rows of /api/models/local, so the CLI and the engine agree."""

from __future__ import annotations

from pathlib import Path


def pick(models: list[dict], wanted: str, quant: str | None = None, loose: bool = True) -> tuple[str, str]:
    """The path to load and its name. A model with GGUFs gives its Q4_K_M file unless `quant` says which.

    `loose` also accepts part of a model's name, for people typing; API requests match whole names only
    (`owner/name`, `name`, or a file name with or without .gguf), so a client's "gpt-4o" never loads a stranger.
    Raises LookupError, saying what was wrong.
    """
    if Path(wanted).expanduser().exists() and (loose or Path(wanted).is_absolute()):
        path = Path(wanted).expanduser()
        return str(path), path.name
    w = wanted.lower()
    for m in models:
        for g in m.get("gguf") or []:
            if w in (g["name"].lower(), g["name"].lower().removesuffix(".gguf")):
                return g["path"], g["name"]
    model = next((m for m in models if w in (m["id"].lower(), m["id"].lower().rsplit("/", 1)[-1])), None)
    if model is None and loose:
        model = next((m for m in models if w in m["id"].lower()), None)
    if model is None:
        raise LookupError(f"No model called {wanted!r} on this computer.")
    ggufs = [g for g in model.get("gguf") or [] if g.get("complete", True)]
    if quant:
        ggufs = [g for g in ggufs if quant.lower() in g["name"].lower()]
        if not ggufs:
            raise LookupError(f"{model['id']} has no {quant} file.")
    if ggufs:
        g = next((g for g in ggufs if "q4_k_m" in g["name"].lower()), ggufs[0])
        return g["path"], g["name"]
    if model.get("safetensors") and model.get("path") and model.get("chat", True):
        return model["path"], model["id"]
    raise LookupError(f"{model['id']} has nothing Temper can load.")
