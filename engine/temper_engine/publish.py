"""Publishing to Hugging Face: a fine-tune (its export or its adapter) or a prepared dataset, with a card.

A model's card is written from the run that made it: the base model, the data and its SHA-256, the
method, settings and results, and the library versions, so whoever finds it knows what it is and how
it was made. Repositories are private unless asked otherwise, and nothing is uploaded except on an
explicit request from the app.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import __version__, home
from .jobs import Job, registry

router = APIRouter(prefix="/api/publish", tags=["export"])

SKIP = {"export.json", "status.json", "events.jsonl", "lora_config.yaml", "samples.json"}
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?")


def _run(run_id: str) -> dict:
    from .training import get_run

    return get_run(run_id)


def files_to_upload(folder: Path) -> list[Path]:
    """What goes up from an export or adapter folder: the weights and their configs, not Temper's own bookkeeping."""
    out = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.name in SKIP or p.suffix == ".Modelfile" or p.name.startswith("."):
            continue
        # mlx-lm's numbered checkpoints stay behind; the final adapter goes.
        if re.fullmatch(r"\d{7}_adapters\.safetensors", p.name) or "checkpoints" in p.relative_to(folder).parts:
            continue
        out.append(p)
    return out


def _yaml_list(key: str, items: list[str]) -> list[str]:
    return [f"{key}:"] + [f"- {i}" for i in items] if items else []


def model_card(run: dict, repo: str, files: list[str]) -> str:
    cfg = run.get("config") or {}
    prov = run.get("provenance") or {}
    base = run.get("base_model") or cfg.get("base_model") or ""
    gguf = any(f.endswith(".gguf") for f in files)
    adapter = any(f.endswith(("adapters.safetensors", "adapter_model.safetensors")) for f in files)
    objective = cfg.get("objective") or "sft"
    dataset = cfg.get("dataset") or ""
    hub_dataset = dataset if dataset and not Path(dataset).expanduser().exists() and "/" in dataset else None
    tags = ["temper", "gguf" if gguf else None, "lora" if adapter or cfg.get("method") in ("lora", "qlora") else None, "mlx" if run.get("backend") == "mlx" else None, objective if objective != "sft" else None]
    front = ["---", f"base_model: {base}"] + _yaml_list("datasets", [hub_dataset] if hub_dataset else []) + _yaml_list("tags", [t for t in tags if t])
    if adapter and run.get("backend") != "mlx":
        front.append("library_name: peft")
    front.append("---")

    what = {"sft": "fine-tuned", "dpo": "preference-trained (DPO)", "orpo": "preference-trained (ORPO)", "simpo": "preference-trained (SimPO)"}[objective]
    body = [f"# {repo.split('/')[-1]}", "", f"A {what} version of [{base}](https://huggingface.co/{base}), made with Temper."]
    if gguf:
        body += ["", "## Use it", "", "The GGUF files run in llama.cpp, Ollama, LM Studio and Temper:", "", "```sh", f"llama-server -hf {repo}", "```"]
    elif adapter and run.get("backend") == "mlx":
        body += ["", "## Use it", "", "A LoRA adapter for mlx-lm, on top of the base model:", "", "```sh", f"mlx_lm.generate --model {base} --adapter-path <this repo, downloaded> --prompt \"Hello\"", "```"]

    rows = [
        ("Base model", base + (f" (revision `{prov['base_revision'][:12]}`)" if prov.get("base_revision") else "")),
        ("Training data", (hub_dataset or Path(dataset).name or "—") + (f" (SHA-256 `{run['dataset_sha256'][:16]}…`)" if run.get("dataset_sha256") else "")),
        ("Rows", (run.get("dataset_info") or {}).get("rows")),
        ("Objective", {"sft": "Supervised fine-tuning"}.get(objective, objective.upper())),
        ("Method", f"{cfg.get('method')} on {'Apple Silicon (MLX)' if run.get('backend') == 'mlx' else 'GPU'}"),
        ("LoRA rank / alpha", f"{cfg.get('lora_r')} / {cfg.get('lora_alpha')}" if cfg.get("method") != "full" else None),
        ("Steps", run.get("steps") or None),
        ("Learning rate", cfg.get("learning_rate")),
        ("Batch size × accumulation", f"{cfg.get('batch_size')} × {cfg.get('grad_accum')}" if cfg.get("batch_size") else None),
        ("Context length", cfg.get("max_seq_length")),
        ("Final training loss", f"{run['loss']:.4f}" if run.get("loss") is not None else None),
        ("Final validation loss", f"{run['val_loss']:.4f}" if run.get("val_loss") is not None else None),
        ("Libraries", ", ".join(f"{k} {v}" for k, v in (prov.get("libraries") or {}).items()) or None),
    ]
    body += ["", "## How it was trained", "", "| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows if v not in (None, "")]
    body += ["", f"Made with Temper {__version__}."]
    return "\n".join(front + body) + "\n"


def dataset_card(repo: str, path: Path, rows: int, eval_rows: int, fmt: str | None) -> str:
    front = ["---", "tags:", "- temper", "configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: data/train.jsonl"]
    if eval_rows:
        front += ["  - split: test", "    path: data/test.jsonl"]
    front.append("---")
    body = [f"# {repo.split('/')[-1]}", "", f"{rows} rows" + (f" plus {eval_rows} held out for evaluation" if eval_rows else "") + (f", in {fmt} format" if fmt else "") + ".", "", f"Prepared with Temper {__version__} from `{path.name}`."]
    return "\n".join(front + body) + "\n"


def _lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _upload(job: Job, repo: str, kind: str, private: bool, files: list[tuple[Path | bytes, str]]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    job.log(f"Creating {'a private' if private else 'a public'} {kind} repository {repo}")
    url = api.create_repo(repo, repo_type=kind, private=private, exist_ok=True)
    for i, (src, target) in enumerate(files):
        if job.cancel_requested:
            return
        job.progress = i / len(files)
        job.emit("progress", value=job.progress, file=target)
        job.log(f"Uploading {target}")
        api.upload_file(path_or_fileobj=src, path_in_repo=target, repo_id=repo, repo_type=kind, commit_message=f"Upload {target} with Temper")
    job.finish("done", url=str(url), repo=repo)


def _repo_id(name: str) -> str:
    from huggingface_hub import HfApi

    if not REPO.fullmatch(name):
        raise HTTPException(422, "A repository name is letters, numbers, dots, dashes and underscores, like my-model or your-name/my-model.")
    if "/" in name:
        return name
    try:
        return f"{HfApi().whoami()['name']}/{name}"
    except Exception as e:  # noqa: BLE001 — no token, or a bad one
        raise HTTPException(401, "Add a Hugging Face token with write access in Settings first.") from e


class PublishModel(BaseModel):
    run_id: str | None = None  # found from the export when one is given
    repo: str
    export: str | None = None  # an export folder; without one, the run's adapter
    private: bool = True


@router.post("/model")
def publish_model(req: PublishModel):
    if req.export:
        folder = Path(req.export).expanduser()
        if (home() / "exports").resolve() not in folder.resolve().parents or not (folder / "export.json").is_file():
            raise HTTPException(400, "That isn't one of Temper's exports.")
        run = _run(req.run_id or json.loads((folder / "export.json").read_text())["run_id"])
    elif req.run_id:
        run = _run(req.run_id)
        folder = Path(run["path"]) / "adapter"
    else:
        raise HTTPException(422, "Say which run or export to publish.")
    if not folder.is_dir():
        raise HTTPException(404, "There's nothing to publish there yet.")
    files = files_to_upload(folder)
    if not files:
        raise HTTPException(409, "The folder has no model files to publish.")
    repo = _repo_id(req.repo)
    # GGUFs at the top, where llama.cpp's -hf and Ollama's hf.co/ look for them.
    names = [p.name if p.suffix == ".gguf" else p.relative_to(folder).as_posix() for p in files]
    card = model_card(run, repo, names).encode()
    job = registry.create("publish", f"Publish {repo}")
    registry.run_thread(job, lambda j: _upload(j, repo, "model", req.private, [(card, "README.md"), *zip(files, names)]))
    return job.summary()


class PublishDataset(BaseModel):
    path: str
    repo: str
    private: bool = True


@router.post("/dataset")
def publish_dataset(req: PublishDataset):
    from .formats import detect_format

    path = Path(req.path).expanduser()
    if not path.is_file() or path.suffix != ".jsonl":
        raise HTTPException(404, "Publishing takes a prepared .jsonl dataset.")
    held = path.with_suffix(".eval.jsonl")
    eval_rows = _lines(held) if held.is_file() else 0
    with open(path, encoding="utf-8") as f:
        first = next((json.loads(line) for line in f if line.strip()), {})
    repo = _repo_id(req.repo)
    card = dataset_card(repo, path, _lines(path), eval_rows, detect_format(list(first)))
    files: list[tuple[Path | bytes, str]] = [(card.encode(), "README.md"), (path, "data/train.jsonl")]
    if eval_rows:
        files.append((held, "data/test.jsonl"))
    job = registry.create("publish", f"Publish {repo}")
    registry.run_thread(job, lambda j: _upload(j, repo, "dataset", req.private, files))
    return job.summary()
