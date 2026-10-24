"""Data Recipes: build training datasets from documents, seed data and samplers with an LLM.

A recipe is a list of columns in NVIDIA data-designer's format (samplers, LLM text / structured /
code / judge columns, Jinja expressions and validators) plus Temper's own `seed`, `models` and
`providers` sections, which `recipe_worker.py` turns into a data-designer config.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home
from .jobs import registry
from .llamacpp import uv_binary, uv_env
from .runtime import manager

router = APIRouter(prefix="/api/recipes", tags=["recipes"])

# The worker's environment. Pinned so a recipe that worked keeps working.
REQUIREMENTS = ["data-designer==0.9.7", "pypdf==6.21.2", "python-docx==1.2.0"]
WORKER = Path(__file__).with_name("recipe_worker.py")


def recipes_dir() -> Path:
    path = home() / "recipes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    path = home() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "recipe"


# ------------------------------------------------------------------ templates

WRITER = [{"alias": "writer", "provider": "local", "model": "local", "temperature": 0.7, "max_tokens": 1024}]


def _template(id: str, name: str, description: str, columns: list[dict], seed: dict | None = None, processors: list[dict] | None = None) -> dict:
    return {"id": id, "name": name, "description": description, "seed": seed, "models": WRITER, "providers": [], "columns": columns, "processors": processors or []}


TEMPLATES = [
    _template(
        "document-qa",
        "Q&A from documents",
        "Split PDFs, Word files, web pages or a GitHub repo into chunks and write one grounded question and answer per chunk.",
        seed={"type": "documents", "sources": [], "chunk_size": 1200, "chunk_overlap": 200, "sampling": "ordered"},
        columns=[
            {
                "column_type": "llm-structured",
                "name": "qa",
                "drop": True,
                "model_alias": "writer",
                "prompt": "Read this passage:\n\n{{ chunk_text }}\n\nWrite one question that the passage answers, and the answer, using only facts from the passage.",
                "output_format": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question", "answer"],
                    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                },
            },
            {"column_type": "expression", "name": "instruction", "expr": "{{ qa.question }}"},
            {"column_type": "expression", "name": "output", "expr": "{{ qa.answer }}"},
        ],
        processors=[{"processor_type": "drop_columns", "name": "drop_sources", "column_names": ["chunk_text", "source_file"]}],
    ),
    _template(
        "topic-qa",
        "Questions on your topics",
        "Pick topics and difficulty levels; the model writes a realistic question and a helpful answer for each mix.",
        columns=[
            {"column_type": "sampler", "name": "topic", "drop": True, "sampler_type": "category", "params": {"values": ["personal finance", "home cooking", "gardening", "first aid"]}},
            {"column_type": "sampler", "name": "level", "drop": True, "sampler_type": "category", "params": {"values": ["beginner", "intermediate", "expert"], "weights": [0.5, 0.3, 0.2]}},
            {"column_type": "llm-text", "name": "instruction", "model_alias": "writer", "prompt": "Write one question a {{ level }} might ask about {{ topic }}. Reply with the question only."},
            {"column_type": "llm-text", "name": "output", "model_alias": "writer", "prompt": "Answer this question clearly and accurately for a {{ level }}:\n\n{{ instruction }}"},
        ],
    ),
    _template(
        "graded-answers",
        "Answers with a quality grade",
        "Like topic questions, plus a judge column that scores every answer so you can filter out weak rows.",
        columns=[
            {"column_type": "sampler", "name": "topic", "drop": True, "sampler_type": "category", "params": {"values": ["history", "physics", "programming", "health"]}},
            {"column_type": "llm-text", "name": "instruction", "model_alias": "writer", "prompt": "Write one specific question about {{ topic }}. Reply with the question only."},
            {"column_type": "llm-text", "name": "output", "model_alias": "writer", "prompt": "{{ instruction }}"},
            {
                "column_type": "llm-judge",
                "name": "grade",
                "model_alias": "writer",
                "prompt": "Grade the answer to the question.\n\nQuestion: {{ instruction }}\n\nAnswer: {{ output }}",
                "scores": [
                    {"name": "accuracy", "description": "Is the answer factually correct?", "options": {"1": "Wrong", "2": "Partly correct", "3": "Correct"}},
                    {"name": "helpfulness", "description": "Does it fully answer the question?", "options": {"1": "Not helpful", "2": "Somewhat", "3": "Fully"}},
                ],
            },
        ],
    ),
    _template(
        "text-to-sql",
        "Text to SQL",
        "Business questions paired with a table schema and a SQL answer, checked by a SQL linter.",
        columns=[
            {"column_type": "sampler", "name": "domain", "drop": True, "sampler_type": "category", "params": {"values": ["online shop", "hospital", "school", "logistics", "bank"]}},
            {"column_type": "llm-code", "name": "input", "model_alias": "writer", "code_lang": "sql:sqlite", "prompt": "Write CREATE TABLE statements (two or three tables) for a {{ domain }} database. SQL only."},
            {"column_type": "llm-text", "name": "instruction", "model_alias": "writer", "prompt": "Given this schema:\n\n{{ input }}\n\nWrite one question a manager would ask that needs a SQL query with a join or aggregate. Reply with the question only."},
            {"column_type": "llm-code", "name": "output", "model_alias": "writer", "code_lang": "sql:sqlite", "prompt": "Schema:\n\n{{ input }}\n\nWrite one SQLite query that answers: {{ instruction }}"},
            {"column_type": "validation", "name": "sql_check", "target_columns": ["output"], "validator_type": "code", "validator_params": {"code_lang": "sql:sqlite"}},
        ],
    ),
    _template(
        "python-tasks",
        "Python coding tasks",
        "Small programming tasks with working Python solutions, linted by a code validator.",
        columns=[
            {"column_type": "sampler", "name": "area", "drop": True, "sampler_type": "category", "params": {"values": ["string processing", "file handling", "data structures", "dates and times", "parsing"]}},
            {"column_type": "llm-text", "name": "instruction", "model_alias": "writer", "prompt": "Write a short, self-contained Python programming task about {{ area }}. Describe the input and expected output. Task only."},
            {"column_type": "llm-code", "name": "output", "model_alias": "writer", "code_lang": "python", "prompt": "Write a Python solution with a docstring for this task:\n\n{{ instruction }}"},
            {"column_type": "validation", "name": "code_check", "target_columns": ["output"], "validator_type": "code", "validator_params": {"code_lang": "python"}},
        ],
    ),
    _template(
        "instruction-from-text",
        "Instructions for existing text",
        "Start from your own texts (a CSV or JSONL file with a `text` column) and write the instruction each text answers.",
        seed={"type": "file", "path": "", "sampling": "ordered"},
        columns=[
            {"column_type": "llm-text", "name": "instruction", "model_alias": "writer", "prompt": "Here is a piece of writing:\n\n{{ text }}\n\nWrite the request or instruction that this writing would be a perfect reply to. Reply with the instruction only."},
            {"column_type": "expression", "name": "output", "expr": "{{ text }}"},
        ],
    ),
]


# ------------------------------------------------------------------ saved recipes


class Recipe(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    seed: dict | None = None
    models: list[dict] = Field(default_factory=list)
    providers: list[dict] = Field(default_factory=list)
    columns: list[dict]
    processors: list[dict] = Field(default_factory=list)
    template: str | None = None


@router.get("/templates")
def templates():
    return TEMPLATES


@router.get("")
def list_recipes():
    out = []
    for path in sorted(recipes_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            recipe = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({**recipe, "updated": path.stat().st_mtime})
    return out


@router.put("")
def save(recipe: Recipe):
    if not recipe.id:
        recipe.id = f"{slug(recipe.name)}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = recipes_dir() / f"{slug(recipe.id)}.json"
    path.write_text(json.dumps(recipe.model_dump(), indent=2, ensure_ascii=False))
    return recipe.model_dump()


@router.delete("/{recipe_id}")
def delete(recipe_id: str):
    path = recipes_dir() / f"{slug(recipe_id)}.json"
    if not path.is_file():
        raise HTTPException(404, f"No recipe {recipe_id}")
    path.unlink()
    return {"ok": True}


# ------------------------------------------------------------------ runs


class RunRequest(BaseModel):
    recipe: Recipe
    mode: str = "preview"  # preview | create
    num_records: int = Field(5, ge=1, le=100_000)


def check(recipe: Recipe) -> None:
    """Catch the usual mistakes here, where the message can be plain, instead of deep in the worker."""
    if not recipe.columns:
        raise HTTPException(400, "Add at least one column.")
    names = [c.get("name") for c in recipe.columns]
    if len(set(names)) != len(names) or not all(names):
        raise HTTPException(400, "Every column needs a unique name.")
    aliases = {m.get("alias") for m in recipe.models}
    for column in recipe.columns:
        if str(column.get("column_type", "")).startswith("llm") and column.get("model_alias") not in aliases:
            raise HTTPException(400, f"Column “{column['name']}” uses model “{column.get('model_alias')}”, which isn't in this recipe.")
    providers = {p.get("name") for p in recipe.providers} | {"local"}
    for model in recipe.models:
        if model.get("provider", "local") not in providers:
            raise HTTPException(400, f"Model “{model.get('alias')}” uses provider “{model.get('provider')}”, which isn't set up.")
    seed = recipe.seed or {}
    if seed.get("type") == "documents" and not any(s.strip() for s in seed.get("sources", [])):
        raise HTTPException(400, "Add at least one document, folder, web page or GitHub repository.")
    if seed.get("type") in ("file", "hf") and not seed.get("path"):
        raise HTTPException(400, "Choose the seed dataset.")


@router.post("/run")
def run(req: RunRequest):
    recipe = req.recipe
    check(recipe)
    uses_local = any(m.get("provider", "local") == "local" for m in recipe.models)
    if uses_local and manager.state != "ready":
        raise HTTPException(409, "Recipes with the “local” provider use the model loaded for chat. Load one on the Models page, or switch the model to a cloud provider.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = home() / "recipe-runs" / f"{slug(recipe.name)}-{stamp}"
    work.mkdir(parents=True)
    output = str(datasets_dir() / f"{slug(recipe.name)}-{stamp}.jsonl") if req.mode == "create" else None
    job_cfg = {
        "recipe": recipe.model_dump(),
        "mode": req.mode,
        "num_records": req.num_records,
        "output": output,
        "local_endpoint": f"{manager.base}/v1" if uses_local else None,
    }
    (work / "job.json").write_text(json.dumps(job_cfg, indent=2, ensure_ascii=False))

    cmd = [uv_binary(), "run", "--no-project", "--python", "3.12"]
    for requirement in REQUIREMENTS:
        cmd += ["--with", requirement]
    cmd += ["python", "-I", str(WORKER), str(work / "job.json")]

    verb = "Preview" if req.mode == "preview" else "Generate"
    job = registry.create("recipe", f"{verb} {recipe.name} ({req.num_records} rows)")
    job.result.update({"mode": req.mode, "output": output, "recipe": recipe.id})
    registry.run_command(job, cmd, uv_env())
    return job.summary()


@router.get("/datasets")
def generated_datasets():
    """Datasets that recipes have written, newest first, ready to train on."""
    out = []
    for path in sorted(datasets_dir().glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(path, encoding="utf-8") as f:
            rows = sum(1 for _ in f)
        out.append({"name": path.stem, "path": str(path), "rows": rows, "size": path.stat().st_size, "modified": path.stat().st_mtime})
    return out
