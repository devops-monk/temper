"""Agent skills: folders of instructions a chat brings in by mentioning them as @name.

A skill is a folder holding SKILL.md, whose front matter gives its `name` and a `description` of
when to use it, and whose body says how; other files in the folder (scripts, templates, references)
go with it. Skills come from Temper's home (skills/<name>/) and from the chat's project folder
(.agents/skills/ and .claude/skills/, the places other agents read them from); a project's skill
of the same name wins.

Mentioning @name in any message of a chat puts that skill's SKILL.md in the system prompt, with
the paths of its other files, for the rest of the chat.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import home

router = APIRouter(prefix="/api/skills", tags=["chat"])

MAX_BYTES = 24 * 1024  # of one SKILL.md
MAX_FILES = 40  # listed from a skill's folder
PROJECT_DIRS = (".agents/skills", ".claude/skills")
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MENTION = re.compile(r"(?<![\w@./])@([a-z0-9][a-z0-9-]{0,63})\b")


def global_dir() -> Path:
    folder = home() / "skills"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _front_matter(text: str) -> tuple[dict, str]:
    """SKILL.md's simple key: value front matter, and the body after it."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "#")):
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[m.end() :]


def _read(folder: Path, source: str) -> dict | None:
    doc = folder / "SKILL.md"
    if not doc.is_file():
        return None
    try:
        text = doc.read_text(encoding="utf-8", errors="replace")[:MAX_BYTES]
    except OSError:
        return None
    meta, body = _front_matter(text)
    name = (meta.get("name") or folder.name).strip().lower().replace(" ", "-")
    return {"name": name, "description": meta.get("description", ""), "path": str(folder), "source": source, "body": body.strip()}


def _in(folder: Path, source: str) -> list[dict]:
    if not folder.is_dir():
        return []
    return [s for d in sorted(folder.iterdir()) if d.is_dir() and (s := _read(d, source))]


def all_skills(project: str | None = None) -> dict[str, dict]:
    """The skills a chat can mention, by name; a project's own win over Temper's."""
    found = {s["name"]: s for s in _in(global_dir(), "global")}
    if project and Path(project).expanduser().is_dir():
        root = Path(project).expanduser().resolve()
        for sub in reversed(PROJECT_DIRS):  # .agents/skills last, so it wins over .claude/skills
            found |= {s["name"]: s for s in _in(root / sub, "project")}
    return found


def mentioned(messages: list[dict], project: str | None) -> list[dict]:
    """The skills mentioned with @name in a chat's user messages, in the order first mentioned."""
    names: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")
        names += [n for n in MENTION.findall(text) if n not in names]
    if not names:
        return []
    skills = all_skills(project)
    return [skills[n] for n in names if n in skills]


def _files(folder: Path) -> list[str]:
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name != "SKILL.md" and not any(part.startswith(".") for part in p.relative_to(folder).parts):
            out.append(str(p))
            if len(out) >= MAX_FILES:
                break
    return out


def prompt(skills: list[dict]) -> str:
    """The system prompt text for the mentioned skills."""
    blocks = []
    for s in skills:
        files = _files(Path(s["path"]))
        listing = ("\n\nFiles that come with it:\n" + "\n".join(f"- {f}" for f in files)) if files else ""
        blocks.append(f'<skill name="{s["name"]}">\n{s["body"]}{listing}\n</skill>')
    return "The person asked you to use these skills. Follow their instructions.\n\n" + "\n\n".join(blocks) if blocks else ""


# --- API ------------------------------------------------------------------------------------------


@router.get("")
def list_skills(project: str | None = None):
    return {"skills": [{k: v for k, v in s.items() if k != "body"} for s in all_skills(project).values()], "folder": str(global_dir())}


class NewSkill(BaseModel):
    name: str
    description: str = Field(max_length=1024)
    instructions: str = Field(min_length=1, max_length=MAX_BYTES)


@router.post("")
def create_skill(req: NewSkill):
    name = req.name.strip().lower().replace(" ", "-")
    if not NAME.match(name):
        raise HTTPException(422, "A skill's name is lowercase letters, digits and dashes, like write-tests.")
    folder = global_dir() / name
    if folder.exists():
        raise HTTPException(409, f"There's already a skill called {name}.")
    folder.mkdir()
    description = req.description.replace("\n", " ").strip()
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{req.instructions.strip()}\n", encoding="utf-8")
    return {k: v for k, v in (_read(folder, "global") or {}).items() if k != "body"}


class ImportSkill(BaseModel):
    path: str


@router.post("/import")
def import_skill(req: ImportSkill):
    """Copies a skill folder (one holding SKILL.md) into Temper's skills."""
    source = Path(req.path).expanduser()
    skill = _read(source, "global") if source.is_dir() else None
    if not skill:
        raise HTTPException(422, "That folder has no SKILL.md in it.")
    if not NAME.match(skill["name"]):
        raise HTTPException(422, f"{skill['name']!r} isn't a usable skill name: use lowercase letters, digits and dashes.")
    target = global_dir() / skill["name"]
    if target.exists():
        raise HTTPException(409, f"There's already a skill called {skill['name']}.")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "node_modules", "__pycache__"))
    return {k: v for k, v in (_read(target, "global") or {}).items() if k != "body"}


@router.delete("/{name}")
def delete_skill(name: str):
    folder = global_dir() / name
    if not NAME.match(name) or not (folder / "SKILL.md").is_file():
        raise HTTPException(404, f"There's no skill called {name} in Temper's skills.")
    shutil.rmtree(folder)
    return list_skills()
