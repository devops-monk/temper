import pytest
from fastapi.testclient import TestClient

from temper_engine import skills
from temper_engine.app import create_app

AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPER_HOME", str(tmp_path / "home"))
    with TestClient(create_app(token="secret")) as c:
        yield c


def write(folder, name, description, body, files=()):
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    for f in files:
        (folder / f).write_text("x")


def test_skills_are_made_listed_and_removed(client):
    made = client.post("/api/skills", headers=AUTH, json={"name": "Release Notes", "description": "Writing release notes", "instructions": "Group changes by area."}).json()
    assert made["name"] == "release-notes" and made["source"] == "global"
    assert client.post("/api/skills", headers=AUTH, json={"name": "release-notes", "description": "", "instructions": "x"}).status_code == 409
    assert client.post("/api/skills", headers=AUTH, json={"name": "no/slashes", "description": "", "instructions": "x"}).status_code == 422
    assert [s["name"] for s in client.get("/api/skills", headers=AUTH).json()["skills"]] == ["release-notes"]
    assert client.delete("/api/skills/release-notes", headers=AUTH).json()["skills"] == []


def test_a_skill_folder_is_imported_with_its_files(client, tmp_path):
    write(tmp_path / "pdf-forms", "pdf-forms", "Filling PDF forms", "Use the script.", files=["fill.py"])
    assert client.post("/api/skills/import", headers=AUTH, json={"path": str(tmp_path / "pdf-forms")}).json()["name"] == "pdf-forms"
    assert (skills.global_dir() / "pdf-forms" / "fill.py").is_file()
    assert client.post("/api/skills/import", headers=AUTH, json={"path": str(tmp_path)}).status_code == 422


def test_mentions_bring_in_skills_and_a_project_s_own_win(client, tmp_path):
    write(skills.global_dir() / "review", "review", "Code review", "Global review rules.")
    write(skills.global_dir() / "tone", "tone", "Tone", "Be warm.", files=["examples.md"])
    project = tmp_path / "repo"
    write(project / ".agents" / "skills" / "review", "review", "Code review", "Project review rules.")
    thread = [{"role": "user", "content": "Please @review this, and email me@example.com"}, {"role": "assistant", "content": "@tone"}, {"role": "user", "content": [{"type": "text", "text": "now @tone and @nope"}]}]
    used = skills.mentioned(thread, str(project))
    assert [(s["name"], s["body"]) for s in used] == [("review", "Project review rules."), ("tone", "Be warm.")]  # an email address isn't a mention
    text = skills.prompt(used)
    assert '<skill name="review">\nProject review rules.' in text and "examples.md" in text
    assert skills.mentioned([{"role": "user", "content": "no mentions"}], None) == []
