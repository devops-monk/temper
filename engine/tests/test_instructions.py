from pathlib import Path

from fastapi.testclient import TestClient

from temper_engine import instructions
from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def test_global_and_project_files_are_read_root_first(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "AGENTS.md").write_text("Use tabs.")
    (repo / "web").mkdir()
    (repo / "web" / "AGENTS.md").write_text("Ignored: the override wins.")
    (repo / "web" / "AGENTS.override.md").write_text("Use TypeScript.")
    (repo / "web" / "src").mkdir()
    (tmp_path / "AGENTS.md").write_text("Outside the repo: never read.")
    client.put("/api/instructions/global", headers=AUTH, json={"text": "Answer briefly."})
    out = client.get("/api/instructions", headers=AUTH, params={"project": str(repo / "web" / "src")}).json()
    assert out["global"] == "Answer briefly."
    assert [Path(p["path"]).parts[-2:] for p in out["parts"]][1:] == [("repo", "AGENTS.md"), ("web", "AGENTS.override.md")]
    assert instructions.with_instructions("Be kind.", str(repo / "web" / "src")) == "Answer briefly.\n\nUse tabs.\n\nUse TypeScript.\n\nBe kind."
    # Clearing the global file removes it; a folder outside a repo reads only its own AGENTS.md.
    client.put("/api/instructions/global", headers=AUTH, json={"text": " "})
    assert instructions.with_instructions("", str(tmp_path)) == "Outside the repo: never read."


def test_instructions_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(instructions, "MAX_BYTES", 10)
    (tmp_path / "AGENTS.md").write_text("0123456789ABCDEF")
    out = instructions.gather(str(tmp_path))
    assert out["text"] == "0123456789" and out["truncated"]
