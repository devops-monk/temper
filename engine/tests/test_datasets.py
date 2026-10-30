import json
import time

from fastapi.testclient import TestClient

from temper_engine.app import create_app
from temper_engine.jobs import registry

AUTH = {"Authorization": "Bearer secret"}


def test_documents_become_training_text_with_an_eval_split(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(30):
        (docs / f"note{i}.md").write_text(f"# Note {i}\n" + f"Paragraph about topic {i}. " * 40)
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    with TestClient(create_app(token="secret")) as c:
        assert c.post("/api/datasets/from-documents", headers=AUTH, json={"paths": [str(tmp_path / "nope")]}).status_code == 422
        job = c.post("/api/datasets/from-documents", headers=AUTH, json={"paths": [str(docs)], "name": "My notes", "chunk_chars": 500}).json()
        for _ in range(200):
            if registry.get(job["id"]).status != "running":
                break
            time.sleep(0.02)
        j = registry.get(job["id"])
        assert j.status == "done", j.error
        result = next(e for e in j.events if e["event"] == "result")
        train = [json.loads(line) for line in open(result["output"])]
        held = [json.loads(line) for line in open(result["eval_output"])]
        assert result["output"].endswith(".jsonl") and "my-notes-text-" in result["output"] and result["eval_output"].endswith(".eval.jsonl")
        assert all(set(r) == {"text"} and len(r["text"]) <= 500 for r in train + held)
        assert result["stats"]["files"] == 31 and result["stats"]["unreadable"] == 1 and len(train) + len(held) == result["stats"]["rows_in"] - result["stats"]["duplicates"]
        assert len(held) == int((len(train) + len(held)) * 0.05)
