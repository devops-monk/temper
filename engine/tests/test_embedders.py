import base64
import json
import struct
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from temper_engine import classifiers, embedders, knowledge
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
# Tunes by recording its plan; embeds each text as [length, 1, 0, 0] made unit length.
FAKE_TUNE = """
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
before = {"recall_at_1": 0.2, "recall_at_5": 0.4, "mrr_at_10": 0.3, "candidates": len(plan["eval"])}
after = {"recall_at_1": 0.6, "recall_at_5": 0.9, "mrr_at_10": 0.7, "candidates": len(plan["eval"])}
Path(plan["out"], "model").mkdir()
info = {k: plan[k] for k in ("name", "base_id", "dataset", "created")}
info.update(train_pairs=len(plan["train"]), eval_pairs=len(plan["eval"]), epochs=plan["epochs"], before=before, after=after, dimensions=4)
Path(plan["out"], "embedder.json").write_text(json.dumps(info))
print("@@" + json.dumps({"event": "result", "name": plan["name"], "before": before, "after": after}), flush=True)
"""
FAKE_SERVE = """
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    vectors = []
    for t in req["texts"]:
        n = (len(t) ** 2 + 1) ** 0.5
        vectors.append([len(t) / n, 1 / n, 0.0, 0.0])
    print(json.dumps({"done": {"vectors": vectors, "seen": req["embed"]}}), flush=True)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    tune, serve = tmp_path / "tune.py", tmp_path / "serve.py"
    tune.write_text(FAKE_TUNE)
    serve.write_text(FAKE_SERVE)
    monkeypatch.setattr(embedders, "WORKER", tune)
    monkeypatch.setattr(classifiers, "python", lambda: sys.executable)
    monkeypatch.setattr(classifiers, "encoder_folder", lambda e: str(tmp_path / "st") if e.startswith("sentence-transformers/") else None)
    monkeypatch.setattr(classifiers.worker, "command", lambda: [sys.executable, "-u", str(serve)])
    monkeypatch.setattr(knowledge, "embedder_path", lambda eid: None)
    with TestClient(create_app(token="t")) as c:
        yield c
    classifiers.worker.stop()


def wait(client, job):
    for _ in range(200):
        done = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if done["status"] != "running":
            return done
        time.sleep(0.05)
    raise AssertionError("still running")


def write_pairs(path, n, keys=("question", "answer")):
    path.write_text("".join(json.dumps({keys[0]: f"q{i}", keys[1]: f"a{i}", "id": i}) + "\n" for i in range(n)))
    return str(path)


def test_pairs_are_found_by_column_name(client, tmp_path):
    out = client.post("/api/embedders/peek", headers=H, json={"dataset": write_pairs(tmp_path / "p.jsonl", 12)}).json()
    assert (out["anchor_column"], out["positive_column"], out["pairs"], out["problem"]) == ("question", "answer", 12, None)
    few = client.post("/api/embedders/peek", headers=H, json={"dataset": write_pairs(tmp_path / "f.jsonl", 4)}).json()
    assert few["problem"] == "Give at least 10 pairs."
    odd = client.post("/api/embedders/peek", headers=H, json={"dataset": write_pairs(tmp_path / "o.jsonl", 12, ("left", "right"))}).json()
    assert "Which column" in odd["problem"]


def test_tuning_holds_out_pairs_and_reports_before_and_after(client, tmp_path):
    job = client.post("/api/embedders", headers=H, json={"name": "Our Docs", "dataset": write_pairs(tmp_path / "p.jsonl", 50), "epochs": 2, "eval_share": 0.2}).json()
    assert (job["train"], job["eval"]) == (40, 10)
    assert wait(client, job)["status"] == "done"
    found = client.get("/api/embedders", headers=H).json()["tuned"]
    assert found[0]["name"] == "our-docs" and found[0]["after"]["recall_at_1"] == 0.6 and found[0]["epochs"] == 2
    plan = json.loads((embedders.folder() / "our-docs" / "plan.json").read_text())
    assert not {tuple(p) for p in plan["train"]} & {tuple(p) for p in plan["eval"]}
    assert client.post("/api/embedders", headers=H, json={"name": "our docs", "dataset": write_pairs(tmp_path / "p.jsonl", 50)}).status_code == 409
    assert client.post("/api/embedders", headers=H, json={"name": "x", "dataset": write_pairs(tmp_path / "s.jsonl", 5)}).status_code == 400
    assert client.post("/api/embedders", headers=H, json={"name": "x", "dataset": write_pairs(tmp_path / "p.jsonl", 50), "base": "nobody/model"}).status_code == 409


def test_openai_embeddings_from_a_tuned_or_downloaded_model(client, tmp_path):
    wait(client, client.post("/api/embedders", headers=H, json={"name": "docs", "dataset": write_pairs(tmp_path / "p.jsonl", 20)}).json())
    out = client.post("/v1/embeddings", headers=H, json={"model": "docs", "input": ["abc", "a"]}).json()
    assert out["model"] == "docs" and [d["index"] for d in out["data"]] == [0, 1] and len(out["data"][0]["embedding"]) == 4
    assert out["usage"]["prompt_tokens"] == 2
    short = client.post("/v1/embeddings", headers=H, json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "abc", "dimensions": 2}).json()
    v = short["data"][0]["embedding"]
    assert len(v) == 2 and abs(sum(x * x for x in v) - 1) < 1e-6
    packed = client.post("/v1/embeddings", headers=H, json={"model": "docs", "input": "abc", "encoding_format": "base64"}).json()["data"][0]["embedding"]
    assert np.allclose(struct.unpack("<4f", base64.b64decode(packed)), out["data"][0]["embedding"], atol=1e-6)
    # OpenAI's names fall back to a general model on this computer, never a tuned one.
    assert client.post("/v1/embeddings", headers=H, json={"model": "text-embedding-3-small", "input": "x"}).json()["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert client.post("/v1/embeddings", headers=H, json={"input": []}).status_code == 400


def test_a_knowledge_embedder_answers_by_its_id_or_repo(client, monkeypatch):
    monkeypatch.setattr(knowledge, "embedder_path", lambda eid: "/x.gguf" if eid == "bge-small-en-1.5" else None)
    monkeypatch.setattr(knowledge.embedder, "embed", lambda texts, eid, query=False: np.ones((len(texts), 3)) / 3**0.5)
    out = client.post("/v1/embeddings", headers=H, json={"model": "CompendiumLabs/bge-small-en-v1.5-gguf", "input": ["a", "b"]}).json()
    assert out["model"] == "bge-small-en-1.5" and len(out["data"]) == 2
    assert client.post("/v1/embeddings", headers=H, json={"model": "gpt-whatever", "input": "a"}).json()["model"] == "bge-small-en-1.5"


def test_without_any_embedding_model_it_says_where_to_get_one(client, monkeypatch):
    monkeypatch.setattr(classifiers, "encoder_folder", lambda e: None)
    r = client.post("/v1/embeddings", headers=H, json={"input": "a"})
    assert r.status_code == 404 and "Knowledge bases" in r.json()["detail"]
