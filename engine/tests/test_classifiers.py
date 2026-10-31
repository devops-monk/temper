import csv
import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from temper_engine import classifiers
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
# Trains by writing the plan it was given as the classifier; classifies by the first word of each text.
FAKE = """
import json, sys, time
from pathlib import Path
if sys.argv[1] == "train":
    plan = json.loads(Path(sys.argv[2]).read_text())
    print("@@" + json.dumps({"event": "progress", "value": 0.5}), flush=True)
    info = {k: plan[k] for k in ("name", "labels", "dataset", "encoder_id", "created", "train", "eval")}
    info.update(metrics={"accuracy": 0.9, "macro_f1": 0.8}, head={}, fine_tuned=plan["fine_tune"], encoder=plan["encoder"])
    Path(plan["out"], "classifier.json").write_text(json.dumps(info))
    print("@@" + json.dumps({"event": "result", "name": plan["name"]}), flush=True)
else:
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        labels = json.loads(Path(req["classifier"], "classifier.json").read_text())["labels"]
        out = []
        for t in req["texts"]:
            label = t.split()[0] if t.split()[0] in labels else labels[0]
            out.append({"label": label, "confidence": 0.9, "scores": {label: 0.9}})
        print(json.dumps({"done": {"results": out}}), flush=True)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    script = tmp_path / "fake_classifier.py"
    script.write_text(FAKE)
    monkeypatch.setattr(classifiers, "WORKER", script)
    monkeypatch.setattr(classifiers, "python", lambda: sys.executable)
    monkeypatch.setattr(classifiers, "encoder_folder", lambda e: str(tmp_path / "encoder"))
    monkeypatch.setattr(classifiers.worker, "command", lambda: [sys.executable, "-u", str(script), "serve"])
    with TestClient(create_app(token="t")) as c:
        yield c
    classifiers.worker.stop()


def tickets(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Message", "Category", "id"])
        w.writerows([t, label, i] for i, (t, label) in enumerate(rows))
    return str(path)


def wait(client, job):
    for _ in range(200):
        done = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if done["status"] != "running":
            return done
        time.sleep(0.05)
    raise AssertionError("still running")


ROWS = [(f"billing charge {i}", "billing") for i in range(10)] + [(f"bug crash {i}", "bug") for i in range(5)]


def test_columns_are_found_by_name_and_labels_counted(client, tmp_path):
    path = tickets(tmp_path / "t.csv", ROWS)
    out = client.post("/api/classifiers/peek", headers=H, json={"dataset": path}).json()
    assert (out["text_column"], out["label_column"], out["labels"], out["problem"]) == ("Message", "Category", {"billing": 10, "bug": 5}, None)
    wrong = client.post("/api/classifiers/peek", headers=H, json={"dataset": path, "label_column": "id"}).json()
    assert "only one" in wrong["problem"]
    assert "no column called" in client.post("/api/classifiers/peek", headers=H, json={"dataset": path, "text_column": "nope"}).json()["problem"]


def test_each_label_is_held_out_in_proportion_and_kept_for_training():
    rows = [[f"a{i}", "a"] for i in range(10)] + [[f"b{i}", "b"] for i in range(2)]
    train, held = classifiers.split(rows, 0.2)
    assert sorted(r[1] for r in held) == ["a", "a"] and len(train) == 10  # 20% of two rounds to none
    assert classifiers.split(rows, 0.0)[1] == []


def test_training_writes_a_classifier_that_answers(client, tmp_path):
    job = client.post("/api/classifiers", headers=H, json={"name": "Support Tickets!", "dataset": tickets(tmp_path / "t.csv", ROWS)}).json()
    assert job["labels"] == {"billing": 10, "bug": 5} and (job["train"], job["eval"]) == (12, 3)
    assert wait(client, job)["status"] == "done"
    listed = client.get("/api/classifiers", headers=H).json()["classifiers"]
    assert [c["name"] for c in listed] == ["support-tickets"] and listed[0]["labels"] == ["billing", "bug"] and "head" not in listed[0]
    out = client.post("/v1/classifiers/support-tickets", headers=H, json={"texts": ["bug in the app", "where is my refund"]}).json()
    assert [r["label"] for r in out["results"]] == ["bug", "billing"]
    assert client.post("/v1/classifiers/support-tickets", headers=H, json={"text": "bug again"}).json()["label"] == "bug"
    again = client.post("/api/classifiers", headers=H, json={"name": "support tickets", "dataset": tickets(tmp_path / "t.csv", ROWS)})
    assert again.status_code == 409
    assert client.delete("/api/classifiers/support-tickets", headers=H).json()["classifiers"] == []
    assert client.post("/v1/classifiers/support-tickets", headers=H, json={"text": "x"}).status_code == 404


def test_a_prepared_eval_file_is_used_instead_of_holding_out(client, tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"text": t, "label": label}) + "\n" for t, label in ROWS))
    (tmp_path / "d.eval.jsonl").write_text(json.dumps({"text": "bug x", "label": "bug"}) + "\n" + json.dumps({"text": "?", "label": "unknown"}) + "\n")
    wait(client, client.post("/api/classifiers", headers=H, json={"name": "d", "dataset": str(data)}).json())
    saved = json.loads((classifiers.folder() / "d" / "classifier.json").read_text())
    assert len(saved["train"]) == 15 and saved["eval"] == [["bug x", "bug"]]  # a label it never saw is left out


def test_bad_training_data_says_why(client, tmp_path):
    one = tickets(tmp_path / "one.csv", [("a", "x"), ("b", "x")])
    assert "at least two labels" in client.post("/api/classifiers", headers=H, json={"name": "n", "dataset": one}).json()["detail"]
    lonely = tickets(tmp_path / "lonely.csv", [("a", "x"), ("b", "x"), ("c", "y")])
    assert "only one" in client.post("/api/classifiers", headers=H, json={"name": "n", "dataset": lonely}).json()["detail"]
    (tmp_path / "f.txt").write_text("hi")
    assert client.post("/api/classifiers", headers=H, json={"name": "n", "dataset": str(tmp_path / "f.txt")}).status_code == 400


def test_a_saved_decision_can_answer_with_a_classifier(client, tmp_path):
    wait(client, client.post("/api/classifiers", headers=H, json={"name": "team", "dataset": tickets(tmp_path / "t.csv", ROWS)}).json())
    client.put("/api/decisions/route", headers=H, json={"question": "Which team?", "kind": "choice", "classifier": "team"})
    out = client.post("/v1/decisions/route", headers=H, json={"text": "bug report"}).json()
    assert out["answer"] == "bug" and out["confidence"] == 0.9 and "90% sure" in out["reason"]
    client.put("/api/decisions/yn", headers=H, json={"question": "q", "kind": "yes_no", "classifier": "team"})
    assert "aren't yes and no" in client.post("/v1/decisions/yn", headers=H, json={"text": "x"}).json()["detail"]
    assert client.put("/api/decisions/sc", headers=H, json={"question": "q", "kind": "score", "classifier": "team"}).status_code == 422
