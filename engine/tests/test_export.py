import json

from fastapi.testclient import TestClient

from temper_engine.app import create_app


def test_a_run_is_published_with_a_card_and_only_its_model_files(tmp_path, monkeypatch):
    import time

    from temper_engine import home

    calls = []

    class FakeApi:
        def whoami(self):
            return {"name": "someone"}

        def create_repo(self, repo, repo_type, private, exist_ok):
            calls.append(("create", repo, repo_type, private))
            return f"https://huggingface.co/{repo}"

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            calls.append(("upload", path_in_repo, path_or_fileobj if isinstance(path_or_fileobj, bytes) else None))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    run = home() / "runs" / "20261030-120000-tutor"
    (run / "adapter").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"name": "tutor", "base_model": "mlx-community/Qwen3-0.6B-4bit", "dataset": "openai/gsm8k", "method": "lora", "backend": "mlx", "lora_r": 16, "lora_alpha": 16, "learning_rate": 1e-4, "batch_size": 2, "grad_accum": 4, "max_seq_length": 2048}))
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "dataset", "rows": 120}, {"event": "provenance", "dataset_sha256": "ab" * 32, "base_revision": "c" * 40, "libraries": {"mlx_lm": "0.32.2"}},
        {"event": "metrics", "step": 60, "loss": 0.81}, {"event": "eval", "step": 60, "val_loss": 0.9},
    ]))
    for f in ("adapters.safetensors", "adapter_config.json", "0000020_adapters.safetensors"):
        (run / "adapter" / f).write_text("x")
    with TestClient(create_app(token="t")) as c:
        h = {"Authorization": "Bearer t"}
        assert c.post("/api/publish/model", headers=h, json={"run_id": run.name, "repo": "bad name"}).status_code == 422
        assert c.post("/api/publish/model", headers=h, json={"run_id": run.name, "repo": "x", "export": str(tmp_path)}).status_code == 400
        job = c.post("/api/publish/model", headers=h, json={"run_id": run.name, "repo": "tutor"}).json()
        for _ in range(100):
            if c.get(f"/api/jobs/{job['id']}", headers=h).json()["status"] != "running":
                break
            time.sleep(0.05)
        done = c.get(f"/api/jobs/{job['id']}", headers=h).json()
    assert done["status"] == "done" and done["result"]["url"] == "https://huggingface.co/someone/tutor"
    assert calls[0] == ("create", "someone/tutor", "model", True)  # private unless asked
    assert [c[1] for c in calls[1:]] == ["README.md", "adapter_config.json", "adapters.safetensors"]  # no checkpoints
    card = calls[1][2].decode()
    assert "base_model: mlx-community/Qwen3-0.6B-4bit" in card and "- openai/gsm8k" in card and "| Final validation loss | 0.9000 |" in card
    assert "SHA-256 `abababababababab…`" in card and "| Rows | 120 |" in card and "mlx_lm 0.32.2" in card


def test_a_dataset_is_published_with_its_held_out_split(tmp_path, monkeypatch):
    import time

    calls = []

    class FakeApi:
        def whoami(self):
            return {"name": "someone"}

        def create_repo(self, repo, repo_type, private, exist_ok):
            calls.append(("create", repo, repo_type, private))
            return f"https://huggingface.co/datasets/{repo}"

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            calls.append(("upload", path_in_repo, repo_type))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    data = tmp_path / "qa.jsonl"
    data.write_text('{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}\n' * 3)
    data.with_suffix(".eval.jsonl").write_text('{"messages": []}\n')
    with TestClient(create_app(token="t")) as c:
        h = {"Authorization": "Bearer t"}
        job = c.post("/api/publish/dataset", headers=h, json={"path": str(data), "repo": "me/qa", "private": False}).json()
        for _ in range(100):
            if c.get(f"/api/jobs/{job['id']}", headers=h).json()["status"] != "running":
                break
            time.sleep(0.05)
        assert c.get(f"/api/jobs/{job['id']}", headers=h).json()["status"] == "done"
    assert calls == [("create", "me/qa", "dataset", False), ("upload", "README.md", "dataset"), ("upload", "data/train.jsonl", "dataset"), ("upload", "data/test.jsonl", "dataset")]
