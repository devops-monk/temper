import json

import pytest
from fastapi.testclient import TestClient

from temper_engine import training
from temper_engine.app import create_app

AUTH = {"Authorization": "Bearer secret"}


def test_full_and_embedding_training_need_an_unquantized_mlx_model(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "probe", lambda: {"backend": "metal", "capabilities": {"trainer": "mlx"}})
    started = []
    monkeypatch.setattr(training.registry, "run_worker", lambda job, module, cfg: started.append(json.loads(open(cfg).read())))
    four_bit = tmp_path / "Qwen3-0.6B"
    four_bit.mkdir()
    (four_bit / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    sixteen = tmp_path / "Qwen3-0.6B-16"
    sixteen.mkdir()
    (sixteen / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with TestClient(create_app(token="secret")) as c:
        refused = c.post("/api/train", headers=AUTH, json={"base_model": str(four_bit), "method": "full", "dataset": "x.jsonl"})
        assert refused.status_code == 400 and "16-bit" in refused.json()["detail"]
        assert c.post("/api/train", headers=AUTH, json={"base_model": "mlx-community/Qwen3-0.6B-4bit", "method": "lora", "train_embeddings": True, "dataset": "x.jsonl"}).status_code == 400
        assert c.post("/api/train", headers=AUTH, json={"base_model": str(sixteen), "method": "full", "train_embeddings": True, "dataset": "x.jsonl"}).status_code == 200
        assert c.post("/api/train", headers=AUTH, json={"base_model": str(four_bit), "method": "lora", "dataset": "x.jsonl"}).status_code == 200
    assert started[0]["method"] == "full" and started[0]["train_embeddings"] is True


def test_embeddings_and_head_are_trained_when_asked():
    pytest.importorskip("mlx_lm")
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    from temper_engine import mlx_lora

    args = llama.ModelArgs(model_type="llama", hidden_size=16, num_hidden_layers=2, intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-5, vocab_size=50, num_key_value_heads=2, tie_word_embeddings=False)
    model = llama.Model(args)

    def fake_train(args, model):  # what mlx-lm does first: freeze it all, then unfreeze one layer
        model.freeze()
        model.layers[-1].unfreeze()
        return sorted(k for k, _ in tree_flatten(model.trainable_parameters()))

    original = mlx_lora._train_model
    mlx_lora._train_model = fake_train
    try:
        trained = mlx_lora._train_model_with_embeddings(None, model)
    finally:
        mlx_lora._train_model = original
    assert "model.embed_tokens.weight" in trained and "lm_head.weight" in trained
    assert any(k.startswith("model.layers.1.") for k in trained) and not any(k.startswith("model.layers.0.") for k in trained)


def make_run(name: str, steps: list[tuple[int, float]], checkpoint_steps=(10, 20)) -> str:
    d = training.runs_dir() / name
    (d / "adapter").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"name": "math tutor", "base_model": "mlx-community/Qwen3-0.6B-4bit", "dataset": "data.jsonl", "method": "lora", "backend": "mlx", "max_steps": 30, "output_dir": str(d)}))
    events = [{"event": "dataset", "rows": 100, "format": "messages", "sample": "…"}, {"event": "provenance", "dataset_sha256": "ab" * 32, "base_revision": "c" * 40}]
    events += [{"event": "metrics", "step": s, "loss": loss, "total_steps": 30} for s, loss in steps] + [{"event": "eval", "step": 20, "val_loss": 0.9}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    (d / "status.json").write_text(json.dumps({"status": "done"}))
    (d / "adapter" / "adapter_config.json").write_text("{}")
    for s in checkpoint_steps:
        (d / "adapter" / f"{s:07d}_adapters.safetensors").write_bytes(b"w" * s)
    return name


def test_runs_keep_their_history_and_checkpoints(monkeypatch):
    rid = make_run("20261030-100000-math", [(1, 2.0), (2, 1.5), (30, 0.7)])
    with TestClient(create_app(token="secret")) as c:
        listed = c.get("/api/train/runs", headers=AUTH).json()[0]
        assert listed | {"created": 0} == {**listed, "created": 0, "status": "done", "steps": 30, "loss": 0.7, "val_loss": 0.9, "checkpoints": 2, "dataset_sha256": "ab" * 32}
        run = c.get(f"/api/train/runs/{rid}", headers=AUTH).json()
        assert [m["loss"] for m in run["metrics"]] == [2.0, 1.5, 0.7] and run["provenance"]["base_revision"] == "c" * 40 and run["dataset_info"]["rows"] == 100
        assert [p["step"] for p in run["checkpoint_list"]] == [10, 20]
        adapter = c.post(f"/api/train/runs/{rid}/checkpoints/20/adapter", headers=AUTH).json()["adapter"]
        assert sorted(p.name for p in training.Path(adapter).iterdir()) == ["adapter_config.json", "adapters.safetensors"]
        assert c.post(f"/api/train/runs/{rid}/checkpoints/15/adapter", headers=AUTH).status_code == 404
        assert c.get("/api/train/runs/../../etc", headers=AUTH).status_code == 404

        monkeypatch.setattr(training, "probe", lambda: {"backend": "metal", "capabilities": {"trainer": "mlx"}})
        started = []
        monkeypatch.setattr(training.registry, "run_worker", lambda job, module, cfg: started.append(json.loads(open(cfg).read())))
        assert c.post(f"/api/train/runs/{rid}/rerun", headers=AUTH).status_code == 200
        assert c.post(f"/api/train/runs/{rid}/resume", headers=AUTH, json={"step": 20}).status_code == 200
    assert started[0]["max_steps"] == 30 and started[0]["resume_from"] is None and started[0]["name"] == "math tutor"
    assert started[1]["resume_from"].endswith("0000020_adapters.safetensors") and started[1]["max_steps"] == 10 and started[1]["start_step"] == 20 and started[1]["name"] == "math tutor from step 20"


def test_the_worker_records_its_events(tmp_path, monkeypatch, capsys):
    from temper_engine import train_worker

    monkeypatch.setattr(train_worker, "RUN", tmp_path)
    train_worker.emit("metrics", step=1, loss=2.5)
    train_worker.emit("log", line="not kept")
    assert "@@" in capsys.readouterr().out
    kept = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(e["event"], e["loss"]) for e in kept] == [("metrics", 2.5)]
    assert train_worker.save_every({"save_every": 0}, 200) == 40 and train_worker.save_every({"save_every": 0}, 20) == 10 and train_worker.save_every({"save_every": 7}, 200) == 7


def test_held_out_conversations_become_sample_prompts():
    from temper_engine import train_worker

    convs = [
        "plain text",
        [{"role": "user", "content": "Hi"}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
        [{"role": "user", "content": "Capital of France?"}, {"role": "assistant", "content": "Paris"}],
        [{"role": "user", "content": "Third"}, {"role": "assistant", "content": "not needed"}],
    ]
    prompts = train_worker.sample_prompts(convs)
    assert [p["expected"] for p in prompts] == ["4", "Paris"]
    assert prompts[0]["messages"][-1] == {"role": "user", "content": "2+2?"} and len(prompts[0]["messages"]) == 2
    assert train_worker.eval_every(200) == 20 and train_worker.eval_every(5) == 1


def test_mlx_training_stops_early_and_keeps_the_best_adapter(tmp_path, monkeypatch, capsys):
    pytest.importorskip("mlx_lm")
    import mlx.core as mx

    from temper_engine import mlx_lora

    class Model:
        weight = 0.0

        def trainable_parameters(self):
            return {"w": mx.array([self.weight])}

        def eval(self):
            pass

        def train(self):
            pass

    class Args:
        adapter_file = str(tmp_path / "adapters.safetensors")

    monkeypatch.setenv("TEMPER_EARLY_STOP", "2")
    monkeypatch.delenv("TEMPER_SAMPLES", raising=False)
    model = Model()

    def fake_train(model, args, training_callback, **kw):
        for it, (w, loss) in enumerate([(1, 2.0), (2, 1.5), (3, 1.6), (4, 1.7), (5, 1.0)]):
            model.weight = w
            training_callback.on_val_loss_report({"iteration": it * 10, "val_loss": loss})
        mx.save_safetensors(args.adapter_file, {"w": mx.array([float(model.weight)])})

    monkeypatch.setattr(mlx_lora, "_train", fake_train)
    mlx_lora._train_watched(model=model, args=Args(), optimizer=None, train_dataset=None, val_dataset=None, training_callback=None)
    assert mx.load(Args.adapter_file)["w"].item() == 2  # the best evaluation's weights, not the last step's
    best = [json.loads(line[7:]) for line in capsys.readouterr().out.splitlines() if line.startswith("TEMPER ")]
    assert best == [{"event": "best", "step": 11, "val_loss": 1.5, "stopped_at": 31}]
    assert not (tmp_path / "best_adapters.safetensors").exists()


def test_the_token_preview_masks_the_prompt_and_counts_tokens(tmp_path, monkeypatch):
    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    monkeypatch.setattr(training, "_tokenizer", lambda base: Tok())
    data = tmp_path / "d.jsonl"
    rows = [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]}, {"text": "plain"}]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    with TestClient(create_app(token="t")) as c:
        got = c.post("/api/train/preview", headers={"Authorization": "Bearer t"}, json={"base_model": "m", "dataset": str(data)}).json()
        assert "".join(t["text"] for t in got["tokens"][: got["masked"]]) == "<user>Hi</><assistant>"
        assert "".join(t["text"] for t in got["tokens"][got["masked"]:]) == "Yo</>"
        assert got["count"] == len(got["tokens"]) and got["format"] == "messages" and got["more"] is True
        text = c.post("/api/train/preview", headers={"Authorization": "Bearer t"}, json={"base_model": "m", "dataset": str(data), "dataset_format": "text", "row": 1}).json()
        assert text["masked"] == 0 and "".join(t["text"] for t in text["tokens"]) == "plain" and text["more"] is False
