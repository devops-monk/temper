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
