from fastapi.testclient import TestClient

from temper_engine import catalog
from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def test_catalog_entries_are_complete():
    ids = set()
    for family in catalog.load()["families"]:
        assert family["name"] and family["about"] and family["variants"]
        for v in family["variants"]:
            assert v["id"] not in ids
            ids.add(v["id"])
            assert v["params"] > 0 and v["context"] and v["license"] and v["model_type"]
            assert v["gguf"], f"{v['id']} has nothing to chat with"
            assert set(v["sampling"]) <= {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty"}
            assert "unsloth" not in (str(v["gguf"]) + str(v["mlx"])).lower()


def test_endpoint_adds_fit_and_training():
    body = client.get("/api/catalog", headers=AUTH).json()
    v = body["families"][0]["variants"][0]
    assert v["fit"]["verdict"] in {"fits", "tight", "too_large"} and "trainable" in v and "train_repo" in v
    for family in body["families"]:
        if family["suggested"]:
            assert any(x["id"] == family["suggested"] and x["fit"]["verdict"] == "fits" for x in family["variants"])


def test_match_prefers_the_longest_name_and_respects_boundaries():
    assert catalog.match("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")["id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert catalog.match("/hub/models--lmstudio-community--Qwen3.5-4B-GGUF/snapshots/x/Qwen3.5-4B-Q4_K_M.gguf")["id"] == "Qwen/Qwen3.5-4B"
    assert catalog.match("google_gemma-4-E4B-it-Q8_0.gguf")["id"] == "google/gemma-4-E4B-it"
    assert catalog.match("Qwen3.5-40B-Q4_K_M.gguf") is None
    assert catalog.match("SmolLM2-135M-Instruct.Q4_0.gguf") is None
    r = client.get("/api/catalog/match", params={"model": "gpt-oss-20b-mxfp4.gguf"}, headers=AUTH).json()
    assert r["match"]["sampling"]["temperature"] == 1.0
