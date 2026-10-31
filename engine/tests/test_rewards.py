import json

import pytest
from fastapi.testclient import TestClient

from temper_engine import rewards, training
from temper_engine.app import create_app

AUTH = {"Authorization": "Bearer secret"}


def test_final_answers_are_found_and_normalised():
    cases = {"The sum is 42.": "42", "so \\boxed{3.50}": "3.5", "3 and 4 #### 1,234": "1234", "The answer is Paris.": "paris",
             "Answer: 12 apples, not 13": "12", "<think>7</think>I get 9": "9"}
    assert {t: rewards.final_answer(t) for t in cases} == cases
    assert rewards.expected({"answer": "Janet sells 16 - 3 - 4 = 9 eggs. #### 18"}) == " 18"
    assert rewards.score_answer("so 18", {"answer": "work... #### 18"}) == 1.0 and rewards.score_answer("19", {"solution": "18"}) == 0.0
    assert rewards.score_answer("18", {"question": "no answer column"}) == 0.0


def test_prompts_from_columns_or_messages():
    rows = [{"question": "2+2?", "system": "Be brief.", "answer": "4"},
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
            {"nothing": 1}]
    found = rewards.prompts(rows)
    assert [p for p, _ in found] == [[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}], [{"role": "user", "content": "Hi"}]]
    with pytest.raises(ValueError, match="No usable rows"):
        rewards.prompts([{"nothing": 1}])


def test_rewards_add_up_and_bad_setups_are_refused():
    code = "def reward(prompt, completion, row):\n    return 0.5 if prompt.startswith('2+2') else 0\n"
    score = rewards.scorer({"rewards": ["answer", "format", "python"], "reward_pattern": r"Answer: *\d+", "reward_code": code})
    assert score([{"role": "user", "content": "2+2?"}], "Answer: 4", {"answer": "4"}) == 2.5
    assert score([{"role": "user", "content": "3+3?"}], "it's 5", {"answer": "6"}) == 0.0
    for cfg, why in (({"rewards": []}, "at least one"), ({"rewards": ["vibes"]}, "Unknown"), ({"rewards": ["format"]}, "needs a pattern"),
                     ({"rewards": ["format"], "reward_pattern": "("}, "regular expression"), ({"rewards": ["python"], "reward_code": "def nope(): pass"}, "called reward"),
                     ({"rewards": ["python"], "reward_code": "def reward(:"}, "doesn't parse")):
        with pytest.raises(ValueError, match=why):
            rewards.check(cfg)


def test_an_rl_run_is_checked_before_it_starts(monkeypatch):
    monkeypatch.setattr(training, "probe", lambda: {"backend": "metal", "capabilities": {"trainer": "mlx"}})
    started = []
    monkeypatch.setattr(training.registry, "run_worker", lambda job, module, cfg: (started.append(json.loads(open(cfg).read())), job.finish("done")))
    base = {"base_model": "m", "dataset": "x.jsonl", "objective": "dapo", "method": "lora"}
    with TestClient(create_app(token="secret")) as c:
        assert c.post("/api/train", headers=AUTH, json=base | {"rewards": ["format"]}).status_code == 400
        assert c.post("/api/train", headers=AUTH, json=base | {"method": "full"}).json()["detail"].startswith("Reinforcement learning trains LoRA")
        assert c.post("/api/train", headers=AUTH, json=base | {"max_completion_length": 4096, "max_seq_length": 2048}).status_code == 400
        assert c.post("/api/train", headers=AUTH, json=base | {"generations": 8}).status_code == 200
    assert (started[0]["objective"], started[0]["generations"], started[0]["rewards"]) == ("dapo", 8, ["answer"])


def test_group_advantages_and_loss_normalisers():
    pytest.importorskip("mlx_lm")
    import mlx.core as mx

    from temper_engine import mlx_grpo

    adv, useful = mlx_grpo.advantages([1, 0, 1, 0, 2, 2, 2, 2], 4, "grpo")
    assert useful == [True, False] and adv[4:] == [0, 0, 0, 0] and adv[0] == pytest.approx(0.5 / (1 / 3) ** 0.5, rel=1e-3)
    assert mlx_grpo.advantages([1, 0, 1, 0], 4, "dr_grpo")[0] == [0.5, -0.5, 0.5, -0.5]  # not scaled by the spread
    lengths, useful = [3, 5, 2, 2, 4, 4, 4, 4], [True, False]
    assert mlx_grpo.denominator("grpo", lengths, useful, 4, 100) == 8
    assert mlx_grpo.denominator("dr_grpo", lengths, useful, 4, 100) == 800
    assert mlx_grpo.denominator("dapo", lengths, useful, 4, 100) == 12  # tokens of the group with a signal only
    logps, mask = mx.array([[-1.0, -2.0, 0.0]]), mx.array([[1.0, 1.0, 0.0]])
    assert mlx_grpo.policy_loss("grpo", logps, mask, mx.array([2.0]), 4).item() == pytest.approx(-0.5)  # -(2 per token, averaged) / 4 replies
    assert mlx_grpo.policy_loss("dapo", logps, mask, mx.array([2.0]), 4).item() == pytest.approx(-1.0)  # -(2 * 2 tokens) / 4 tokens
