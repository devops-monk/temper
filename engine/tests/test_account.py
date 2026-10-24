import os

import httpx
import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from temper_engine import account
from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def http_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co"))
    return cls("nope", response=response)


class FakeApi:
    good = "hf_good"

    def whoami(self, token=None):
        if token != self.good:
            raise http_error(HfHubHTTPError, 401)
        return {"name": "ada", "fullname": "Ada", "orgs": [{"name": "lab"}]}

    def auth_check(self, repo_id):
        if repo_id.startswith("gated/"):
            raise http_error(GatedRepoError, 403)


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch, tmp_path):
    monkeypatch.setattr(account, "HfApi", FakeApi)
    monkeypatch.setattr(account.constants, "HF_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.setattr(account, "get_token", lambda: os.environ.get("HF_TOKEN"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_signed_out_by_default():
    assert client.get("/api/hf/account", headers=AUTH).json() == {"signed_in": False, "source": None}


def test_saving_a_token_signs_in_and_sets_the_environment():
    r = client.post("/api/hf/token", json={"token": " hf_good "}, headers=AUTH)
    assert r.status_code == 200 and r.json()["name"] == "ada" and r.json()["orgs"] == ["lab"]
    assert os.environ["HF_TOKEN"] == "hf_good"
    assert client.get("/api/hf/account", headers=AUTH).json()["source"] == "temper"
    assert client.delete("/api/hf/token", headers=AUTH).json()["signed_in"] is False
    assert "HF_TOKEN" not in os.environ


def test_bad_tokens_are_refused():
    assert "start with hf_" in client.post("/api/hf/token", json={"token": "abc"}, headers=AUTH).json()["detail"]
    r = client.post("/api/hf/token", json={"token": "hf_wrong"}, headers=AUTH)
    assert r.status_code == 400 and "didn't accept" in r.json()["detail"]
    assert "HF_TOKEN" not in os.environ


def test_gated_model_access():
    gated = client.get("/api/hf/access", params={"repo_id": "gated/llama"}, headers=AUTH).json()
    assert gated["gated"] and not gated["access"] and "add a Hugging Face token" in gated["message"]
    assert gated["url"] == "https://huggingface.co/gated/llama"
    assert client.get("/api/hf/access", params={"repo_id": "open/qwen"}, headers=AUTH).json()["access"]
