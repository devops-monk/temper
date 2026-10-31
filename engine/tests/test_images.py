import base64
import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from temper_engine import images
from temper_engine.app import create_app

H = {"Authorization": "Bearer t"}
FAKE = """
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["prompt"] == "fail":
        print(json.dumps({"error": "RuntimeError: out of memory"}), flush=True)
        continue
    base = req["seed"] if req["seed"] is not None else 100
    for i, out in enumerate(req["out"]):
        print(json.dumps({"progress": (i + 1) / len(req["out"])}), flush=True)
        open(out, "w").write(json.dumps({k: req[k] for k in ("prompt", "width", "height", "steps", "guidance", "image", "strength", "variant", "lora")}))
    print(json.dumps({"done": {"seeds": [base + i for i in range(len(req["out"]))]}}), flush=True)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The engine with diffusers installed, played by a stand-in worker that writes what it was asked instead of a PNG."""
    script = tmp_path / "fake_images.py"
    script.write_text(FAKE)
    monkeypatch.setattr(images, "python", lambda: sys.executable)
    monkeypatch.setattr(images, "model_folder", lambda model: str(tmp_path / model))
    monkeypatch.setattr(images.worker, "command", lambda: [sys.executable, "-u", str(script)])
    with TestClient(create_app(token="t")) as c:
        yield c
    images.worker.stop()


def wait(client, job):
    for _ in range(200):
        done = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if done["status"] != "running":
            return done
        time.sleep(0.05)
    raise AssertionError("still running")


def asked(client, image):
    return json.loads(client.get(f"/api/images/gallery/{image}.png", headers=H).text)


def test_a_prompt_makes_images_with_the_models_defaults(client):
    done = wait(client, client.post("/api/images/generate", headers=H, json={"prompt": " a fox ", "count": 2, "seed": 5}).json())
    assert done["status"] == "done" and done["progress"] == 1.0
    made = client.get("/api/images/gallery", headers=H).json()["images"]
    assert sorted(i["seed"] for i in made) == [5, 6] and {i["id"] for i in made} == set(done["result"]["images"])
    first = asked(client, done["result"]["images"][0])
    assert first == {"prompt": "a fox", "width": 512, "height": 512, "steps": 4, "guidance": 8.0, "image": None, "strength": None, "variant": None, "lora": None}


def test_a_gallery_image_can_be_changed_and_deleted(client):
    done = wait(client, client.post("/api/images/generate", headers=H, json={"prompt": "a cabin", "model": "stabilityai/sd-turbo"}).json())
    source = done["result"]["images"][0]
    assert asked(client, source)["variant"] == "fp16" and asked(client, source)["steps"] == 1
    done = wait(client, client.post("/api/images/generate", headers=H, json={"prompt": "in spring", "image": source, "strength": 0.4}).json())
    changed = asked(client, done["result"]["images"][0])
    assert changed["image"].endswith(f"{source}.png") and changed["strength"] == 0.4 and changed["width"] is None
    assert client.delete(f"/api/images/gallery/{source}", headers=H).status_code == 200
    assert source not in {i["id"] for i in client.get("/api/images/gallery", headers=H).json()["images"]}
    assert client.get("/api/images/gallery/..%2Fsecret.png", headers=H).status_code == 404


def test_failures_and_bad_requests_say_why(client, monkeypatch):
    done = wait(client, client.post("/api/images/generate", headers=H, json={"prompt": "fail"}).json())
    assert done["status"] == "failed" and "out of memory" in done["error"]
    assert client.post("/api/images/generate", headers=H, json={"prompt": "x", "model": "someone/else"}).status_code == 400
    assert client.post("/api/images/generate", headers=H, json={"prompt": "x", "image": "/no/such.png"}).status_code == 404
    monkeypatch.setattr(images, "model_folder", lambda model: None)
    assert "isn't downloaded" in client.post("/api/images/generate", headers=H, json={"prompt": "x"}).json()["detail"]
    monkeypatch.setattr(images, "python", lambda: None)
    assert "isn't installed" in client.post("/api/images/generate", headers=H, json={"prompt": "x"}).json()["detail"]


def test_the_chosen_model_is_remembered(client):
    assert client.get("/api/images", headers=H).json()["model"] == images.MODELS[0]["id"]
    assert client.put("/api/images/model", headers=H, json={"model": "segmind/SSD-1B"}).json()["model"] == "segmind/SSD-1B"
    assert client.put("/api/images/model", headers=H, json={"model": "nope"}).status_code == 400


def test_openai_images_answers_with_base64(client):
    out = client.post("/v1/images/generations", headers=H, json={"prompt": "a boat", "n": 2, "size": "768x512"}).json()
    assert len(out["data"]) == 2 and out["data"][0]["revised_prompt"] == "a boat"
    sent = json.loads(base64.b64decode(out["data"][0]["b64_json"]))
    assert (sent["width"], sent["height"]) == (768, 512)
