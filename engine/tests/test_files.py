import json

from fastapi.testclient import TestClient

from temper_engine import home, library
from temper_engine.app import create_app

from .test_library import gguf

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def items():
    return {i["id"]: i for i in client.get("/api/files", headers=AUTH).json()["items"]}


def make_library():
    root = home()
    (root / "datasets").mkdir(exist_ok=True)
    (root / "datasets" / "math.jsonl").write_text('{"a":1}\n')
    (root / "datasets" / "math.eval.jsonl").write_text('{"a":2}\n')
    run = root / "runs" / "20261025-tutor"
    (run / "adapter").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"name": "Math tutor", "base_model": "Qwen/Qwen3-0.6B", "dataset": "math"}))
    (root / "exports" / "20261025-tutor-1").mkdir(parents=True, exist_ok=True)
    gguf(root / "exports" / "20261025-tutor-1" / "tutor-Q4_K_M.gguf")
    (root / "converted" / "Org-M").mkdir(parents=True, exist_ok=True)
    return gguf(root / "converted" / "Org-M" / "M-Q8_0.gguf")


def test_lists_every_kind_and_groups_eval_splits(tmp_path):
    make_library()
    (tmp_path / "notes.csv").write_text("q,a\n1,2\n")
    assert client.post("/api/files/upload", json={"paths": [str(tmp_path / "notes.csv")] * 2}, headers=AUTH).json()["added"] == ["uploads/notes.csv", "uploads/notes (2).csv"]
    found = items()
    assert found["datasets/math.jsonl"]["kind"] == "dataset" and len(found["datasets/math.jsonl"]["parts"]) == 2
    assert "datasets/math.eval.jsonl" not in found
    assert found["runs/20261025-tutor"]["name"] == "Math tutor" and found["runs/20261025-tutor"]["adapter"]
    assert found["exports/20261025-tutor-1"]["gguf"][0].endswith("tutor-Q4_K_M.gguf")
    assert found["converted/Org-M/M-Q8_0.gguf"]["kind"] == "model"
    assert found["uploads/notes.csv"]["dataset"]
    assert client.post("/api/files/upload", json={"paths": [str(tmp_path / "nope")]}, headers=AUTH).status_code == 400


def test_rename_favourite_and_folders():
    make_library()
    item = "datasets/math.jsonl"
    r = client.patch("/api/files", json={"id": item, "name": "  Grade-school maths ", "favourite": True}, headers=AUTH).json()
    assert r["name"] == "Grade-school maths" and r["favourite"] and r["original"] == "math"
    assert (home() / "datasets" / "math.jsonl").is_file()  # the file keeps its name
    assert client.patch("/api/files", json={"id": item, "folder": "Maths"}, headers=AUTH).status_code == 404
    client.post("/api/files/folders", json={"name": "Maths"}, headers=AUTH)
    assert client.post("/api/files/folders", json={"name": "Maths"}, headers=AUTH).status_code == 409
    client.patch("/api/files", json={"id": item, "folder": "Maths"}, headers=AUTH)
    client.put("/api/files/folders", json={"name": "Maths", "new_name": "Arithmetic"}, headers=AUTH)
    assert items()[item]["folder"] == "Arithmetic"
    client.delete("/api/files/folders", params={"name": "Arithmetic"}, headers=AUTH)
    assert items()[item]["folder"] is None and items()[item]["favourite"]
    client.patch("/api/files", json={"id": item, "name": "", "favourite": False}, headers=AUTH)
    assert items()[item]["name"] == "math"
    assert item not in json.loads((home() / "files.json").read_text())["items"]


def test_delete_removes_files_and_library_entries():
    converted = make_library()
    library.add_entry(str(converted), name="Org-M Q8_0")
    client.patch("/api/files", json={"id": "datasets/math.jsonl", "favourite": True}, headers=AUTH)
    assert client.delete("/api/files", params={"id": "datasets/math.jsonl"}, headers=AUTH).json()["freed"] > 0
    assert not (home() / "datasets" / "math.eval.jsonl").exists()
    client.delete("/api/files", params={"id": "converted/Org-M/M-Q8_0.gguf"}, headers=AUTH)
    assert not (home() / "converted" / "Org-M").exists()
    assert all(i["path"] != str(converted) for i in library._imports())
    client.delete("/api/files", params={"id": "runs/20261025-tutor"}, headers=AUTH)
    assert not (home() / "runs" / "20261025-tutor").exists()
    assert client.delete("/api/files", params={"id": "runs/20261025-tutor"}, headers=AUTH).status_code == 404


def test_pictures_speech_and_stems_are_listed_and_deleted_whole():
    root = home()
    (root / "images").mkdir(exist_ok=True)
    (root / "images" / "0123456789ab.png").write_bytes(b"png")
    (root / "images" / "0123456789ab.json").write_text(json.dumps({"prompt": "a fox in the snow", "model": "SimianLuo/LCM_Dreamshaper_v7"}))
    (root / "voice").mkdir(exist_ok=True)
    (root / "voice" / "abc.wav").write_bytes(b"wav")
    (root / "voice" / "abc.json").write_text(json.dumps({"text": "Hello there", "voice": "af_heart"}))
    song = root / "stems" / "20261031-120000-Song"
    song.mkdir(parents=True, exist_ok=True)
    for name in ("vocals", "drums"):
        (song / f"{name}.wav").write_bytes(b"x")
    (song / "source.json").write_text(json.dumps({"name": "My Song"}))
    found = items()
    picture, clip, stems = found["images/0123456789ab.png"], found["voice/abc.wav"], found["stems/20261031-120000-Song"]
    assert (picture["kind"], picture["name"], picture["size"], picture["made_with"]) == ("picture", "a fox in the snow", 3, "SimianLuo/LCM_Dreamshaper_v7")
    assert (clip["kind"], clip["name"], clip["voice"]) == ("speech", "Hello there", "af_heart")
    assert (stems["kind"], stems["name"], stems["stems"]) == ("stems", "My Song", ["drums", "vocals"])
    for item in (picture, clip, stems):
        client.delete("/api/files", params={"id": item["id"]}, headers=AUTH)
    assert not list((root / "images").iterdir()) and not list((root / "voice").iterdir()) and not song.exists()
