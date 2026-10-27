import json

import pytest
from fastapi.testclient import TestClient

from temper_engine import engines, model_settings, storage
from temper_engine.app import create_app
from temper_engine.runtime import LoadRequest, RuntimeManager

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}
SETTINGS = {**model_settings.DEFAULTS, "context": 16384, "cache_type": "q8_0", "tensor_parallel": 2, "pipeline_parallel": 2}


def folder(path, config, template="{{ '<think>' }}"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    return path


def test_formats_and_the_hugging_face_cache(tmp_path, monkeypatch):
    mlx = folder(tmp_path / "mlx", {"quantization": {"bits": 4}, "max_position_embeddings": 40960})
    hf = folder(tmp_path / "hf", {"text_config": {"max_position_embeddings": 8192}})
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    assert [engines.model_format(str(p)) for p in (mlx, hf, gguf)] == ["mlx", "hf", "gguf"]
    assert engines.context_max(str(mlx)) == 40960 and engines.context_max(str(hf)) == 8192
    # A repo id resolves to the snapshot main points at, from Temper's models folder.
    monkeypatch.setattr(storage, "models_dir", lambda: tmp_path / "cache")
    snap = folder(tmp_path / "cache/models--org--tiny/snapshots/abc", {})
    (tmp_path / "cache/models--org--tiny/refs").mkdir()
    (tmp_path / "cache/models--org--tiny/refs/main").write_text("abc")
    assert engines.local_folder("org/tiny") == str(snap)
    assert engines.local_folder("org/missing") is None and engines.local_folder(str(gguf)) is None


def test_server_commands(monkeypatch):
    monkeypatch.setattr(engines, "python_for", lambda e: "/py")
    assert engines.command("mlx", "/m", SETTINGS, 9, adapter="/run/adapter") == ["/py", "-m", "mlx_lm", "server", "--model", "/m", "--host", "127.0.0.1", "--port", "9", "--adapter-path", "/run/adapter", "--kv-bits", "8"]
    vllm = engines.command("vllm", "/m", SETTINGS, 9)
    assert vllm[:3] == ["/py", "-m", "vllm.entrypoints.openai.api_server"] and "--kv-cache-dtype" in vllm
    assert vllm[vllm.index("--tensor-parallel-size") + 1] == "2" and vllm[vllm.index("--pipeline-parallel-size") + 1] == "2" and vllm[vllm.index("--max-model-len") + 1] == "16384"
    sglang = engines.command("sglang", "/m", SETTINGS, 9, adapter="/a")
    assert sglang[sglang.index("--tp-size") + 1] == "2" and sglang[sglang.index("--lora-paths") + 1] == "temper=/a"
    monkeypatch.setattr(engines, "python_for", lambda e: None)
    with pytest.raises(Exception, match="isn't installed|needs Linux"):
        engines.command("vllm", "/m", SETTINGS, 9)


def test_requests_are_translated_for_each_server():
    body = {"model": "chat", "repeat_penalty": 1.1, "cache_prompt": True, "thinking_budget_tokens": 10, "messages": []}
    assert engines.translate("llama.cpp", body) is body
    assert engines.translate("mlx", body) == {"repetition_penalty": 1.1, "messages": []}
    assert engines.translate("vllm", body)["model"] == "temper"


def test_loading_a_folder_starts_its_engine(tmp_path, monkeypatch):
    mlx = folder(tmp_path / "mlx", {"quantization": {"bits": 4}})
    adapter = tmp_path / "run/adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    started = []

    class Proc:
        stdout = iter(())

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(engines, "python_for", lambda e: "/py" if e == "mlx" else None)
    monkeypatch.setattr(engines, "supported", lambda e: None if e == "mlx" else "no")
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: started.append(cmd) or Proc())
    server = RuntimeManager()
    monkeypatch.setattr(RuntimeManager, "_watch", lambda self, proc: None)
    server.load(LoadRequest(path=str(mlx), name="run", adapter=str(adapter)))
    assert server.engine == "mlx" and server.adapter == str(adapter) and "--adapter-path" in started[-1]
    server.settings = SETTINGS
    server._ready_other()
    assert server.state == "ready" and server.reasoning["thinks"] and server.modalities == {"vision": False, "audio": False}
    with pytest.raises(Exception, match="adapter"):
        server.load(LoadRequest(path=str(mlx), adapter=str(tmp_path / "gone")))
    with pytest.raises(Exception, match="format"):
        server.load(LoadRequest(path=str(mlx), engine="vllm"))
    server.unload()
    assert server.engine == "llama.cpp"


def test_engines_list_and_install_guard():
    out = client.get("/api/runtime/engines", headers=AUTH).json()["engines"]
    assert [e["id"] for e in out] == ["llama.cpp", "mlx", "transformers", "vllm", "sglang"]
    assert client.post("/api/runtime/engines/llama.cpp/install", headers=AUTH).status_code == 404
