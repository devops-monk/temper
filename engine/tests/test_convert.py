import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from temper_engine import convert, library, llamacpp
from temper_engine.app import create_app
from temper_engine.library import quantization

from .test_library import gguf

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}


def folder(path, config=None, files=("model.safetensors", "tokenizer.json")):
    path.mkdir(parents=True)
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))
    for f in files:
        (path / f).write_bytes(b"x" * 100)
    return path


def q(config, tmp_path=Path(".")):
    return quantization({"quantization_config": config} if config else {}, tmp_path)


def test_quantization_labels(tmp_path):
    assert q({}) is None
    assert q({"quant_method": "fp8", "weight_block_size": [128, 128]}) == {"label": "FP8", "convertible": True}
    assert q({"quant_method": "modelopt", "quant_algo": "NVFP4"})["label"] == "NVFP4 (ModelOpt)"
    nvfp4 = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "config_groups": {"g": {}}}
    assert q(nvfp4) == {"label": "NVFP4 (compressed-tensors)", "convertible": True}
    fp8 = {"quant_method": "compressed-tensors", "format": "float-quantized", "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}}
    assert q(fp8) == {"label": "FP8 (compressed-tensors)", "convertible": True}
    mixed = {**fp8, "config_groups": {"a": {"weights": {"num_bits": 8}}, "b": {"weights": {"num_bits": 4}}}}
    assert not q(mixed)["convertible"]
    assert q({"quant_method": "gptq", "bits": 4})["label"] == "GPTQ 4-bit"
    assert not q({"quant_method": "gptq", "bits": 3})["convertible"]
    assert not q({"quant_method": "awq"})["convertible"]
    # MXFP4 only for architectures that keep it natively.
    assert quantization({"quantization_config": {"quant_method": "mxfp4"}, "architectures": ["GptOssForCausalLM"]}, tmp_path)["convertible"]
    assert not quantization({"quantization_config": {"quant_method": "mxfp4"}, "architectures": ["LlamaForCausalLM"]}, tmp_path)["convertible"]
    # ModelOpt can keep its settings beside config.json instead of in it.
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8"}}))
    assert quantization({}, tmp_path)["label"] == "FP8 (ModelOpt)"


def test_import_reports_quantization_and_mistral_format(tmp_path):
    fp8 = folder(tmp_path / "fp8", {"model_type": "qwen3", "quantization_config": {"quant_method": "fp8"}})
    r = client.post("/api/models/import", json={"path": str(fp8)}, headers=AUTH).json()
    assert r["quantization"]["label"] == "FP8" and r["convertible"]
    awq = folder(tmp_path / "awq", {"model_type": "qwen3", "quantization_config": {"quant_method": "awq"}})
    r = client.post("/api/models/import", json={"path": str(awq)}, headers=AUTH).json()
    assert not r["convertible"] and "AWQ" in r["warnings"][-1]
    mistral = folder(tmp_path / "mistral", None, ("params.json", "consolidated.safetensors", "tekken.json"))
    r = client.post("/api/models/import", json={"path": str(mistral)}, headers=AUTH).json()
    assert r["kind"] == "mistral" and r["convertible"]
    mlx = folder(tmp_path / "mlx", {"model_type": "qwen3", "quantization": {"bits": 4}})
    assert not library.checkpoint(mlx)["convertible"]


def test_convert_refuses_what_it_cant_read(tmp_path):
    awq = folder(tmp_path / "awq", {"model_type": "qwen3", "quantization_config": {"quant_method": "awq"}})
    r = client.post("/api/models/convert", json={"path": str(awq)}, headers=AUTH)
    assert r.status_code == 400 and "AWQ" in r.json()["detail"]
    r = client.post("/api/models/convert", json={"path": str(gguf(tmp_path / "m.gguf"))}, headers=AUTH)
    assert "already a GGUF" in r.json()["detail"]
    r = client.post("/api/models/convert", json={"path": str(awq), "quant": "q2"}, headers=AUTH)
    assert r.status_code == 400


def test_convert_runs_the_converter_then_quantizes_and_imports(tmp_path, monkeypatch):
    src = folder(tmp_path / "Org-Model-FP8", {"model_type": "qwen3", "quantization_config": {"quant_method": "fp8"}})
    # A transformers 5 tokenizer config: the converter must read a fixed copy, and the source stays as it was.
    (src / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": ["<a>"]}))
    seen = {}

    def fake_convert(model_dir, outfile, outtype="f16", mistral_format=False):
        seen.update(dir=model_dir, outtype=outtype, mistral=mistral_format)
        seen["tokenizer"] = json.loads((model_dir / "tokenizer_config.json").read_text())
        script = f"import sys; print('Writing:  50%|'); open({str(outfile)!r}, 'wb').write(open({str(gguf(tmp_path / 'g.gguf'))!r}, 'rb').read())"
        return [__import__("sys").executable, "-c", script], None

    def fake_quantize(src_file, dst, quant):
        return [__import__("sys").executable, "-c", f"import shutil; print('[ 1/ 2]'); shutil.copy({str(src_file)!r}, {str(dst)!r})"]

    monkeypatch.setattr(llamacpp, "install", lambda report: None)
    monkeypatch.setattr(llamacpp, "convert_command", fake_convert)
    monkeypatch.setattr(llamacpp, "quantize_command", fake_quantize)
    r = client.post("/api/models/convert", json={"path": str(src)}, headers=AUTH)
    assert r.status_code == 200, r.text
    job = convert.registry.get(r.json()["id"])
    for _ in range(100):
        if job.status != "running":
            break
        time.sleep(0.05)
    assert job.status == "done", job.error
    assert seen["outtype"] == "f16" and not seen["mistral"] and seen["dir"] != src
    assert seen["tokenizer"]["additional_special_tokens"] == ["<a>"]
    assert "extra_special_tokens" in json.loads((src / "tokenizer_config.json").read_text())
    out = Path(job.result["gguf"])
    assert out.name == "Org-Model-FP8-Q4_K_M.gguf" and out.is_file()
    assert [p.name for p in out.parent.iterdir()] == [out.name]  # the 16-bit file and scratch copy are gone
    assert library.entries()[0]["id"] == "Org-Model-FP8 Q4_K_M"
