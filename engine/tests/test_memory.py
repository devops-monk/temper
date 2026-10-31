import io
import struct

import pytest

from temper_engine import memory as m

# Shapes from the models' config.json files.
QWEN3_06B = {"hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
             "intermediate_size": 3072, "vocab_size": 151936, "tie_word_embeddings": True, "max_position_embeddings": 40960,
             "quantization": {"bits": 4, "group_size": 64}}
QWEN3_4B = {"hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 9728, "vocab_size": 151936, "tie_word_embeddings": True, "max_position_embeddings": 262144,
            "quantization": {"bits": 4, "group_size": 64}}

# Peak memory (mx.get_peak_memory) of mlx-lm LoRA training, rank 16, --grad-checkpoint, M1 Pro 16 GB.
MEASURED = [
    (QWEN3_06B, 1, 1024, 2.077), (QWEN3_06B, 2, 1024, 2.611), (QWEN3_06B, 4, 1024, 4.090),
    (QWEN3_4B, 1, 1024, 3.881), (QWEN3_4B, 2, 1024, 4.483), (QWEN3_4B, 4, 1024, 6.073),
    (QWEN3_4B, 1, 2048, 4.484), (QWEN3_4B, 2, 2048, 6.075), (QWEN3_4B, 1, 3001, 5.177),
]


def test_params_from_config():
    assert m.shape_from_config(QWEN3_4B).params == pytest.approx(4.02e9, rel=0.01)
    assert m.shape_from_config(QWEN3_06B).params == pytest.approx(0.596e9, rel=0.01)


@pytest.mark.parametrize("config,batch,seq,peak_gb", MEASURED)
def test_mlx_estimate_matches_measurements(config, batch, seq, peak_gb):
    est = m.training_estimate(m.shape_from_config(config), m.weight_bits(config), "mlx", "lora", 16, batch, seq)
    assert est["total"] / m.GB == pytest.approx(peak_gb, rel=0.10)


def test_kv_cache_matches_llama_cpp():
    # llama-server with Qwen3-4B Q4_K_M: about 1.1 GB at 8K and 4.5 GB at 32K context (f16 cache).
    shape = m.shape_from_config(QWEN3_4B)
    assert m.kv_cache_bytes(shape, 8192) / m.GB == pytest.approx(1.125, rel=0.01)
    assert m.kv_cache_bytes(shape, 32768) / m.GB == pytest.approx(4.5, rel=0.01)


def _gguf(meta: dict) -> bytes:
    out = io.BytesIO()
    out.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta)))
    for key, value in meta.items():
        out.write(struct.pack("<Q", len(key)) + key.encode())
        if isinstance(value, str):
            out.write(struct.pack("<IQ", 8, len(value)) + value.encode())
        elif isinstance(value, list) and isinstance(value[0], str):
            out.write(struct.pack("<IIQ", 9, 8, len(value)))
            for v in value:
                out.write(struct.pack("<Q", len(v)) + v.encode())
        elif isinstance(value, list):
            out.write(struct.pack("<IIQ", 9, 4, len(value)) + b"".join(struct.pack("<I", v) for v in value))
        else:
            out.write(struct.pack("<II", 4, value))
    return out.getvalue()


def test_gguf_header_with_hybrid_layers():
    data = _gguf({
        "general.architecture": "qwen35",
        "qwen35.block_count": 32,
        "qwen35.embedding_length": 4096,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": [0, 0, 0, 4] * 8,  # only every fourth layer keeps a KV cache
        "qwen35.attention.key_length": 256,
        "qwen35.context_length": 262144,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": ["a", "b", "c"],
    })
    shape = m.shape_from_gguf(m.read_gguf_metadata(io.BytesIO(data).read), len(data))
    assert (shape.layers, shape.kv_heads, shape.head_dim, shape.kv_layers, shape.vocab) == (32, 4, 256, 8, 3)


def test_verdicts():
    budget = {"kind": "unified", "device": "M1 Pro", "total": 11 * m.GB, "free": 8 * m.GB}
    assert m.verdict(5 * m.GB, budget) == "fits"
    assert m.verdict(10 * m.GB, budget) == "tight"
    assert m.verdict(12 * m.GB, budget) == "too_large"


def test_training_setups_are_fitted_to_the_memory_there_is():
    def fit(config, gb, method="lora", trainer="mlx"):
        budget = {"kind": "unified", "total": int(gb * m.GB), "free": int(gb * m.GB)}
        return m.training_fit(m.shape_from_config(config), m.weight_bits(config), trainer, method, budget)

    roomy = fit(QWEN3_4B, 64)
    assert (roomy["max_seq_length"], roomy["lora_r"], roomy["batch_size"], roomy["grad_accum"]) == (4096, 16, 4, 2) and roomy["fits"]
    mac16 = fit(QWEN3_4B, 10.7)  # a 16 GB Mac's GPU share: the page's defaults
    assert (mac16["max_seq_length"], mac16["lora_r"], mac16["batch_size"], mac16["grad_accum"]) == (2048, 16, 2, 4)
    assert mac16["fits"] and mac16["total"] <= 10.7 * m.GB * m.FIT_SHARE
    small = fit(QWEN3_4B, 5.5)
    assert (small["max_seq_length"], small["lora_r"], small["batch_size"], small["grad_accum"]) == (1024, 16, 1, 8) and small["fits"]
    hopeless = fit(QWEN3_4B, 2)
    assert not hopeless["fits"] and (hopeless["max_seq_length"], hopeless["lora_r"], hopeless["batch_size"]) == (512, 8, 1)
    short = fit({**QWEN3_06B, "max_position_embeddings": 1024}, 64)
    assert short["max_seq_length"] == 1024


def test_sliding_window_layers_cache_only_their_window():
    base = {"hidden_size": 2560, "num_hidden_layers": 34, "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
            "intermediate_size": 10240, "vocab_size": 262144, "sliding_window": 1024}
    gemma3 = m.shape_from_config({**base, "model_type": "gemma3_text", "sliding_window_pattern": 6})
    assert (gemma3.kv_layers, gemma3.sliding_layers, gemma3.sliding_window) == (5, 29, 1024)
    full = m.shape_from_config({k: v for k, v in base.items() if k != "sliding_window"})
    per_token = 2 * 4 * 256 * 2
    assert m.kv_cache_bytes(full, 32768) == 34 * 32768 * per_token
    assert m.kv_cache_bytes(gemma3, 32768) == (5 * 32768 + 29 * 1024) * per_token
    assert m.kv_cache_bytes(gemma3, 512) == 34 * 512 * per_token  # a context shorter than the window
    typed = m.shape_from_config({**base, "layer_types": ["sliding_attention", "full_attention"] * 17})
    assert (typed.kv_layers, typed.sliding_layers) == (17, 17)
    qwen = m.shape_from_config({**base, "model_type": "qwen2"})  # its window is off unless use_sliding_window
    assert (qwen.kv_layers, qwen.sliding_layers) == (None, 0)
    meta = {"general.architecture": "gemma3", "gemma3.block_count": 34, "gemma3.embedding_length": 2560, "gemma3.attention.head_count": 8,
            "gemma3.attention.head_count_kv": 4, "gemma3.attention.key_length": 256, "gemma3.attention.sliding_window": 1024}
    g = m.shape_from_gguf(meta, 0)
    assert (g.kv_layers, g.sliding_layers, g.sliding_window) == (5, 29, 1024)
