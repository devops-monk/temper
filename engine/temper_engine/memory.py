"""Memory estimator: will this model fit for chat (llama.cpp) or for training, before we try?

Estimates come from the model's shape (layers, heads, hidden size, vocabulary) read from its
config.json or GGUF header, plus constants measured on real runs (see tests/test_memory.py). They are
meant to be within about 15%, which is enough to warn before running out of memory.
"""

from __future__ import annotations

import json
import platform
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import httpx
import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import hardware, home

router = APIRouter(prefix="/api/memory", tags=["memory"])

GB = 1024**3
MB = 1024**2


# ------------------------------------------------------------------ model shape


@dataclass
class Shape:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    params: int  # total parameters
    context_max: int | None = None
    # Layers that keep a KV cache; hybrid models (linear attention, sliding windows) keep fewer.
    kv_layers: int | None = None
    experts: int = 0

    @property
    def cached_layers(self) -> int:
        return self.kv_layers or self.layers


def _params_from_config(c: dict) -> int:
    hidden, layers, vocab = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    heads = c["num_attention_heads"]
    kv = c.get("num_key_value_heads") or heads
    hd = c.get("head_dim") or hidden // heads
    attn = hidden * heads * hd * 2 + hidden * kv * hd * 2
    experts = c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts") or 0
    if experts:
        moe_inter = c.get("moe_intermediate_size") or c["intermediate_size"]
        shared = (c.get("n_shared_experts") or 0) * 3 * hidden * moe_inter + (3 * hidden * c["shared_expert_intermediate_size"] if c.get("shared_expert_intermediate_size") else 0)
        mlp = experts * 3 * hidden * moe_inter + shared
    else:
        mlp = 3 * hidden * c["intermediate_size"]
    embed = vocab * hidden * (1 if c.get("tie_word_embeddings") else 2)
    return layers * (attn + mlp) + embed


def shape_from_config(c: dict) -> Shape:
    c = {**c, **c.get("text_config", {})}  # multimodal models nest the language model's settings
    heads = c["num_attention_heads"]
    layer_types = c.get("layer_types") or []
    full = sum(t in ("full_attention", "attention") for t in layer_types) if layer_types else None
    return Shape(
        layers=c["num_hidden_layers"],
        hidden=c["hidden_size"],
        heads=heads,
        kv_heads=c.get("num_key_value_heads") or heads,
        head_dim=c.get("head_dim") or c["hidden_size"] // heads,
        intermediate=c.get("moe_intermediate_size") or c.get("intermediate_size", 4 * c["hidden_size"]),
        vocab=c["vocab_size"],
        params=_params_from_config(c),
        context_max=c.get("max_position_embeddings"),
        kv_layers=full or None,
        experts=c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts") or 0,
    )


def _hf_config(repo_or_dir: str) -> dict:
    local = Path(repo_or_dir).expanduser()
    if (local / "config.json").is_file():
        return json.loads((local / "config.json").read_text())
    from huggingface_hub import hf_hub_download

    # A separate cache: config files alone in the main one would show up as half-downloaded models.
    path = hf_hub_download(repo_or_dir, "config.json", cache_dir=home() / "cache" / "configs")
    return json.loads(Path(path).read_text())


def weight_bits(config: dict) -> float:
    """Bits per weight as stored: MLX and bitsandbytes checkpoints are quantized, others are 16-bit."""
    q = config.get("quantization") or config.get("quantization_config") or {}
    bits = q.get("bits") or (4 if q.get("load_in_4bit") else 8 if q.get("load_in_8bit") else None)
    if bits:
        # MLX stores a 16-bit scale and bias for every group of weights.
        return bits + 32 / q.get("group_size", 64)
    return 16.0


# ------------------------------------------------------------------ GGUF header

_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, read: Callable[[int], bytes]):
        self._read = read

    def take(self, n: int) -> bytes:
        data = self._read(n)
        if len(data) < n:
            raise ValueError("truncated GGUF header")
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8", "replace")

    def value(self, kind: int):
        if kind == 8:
            return self.string()
        if kind == 9:
            item, count = self.unpack("<I"), self.unpack("<Q")
            if item in _GGUF_SCALARS and count <= 4096:  # small numeric arrays, e.g. per-layer head counts
                return [self.value(item) for _ in range(count)]
            for _ in range(count):  # tokenizer vocabularies: skip
                self.value(item)
            return None
        return self.unpack(_GGUF_SCALARS[kind])


def read_gguf_metadata(read: Callable[[int], bytes]) -> dict:
    """Key/value metadata from a GGUF header, up to the tokenizer's token list.

    Writers put the model's own keys first and the tokenizer (megabytes of vocabulary) last, so this
    stops at the token list and records only its length, the vocabulary size.
    """
    r = _Reader(read)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = r.unpack("<I")
    if version < 2:
        raise ValueError(f"GGUF version {version} is too old")
    r.unpack("<Q")  # tensor count
    meta: dict = {}
    for _ in range(r.unpack("<Q")):
        key = r.string()
        kind = r.unpack("<I")
        if key == "tokenizer.ggml.tokens" and kind == 9:
            r.unpack("<I")
            meta["tokenizer.ggml.tokens.count"] = r.unpack("<Q")
            break
        meta[key] = r.value(kind)
    return meta


def shape_from_gguf(meta: dict, file_size: int) -> Shape:
    arch = meta["general.architecture"]
    get = lambda k, d=None: meta.get(f"{arch}.{k}", d)  # noqa: E731
    layers, hidden = get("block_count"), get("embedding_length")
    heads = get("attention.head_count")
    kv = get("attention.head_count_kv", heads)
    heads = max(heads) if isinstance(heads, list) else heads
    # Per-layer KV head counts: layers with 0 have no KV cache (recurrent or linear-attention layers).
    kv_layers = None
    if isinstance(kv, list):
        kv_layers = sum(1 for h in kv if h) or None
        kv = max(kv)
    if get("full_attention_interval"):
        kv_layers = layers // get("full_attention_interval")
    head_dim = get("attention.key_length") or hidden // heads
    return Shape(
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv,
        head_dim=head_dim,
        intermediate=get("feed_forward_length", 4 * hidden) if not isinstance(get("feed_forward_length"), list) else max(get("feed_forward_length")),
        vocab=meta.get("tokenizer.ggml.tokens.count") or get("vocab_size", 0),
        params=0,
        context_max=get("context_length"),
        kv_layers=kv_layers,
        experts=get("expert_count", 0) or 0,
    )


def gguf_shape_local(path: Path) -> Shape:
    with path.open("rb") as f:
        meta = read_gguf_metadata(f.read)
    return shape_from_gguf(meta, path.stat().st_size)


def gguf_shape_remote(repo_id: str, filename: str) -> Shape:
    """Reads only the header of a GGUF on Hugging Face, with ranged requests (usually one or two)."""
    from huggingface_hub import get_token, hf_hub_url

    url = hf_hub_url(repo_id, filename)
    headers = {"Authorization": f"Bearer {get_token()}"} if get_token() else {}
    chunk, limit = 512 * 1024, 32 * MB
    buf, pos = bytearray(), 0

    with httpx.Client(follow_redirects=True, timeout=20) as client:

        def read(n: int) -> bytes:
            nonlocal pos
            while len(buf) < pos + n:
                if len(buf) >= limit:
                    raise ValueError("GGUF header is larger than expected")
                r = client.get(url, headers={**headers, "Range": f"bytes={len(buf)}-{len(buf) + chunk - 1}"})
                r.raise_for_status()
                if not r.content:
                    break
                buf.extend(r.content)
            data = bytes(buf[pos : pos + n])
            pos += len(data)
            return data

        meta = read_gguf_metadata(read)
    return shape_from_gguf(meta, 0)


# ------------------------------------------------------------------ budget


def memory_budget() -> dict:
    """How much memory a model can use here. On Apple Silicon the GPU may wire only part of unified memory."""
    vm = psutil.virtual_memory()
    gpus = hardware.nvidia_gpus() or hardware.amd_gpus()
    if gpus:
        g = gpus[0]
        return {"kind": "vram", "device": g["name"], "total": g["memory_total"], "free": g["memory_total"] - g["memory_used"]}
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        limit = 0
        out = hardware._run(["sysctl", "-n", "iogpu.wired_limit_mb"])
        if out and out.strip().isdigit():
            limit = int(out.strip()) * MB
        if not limit:
            # macOS's default GPU working-set limit: about two thirds of memory, three quarters above 36 GB.
            limit = int(vm.total * (0.75 if vm.total > 36 * GB else 0.67))
        return {"kind": "unified", "device": hardware.cpu_name(), "total": limit, "free": min(limit, vm.available)}
    return {"kind": "ram", "device": hardware.cpu_name(), "total": vm.total, "free": vm.available}


Verdict = Literal["fits", "tight", "too_large"]


def verdict(total: int, budget: dict) -> Verdict:
    if total <= budget["free"] * 0.9:
        return "fits"
    if total <= budget["total"]:
        return "tight"  # fits the machine, but not with what's open right now
    return "too_large"


# ------------------------------------------------------------------ chat (llama.cpp)


def kv_cache_bytes(shape: Shape, context: int, bytes_per_value: float = 2.0) -> int:
    """K and V for every cached layer at f16, which llama.cpp uses by default."""
    return int(2 * shape.cached_layers * shape.kv_heads * shape.head_dim * context * bytes_per_value)


def inference_estimate(shape: Shape, weights: int, context: int) -> dict:
    kv = kv_cache_bytes(shape, context)
    # llama.cpp's compute buffer: logits for a 512-token batch plus scratch space. Measured on an M1 Pro
    # with Qwen3-4B Q4_K_M and flash attention: about 100 MB at 8K and 32K context; this keeps a margin.
    compute = int(128 * MB + 512 * (shape.vocab or 150_000) * 2)
    return {"weights": weights, "kv_cache": kv, "overhead": compute, "total": weights + kv + compute}


# ------------------------------------------------------------------ training


def lora_params(shape: Shape, rank: int, layers: int) -> int:
    q, kv, o = shape.heads * shape.head_dim, shape.kv_heads * shape.head_dim, shape.hidden
    h, i = shape.hidden, shape.intermediate
    per_layer = rank * ((h + q) + 2 * (h + kv) + (q + o) + 2 * (h + i) + (i + h))
    return per_layer * layers


def training_estimate(shape: Shape, bits: float, trainer: str, method: str, rank: int, batch: int, seq: int) -> dict:
    tokens = batch * seq
    if trainer == "mlx":
        # mlx-lm trains the model as stored (quantized or 16-bit) with LoRA on the last 16 layers, and
        # recomputes activations (--grad-checkpoint). Constants measured on an M1 Pro.
        weights = int(shape.params * bits / 8)
        adapters = lora_params(shape, rank, min(16, shape.layers))
        optimizer = adapters * 12  # fp32 weights + grads + two Adam moments
        activations = int(tokens * (shape.vocab * MLX_LOGIT_BYTES + shape.hidden * shape.layers * MLX_LAYER_BYTES) + tokens**2 * MLX_PAIR_BYTES)
        overhead = MLX_OVERHEAD
    else:
        # PyTorch on NVIDIA/AMD: QLoRA loads 4-bit weights, LoRA 16-bit; activations are checkpointed to
        # system memory and the loss is computed in chunks, so the GPU keeps little per token.
        weights = int(shape.params * (4.5 if method == "qlora" else 16) / 8) if method != "full" else shape.params * 2
        adapters = shape.params if method == "full" else lora_params(shape, rank, shape.layers)
        optimizer = adapters * (2 + 2 + 2) if method != "full" else shape.params * (2 + 2)  # grads + 8-bit Adam
        activations = int(tokens * (shape.hidden * 2 * 2 + shape.intermediate * 6 + shape.vocab * 0.5))
        overhead = int(1.0 * GB)  # CUDA context, kernels, allocator slack
    total = weights + optimizer + activations + overhead
    return {"weights": weights, "adapters_and_optimizer": optimizer, "activations": activations, "overhead": overhead, "total": total}


# Fitted to peak memory of mlx-lm LoRA runs (rank 16, gradient checkpointing) on Qwen3-0.6B and
# Qwen3-4B at 4-bit on an M1 Pro; tests/test_memory.py keeps the estimate within 10% of each run.
MLX_LOGIT_BYTES = 2.0  # per vocabulary entry per token: the logits in 16-bit
MLX_LAYER_BYTES = 1.4  # per hidden unit per layer per token: checkpointed layer inputs
MLX_PAIR_BYTES = 68.0  # per token squared, the same for both models
MLX_OVERHEAD = int(1.25 * GB)  # Metal buffers and caches that don't depend on the model


# ------------------------------------------------------------------ suggestions


def _gb(n: int) -> str:
    return f"{n / GB:.1f} GB"


def chat_advice(est: dict, budget: dict, context: int) -> list[str]:
    tips = []
    if est["kv_cache"] > est["weights"] * 0.5 and context > 4096:
        tips.append(f"The {context:,}-token context alone needs {_gb(est['kv_cache'])}; a shorter context frees most of that.")
    if est["weights"] > budget["total"] * 0.7:
        tips.append("Pick a smaller quantization (for example Q4_K_M instead of Q8_0) or a smaller model.")
    if budget["kind"] == "unified" and est["total"] > budget["free"]:
        tips.append("Close other apps to free memory; macOS lets the GPU use about two thirds of it.")
    return tips


def training_advice(est: dict, budget: dict, trainer: str, method: str, batch: int, seq: int) -> list[str]:
    tips = []
    if est["activations"] > est["weights"] * 0.4:
        if batch > 1:
            tips.append(f"Use batch size 1 and raise gradient accumulation to keep the same effective batch (saves about {_gb(est['activations'] * (batch - 1) // batch)}).")
        if seq > 1024:
            tips.append("Shorten the maximum sequence length if your examples are shorter than it.")
    if trainer != "mlx" and method == "lora":
        tips.append("QLoRA loads the model in 4-bit and needs about a third of the memory for weights.")
    if trainer == "mlx" and est["weights"] > budget["total"] * 0.5:
        tips.append("Start from a 4-bit MLX version of the model (mlx-community/...-4bit).")
    return tips


# ------------------------------------------------------------------ API


class ChatEstimate(BaseModel):
    path: str | None = None  # a local GGUF
    repo_id: str | None = None  # or a GGUF on Hugging Face
    filename: str | None = None
    context: int = Field(default=8192, ge=256, le=1_048_576)


@router.post("/chat")
def estimate_chat(req: ChatEstimate):
    try:
        if req.path:
            path = Path(req.path).expanduser()
            shape, weights = gguf_shape_local(path), path.stat().st_size
        elif req.repo_id and req.filename:
            from huggingface_hub import HfApi

            info = HfApi().model_info(req.repo_id, files_metadata=True)
            size = next((s.size for s in info.siblings or [] if s.rfilename == req.filename), None)
            if size is None:
                raise HTTPException(404, f"{req.filename} isn't in {req.repo_id}")
            shape, weights = gguf_shape_remote(req.repo_id, req.filename), size
        else:
            raise HTTPException(400, "Give a local GGUF path, or a repo_id and filename.")
    except (OSError, ValueError, httpx.HTTPError) as e:
        raise HTTPException(422, f"Couldn't read the model's header: {e}") from e
    context = min(req.context, shape.context_max or req.context)
    est = inference_estimate(shape, weights, context)
    budget = memory_budget()
    return {**est, "context": context, "budget": budget, "verdict": verdict(est["total"], budget), "advice": chat_advice(est, budget, context), "shape": asdict(shape)}


class TrainEstimate(BaseModel):
    base_model: str
    method: Literal["lora", "qlora", "full"] = "qlora"
    lora_r: int = Field(default=16, ge=1, le=512)
    batch_size: int = Field(default=2, ge=1)
    max_seq_length: int = Field(default=2048, ge=128)


@router.post("/train")
def estimate_train(req: TrainEstimate):
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        config = _hf_config(req.base_model)
        shape = shape_from_config(config)
    except GatedRepoError as e:
        raise HTTPException(422, f"{req.base_model} is gated. Accept its licence on Hugging Face and add a token to estimate it.") from e
    except RepositoryNotFoundError as e:
        raise HTTPException(422, f"There's no model called {req.base_model} on Hugging Face, or it's private.") from e
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"{req.base_model} has a configuration Temper can't estimate yet (missing {e}).") from e
    except (OSError, ValueError, httpx.HTTPError) as e:
        raise HTTPException(422, f"Couldn't read {req.base_model}'s configuration: {str(e).splitlines()[0]}") from e
    backend = hardware.probe_backend()
    trainer = "mlx" if backend == "metal" else "pytorch"
    est = training_estimate(shape, weight_bits(config), trainer, req.method, req.lora_r, req.batch_size, req.max_seq_length)
    budget = memory_budget()
    return {
        **est,
        "trainer": trainer,
        "measured": trainer == "mlx",  # the PyTorch constants come from published figures, not our own runs
        "budget": budget,
        "verdict": verdict(est["total"], budget),
        "advice": training_advice(est, budget, trainer, req.method, req.batch_size, req.max_seq_length),
        "shape": asdict(shape),
    }

