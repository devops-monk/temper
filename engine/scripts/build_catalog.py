"""Builds temper_engine/catalog.json: the curated model families on the Models page.

The choice of families and variants, and the words about them, are ours (FAMILIES below). Every fact
is read from the Hub when this runs: size, context length, licence, gating, whether it sees images,
the publisher's sampling settings (generation_config.json), and a GGUF and MLX conversion from a
publisher we trust. Re-run it when new models come out:

    uv run python scripts/build_catalog.py
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jinja2
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temper_engine.memory import gguf_shape_remote  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "temper_engine" / "catalog.json"
GGUF_PUBLISHERS = ("ggml-org", "lmstudio-community", "bartowski")  # after the model's own publisher
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")

FAMILIES: list[dict] = [
    {
        "id": "qwen",
        "name": "Qwen",
        "publisher": "Alibaba",
        "about": "Strong all-rounders in every size, with a thinking mode for harder problems. Qwen3.5 and later also read images.",
        "uses": ["chat", "reasoning", "code", "tools"],
        "variants": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3-4B-Instruct-2507"],
    },
    {
        "id": "gemma",
        "name": "Gemma",
        "publisher": "Google",
        "about": "Google's open models. Gemma 4 reads images; the E2B and E4B sizes are built for laptops and phones.",
        "uses": ["chat", "vision", "multilingual"],
        "variants": ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-12B-it", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it"],
    },
    {
        "id": "llama",
        "name": "Llama",
        "publisher": "Meta",
        "about": "The most widely fine-tuned open models, with the largest ecosystem of guides and adapters. Meta asks you to accept its licence first.",
        "uses": ["chat", "fine-tuning"],
        "variants": ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"],
    },
    {
        "id": "mistral",
        "name": "Mistral",
        "publisher": "Mistral AI",
        "about": "Fast, compact models from Paris. Ministral 3 covers small devices; Mistral Small and Devstral are strong at 24B, Devstral for coding agents.",
        "uses": ["chat", "code", "tools"],
        "variants": ["mistralai/Ministral-3-3B-Instruct-2512", "mistralai/Ministral-3-8B-Instruct-2512", "mistralai/Ministral-3-14B-Instruct-2512", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistralai/Devstral-Small-2-24B-Instruct-2512"],
    },
    {
        "id": "gpt-oss",
        "name": "gpt-oss",
        "publisher": "OpenAI",
        "about": "OpenAI's open-weight reasoning models. The 20B runs on a 16 GB machine; you can set how hard it thinks.",
        "uses": ["reasoning", "tools"],
        "variants": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    },
    {
        "id": "glm",
        "name": "GLM",
        "publisher": "Z.ai",
        "about": "Mixture-of-experts models tuned for coding and agent work, with long context.",
        "uses": ["code", "tools", "reasoning"],
        "variants": ["zai-org/GLM-4.7-Flash", "zai-org/GLM-4.5-Air"],
    },
    {
        "id": "kimi",
        "name": "Kimi",
        "publisher": "Moonshot AI",
        "about": "Moonshot's long-context model: a mixture of experts with only 3B parameters active per token, so it runs faster than its size suggests.",
        "uses": ["chat", "long context"],
        "variants": ["moonshotai/Kimi-Linear-48B-A3B-Instruct"],
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "publisher": "DeepSeek",
        "about": "Reasoning models that show their working. The distilled versions bring that to sizes a laptop can run.",
        "uses": ["reasoning", "math", "code"],
        "variants": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"],
    },
    {
        "id": "phi",
        "name": "Phi",
        "publisher": "Microsoft",
        "about": "Small models trained on carefully chosen data, good at maths and reasoning for their size.",
        "uses": ["reasoning", "math"],
        "variants": ["microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning", "microsoft/phi-4", "microsoft/Phi-4-reasoning-plus"],
    },
    {
        "id": "granite",
        "name": "Granite",
        "publisher": "IBM",
        "about": "Business-friendly models with a permissive licence, good at tools, retrieval and following instructions.",
        "uses": ["tools", "RAG", "chat"],
        "variants": ["ibm-granite/granite-4.2-3b", "ibm-granite/granite-4.2-8b", "ibm-granite/granite-4.2-30b"],
    },
    {
        "id": "small",
        "name": "Small and fast",
        "publisher": "Various",
        "about": "Tiny models for quick experiments, older computers and your first fine-tune: they train in minutes.",
        "uses": ["fine-tuning", "edge"],
        "variants": ["HuggingFaceTB/SmolLM3-3B", "LiquidAI/LFM2.5-1.2B-Instruct", "allenai/Olmo-3-7B-Instruct"],
    },
]

# Settings the publisher gives in the model card rather than generation_config.json, quoted from it.
# Keyed by repo-id prefix; "thinking" applies to models that can think, "instruct" to the rest.
CARD_SAMPLING: dict[str, dict] = {
    "Qwen/Qwen3.5-": {
        "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
        "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    },
    "Qwen/Qwen3.8-": {
        "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
        "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    },
    "mistralai/Ministral-3-": {"any": {"temperature": 0.05}},  # "a temperature below 0.1"
    "mistralai/Mistral-Small-3.2-": {"any": {"temperature": 0.15}},
    "openai/gpt-oss-": {"any": {"temperature": 1.0, "top_p": 1.0}},  # OpenAI's gpt-oss README
    "zai-org/GLM-4.7-": {"any": {"temperature": 1.0, "top_p": 0.95}},
    "deepseek-ai/DeepSeek-R1-": {"any": {"temperature": 0.6, "top_p": 0.95}},
    "microsoft/Phi-4-mini-reasoning": {"any": {"temperature": 0.8, "top_p": 0.95}},
    "microsoft/Phi-4-reasoning": {"any": {"temperature": 0.8, "top_k": 50, "top_p": 0.95}},
    "LiquidAI/LFM2.5-": {"any": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05}},
    "HuggingFaceTB/SmolLM3-": {"any": {"temperature": 0.6, "top_p": 0.95}},
    "allenai/Olmo-3-": {"any": {"temperature": 0.6, "top_p": 0.95}},
    "ibm-granite/granite-4.2-": {"any": {"temperature": 1.0, "top_p": 0.95}},
}

api = HfApi()


def _json(repo: str, name: str) -> dict:
    try:
        return json.loads(Path(hf_hub_download(repo, name)).read_text())
    except (HfHubHTTPError, OSError, ValueError):
        return {}


def _sampling(gen: dict) -> dict:
    s = {k: gen[k] for k in SAMPLING_KEYS if isinstance(gen.get(k), (int, float))}
    # Publishers leave top_p = 1 / top_k = 0 to mean "off"; an explicit temperature alone is still a recommendation.
    return s if "temperature" in s else {}


def _template(repo: str) -> str:
    if "chat_template.jinja" in set(api.list_repo_files(repo)):
        return Path(hf_hub_download(repo, "chat_template.jinja")).read_text()
    template = _json(repo, "tokenizer_config.json").get("chat_template") or ""
    return template if isinstance(template, str) else next((t["template"] for t in template if t.get("name") == "default"), "")


def _thinking(repo: str) -> str | None:
    """"default" when the model thinks unless told not to, "optional" when it must be asked to.

    Renders the chat template's generation prompt with enable_thinking unset, true and false and sees
    which the unset one matches, so it doesn't depend on how a template happens to word the check.
    """
    source = _template(repo)
    if "enable_thinking" not in source:
        return None

    def fail(message):
        raise ValueError(message)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals.update(raise_exception=fail, strftime_now=lambda f: "2026-01-01")
    # transformers' {% generation %} only marks assistant text for training masks.
    source = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", source)

    def render(**flag) -> str:
        return env.from_string(source).render(messages=[{"role": "user", "content": "Hi"}], add_generation_prompt=True, bos_token="", eos_token="", **flag)

    try:
        unset, on, off = render(), render(enable_thinking=True), render(enable_thinking=False)
    except Exception as e:  # noqa: BLE001  (a template we can't render tells us nothing)
        print(f"  {repo}: couldn't render the chat template ({e})", file=sys.stderr)
        return None
    if on == off:
        return None
    return "default" if unset == on else "optional"


def _card_sampling(repo: str, thinking: str | None) -> dict:
    for prefix, modes in CARD_SAMPLING.items():
        if repo.startswith(prefix):
            # llama-server turns thinking on for every template that supports it, so in Temper any
            # model that can think does, and its thinking-mode settings are the ones that apply.
            return modes.get("any") or modes["thinking" if thinking else "instruct"]
    return {}


def _gguf(repo: str) -> str | None:
    org, name = repo.split("/")
    want = {f"{p}{name}-gguf".lower() for p in ("", f"{org}_", "Meta-")}
    found = {m.id.split("/")[0]: m.id for m in api.list_models(search=f"{name}-GGUF", limit=100, sort="downloads") if m.id.split("/")[1].lower() in want}
    return next((found[p] for p in (org, *GGUF_PUBLISHERS) if p in found), None)


def _mlx(repo: str) -> str | None:
    name = repo.split("/")[1].lower()
    ids = {m.id.split("/")[1].lower(): m.id for m in api.list_models(author="mlx-community", search=repo.split("/")[1], limit=50)}
    return next((ids[k] for k in (f"{name}-4bit", f"{name}-mlx-4bit", f"{name}-4bit-dwq", f"{name}-mxfp4-q4") if k in ids), None)


def _label(repo: str, params: float) -> str:
    if m := re.search(r"-(E\d+B)\b", repo):  # Gemma's "effective" sizes
        return m.group(1)
    m = re.search(r"(\d+(?:\.\d+)?[BM](?:-A\d+(?:\.\d+)?B)?)(?![a-z])", repo.split("/")[1], re.I)
    if m:
        return m.group(1).upper().replace("-A", " (A") + (")" if "-A" in m.group(1).upper() else "")
    return f"{params:.0f}B"


def variant(repo: str) -> dict:
    info = api.model_info(repo, expand=["safetensors", "cardData", "gated", "pipeline_tag"])
    gguf, mlx = _gguf(repo), _mlx(repo)
    config = _json(repo, "config.json") or (mlx and _json(mlx, "config.json")) or {}
    text = config.get("text_config") or config
    gen = _json(repo, "generation_config.json") or (mlx and _json(mlx, "generation_config.json")) or {}
    context = text.get("max_position_embeddings")
    if not context and gguf:  # gated repo: read it from the GGUF header instead
        files = [s.rfilename for s in api.model_info(gguf).siblings or [] if s.rfilename.endswith(".gguf") and "mmproj" not in s.rfilename]
        if files:
            context = gguf_shape_remote(gguf, min(files, key=len)).context_max
    params = (info.safetensors.total if info.safetensors else 0) / 1e9
    thinking = _thinking(repo) if not info.gated else None
    return {
        "id": repo,
        "label": _label(repo, params),
        "params": round(params, 2),
        "context": context,
        "license": (info.card_data or {}).get("license"),
        "gated": bool(info.gated),
        "vision": info.pipeline_tag in ("image-text-to-text", "any-to-any"),
        "moe": bool(re.search(r"-A\d+(?:\.\d+)?B", repo) or text.get("num_experts") or text.get("num_local_experts") or text.get("n_routed_experts")),
        "model_type": config.get("model_type") or text.get("model_type"),
        "thinking": thinking,
        "sampling": _sampling(gen) or _card_sampling(repo, thinking),
        "gguf": gguf,
        "mlx": mlx,
    }


def main() -> None:
    repos = [r for f in FAMILIES for r in f["variants"]]
    with ThreadPoolExecutor(8) as pool:
        facts = dict(zip(repos, pool.map(variant, repos)))
    families = []
    for f in FAMILIES:
        variants = [facts[r] for r in f["variants"]]
        for v in variants:
            missing = [k for k in ("context", "license", "gguf", "mlx", "model_type") if not v[k]] + ([] if v["sampling"] else ["sampling"])
            if missing:
                print(f"  {v['id']}: no {', '.join(missing)}", file=sys.stderr)
        families.append({**{k: v for k, v in f.items() if k != "variants"}, "variants": variants})
    OUT.write_text(json.dumps({"families": families}, indent=1) + "\n")
    print(f"{sum(len(f['variants']) for f in families)} models in {len(families)} families → {OUT}")


if __name__ == "__main__":
    main()
