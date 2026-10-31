import json

import pytest
from fastapi.testclient import TestClient

from temper_engine import chat_render, engines, training
from temper_engine.app import create_app

AUTH = {"Authorization": "Bearer secret"}
CONV = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]


class Tokenizer:
    """Renders with jinja2 as transformers does, with whatever chat_template it's given."""

    chat_template = None
    bos_token, eos_token = "<s>", "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        return env.from_string(self.chat_template).render(messages=messages, add_generation_prompt=add_generation_prompt, bos_token=self.bos_token, eos_token=self.eos_token)


def test_every_named_template_renders_the_reply_after_its_prompt():
    for name in chat_render.TEMPLATES:
        tok = Tokenizer()
        chat_render.use(tok, chat_render.template(name))
        prompt, completion = chat_render.render(tok, CONV)
        assert "Hi" in prompt and "Be brief." in prompt and "Hello!" not in prompt, name
        assert completion.startswith("Hello!"), name
    tok = Tokenizer()
    chat_render.use(tok, chat_render.template("alpaca"))
    assert chat_render.render(tok, CONV) == ("Be brief.\n\n### Instruction:\nHi\n\n### Response:\n", "Hello!</s>\n\n")


def test_templates_by_name_file_or_text_and_broken_ones_refused(tmp_path):
    assert chat_render.template(None) is None and chat_render.template(" ") is None
    own = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}"
    assert chat_render.template(own) == own
    (tmp_path / "t.jinja").write_text(own)
    assert chat_render.template(str(tmp_path / "t.jinja")) == own
    for bad, why in (("hello", "Choose one of"), ("{{ x", "doesn't work"), ("{% for m in messages %}x{% endfor %}", "leaves out"),
                     (str(tmp_path / "gone.jinja"), "no template file")):
        with pytest.raises(ValueError, match=why):
            chat_render.template(bad)


def test_mlx_wrapper_is_unwrapped_but_a_fast_tokenizers_own_tokenizer_is_not():
    class TokenizerWrapper:
        def __init__(self):
            self._tokenizer = Tokenizer()

    wrapped = TokenizerWrapper()
    chat_render.use(wrapped, "X")
    assert wrapped._tokenizer.chat_template == "X"
    fast = Tokenizer()
    fast._tokenizer = object()  # a Hugging Face fast tokenizer's Rust tokenizer
    chat_render.use(fast, "Y")
    assert fast.chat_template == "Y"


def test_a_run_starts_with_the_template_text_and_serves_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "probe", lambda: {"backend": "metal", "capabilities": {"trainer": "mlx"}})
    started = []
    # The worker never runs, so each job is finished at once (a running one would show in other tests).
    monkeypatch.setattr(training.registry, "run_worker", lambda job, module, cfg: (started.append(json.loads(open(cfg).read())), job.finish("done")))
    with TestClient(create_app(token="secret")) as c:
        assert [t["id"] for t in c.get("/api/train/templates", headers=AUTH).json()] == list(chat_render.TEMPLATES)
        assert c.post("/api/train", headers=AUTH, json={"base_model": "m", "dataset": "x.jsonl", "chat_template": "llama3"}).status_code == 200
        assert c.post("/api/train", headers=AUTH, json={"base_model": "m", "dataset": "x.jsonl", "chat_template": "nonsense"}).status_code == 400
    assert started[0]["chat_template"] == chat_render.TEMPLATES["llama3"]["template"]

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    chat_render.save(adapter, "{{ messages }}")
    monkeypatch.setattr(engines, "python_for", lambda e: "/py")
    settings = {"cache_type": "f16", "context": 4096}
    mlx = engines.command("mlx", "/m", settings, 9, adapter=str(adapter))
    assert mlx[mlx.index("--chat-template") + 1] == "{{ messages }}"
    vllm = engines.command("vllm", "/m", settings, 9, adapter=str(adapter))
    assert vllm[vllm.index("--chat-template") + 1] == str(adapter / "chat_template.jinja")
    assert "--chat-template" not in engines.command("mlx", "/m", settings, 9, adapter=str(tmp_path))
