"""Task 0099: the shared pluggable-LLM module (CloudLLM + resolve_llm's
provider dispatch). No live network calls — httpx.MockTransport throughout,
same pattern as test_facts.py's OllamaLLM tests."""

from __future__ import annotations

import httpx
import pytest

from cortexlayer._engine import llm
from cortexlayer.errors import CortexConfigError, LLMError


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_cloud_llm_generate_posts_openai_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["auth"] = request.headers.get("authorization")
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"answer": "ok"}'}}]}
        )

    result = llm.CloudLLM(
        "https://api.example.com/chat/completions",
        "some-model",
        "sk-test",
        extra_body={"reasoning_effort": "minimal"},
        http_client=_mock(handler),
    ).generate("SYS", "USER")

    assert result == '{"answer": "ok"}'
    assert seen["auth"] == "Bearer sk-test"
    assert seen["json"]["model"] == "some-model"
    assert seen["json"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert seen["json"]["reasoning_effort"] == "minimal"
    assert seen["json"]["response_format"] == {"type": "json_object"}


def test_cloud_llm_omits_empty_system_message():
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        assert body["messages"] == [{"role": "user", "content": "USER"}]
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    llm.CloudLLM(
        "https://api.example.com/chat/completions", "m", "k", http_client=_mock(handler)
    ).generate("", "USER")


def test_cloud_llm_needs_api_key():
    with pytest.raises(CortexConfigError):
        llm.CloudLLM("https://api.example.com", "m", "")


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: httpx.Response(401, text="unauthorized"),
        lambda r: httpx.Response(200, json={"choices": []}),
        lambda r: httpx.Response(200, text="not json"),
    ],
)
def test_cloud_llm_wraps_failures(handler):
    with pytest.raises(LLMError):
        llm.CloudLLM(
            "https://api.example.com", "m", "k", http_client=_mock(handler)
        ).generate("s", "u")


def test_resolve_llm_ollama_default_and_dict():
    assert isinstance(llm.resolve_llm(None), llm.OllamaLLM)
    resolved = llm.resolve_llm({"model": "x", "host": "http://h"})
    assert isinstance(resolved, llm.OllamaLLM)
    assert resolved.model == "x"


def test_resolve_llm_cloud_provider_preset(monkeypatch):
    monkeypatch.setenv("MUSE_API_KEY", "sk-muse")
    resolved = llm.resolve_llm({"provider": "muse"})
    assert isinstance(resolved, llm.CloudLLM)
    assert resolved.model == "muse-spark-1.3-contributor"
    assert resolved.base_url == llm.CLOUD_PROVIDERS["muse"]["base_url"]
    assert resolved._api_key == "sk-muse"


def test_resolve_llm_cloud_provider_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    resolved = llm.resolve_llm({"provider": "deepseek", "model": "deepseek-v4-pro"})
    assert resolved.model == "deepseek-v4-pro"


def test_resolve_llm_openai_preset(monkeypatch):
    """Verified live 2026-09-27: gpt-5-nano rejects temperature=0 (only default
    1 is supported) and always reasons unless reasoning_effort is capped —
    both baked into the preset default, not left for callers to discover."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    resolved = llm.resolve_llm({"provider": "openai"})
    assert isinstance(resolved, llm.CloudLLM)
    assert resolved.model == "gpt-5-nano"
    assert resolved.base_url == "https://api.openai.com/v1/chat/completions"
    assert resolved._temperature == 1
    assert resolved._extra_body == {"reasoning_effort": "minimal"}


def test_resolve_llm_openai_preset_defaults_are_overridable(monkeypatch):
    """A caller can still override the preset's own defaults."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    resolved = llm.resolve_llm(
        {"provider": "openai", "temperature": 0.7, "extra_body": {"reasoning_effort": "high"}}
    )
    assert resolved._temperature == 0.7
    assert resolved._extra_body == {"reasoning_effort": "high"}


def test_resolve_llm_cloud_provider_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("MUSE_API_KEY", "sk-env")
    resolved = llm.resolve_llm({"provider": "muse", "api_key": "sk-explicit"})
    assert resolved._api_key == "sk-explicit"


def test_resolve_llm_cloud_provider_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    with pytest.raises(CortexConfigError, match="MUSE_API_KEY"):
        llm.resolve_llm({"provider": "muse"})


def test_resolve_llm_generic_cloud_spec():
    resolved = llm.resolve_llm(
        {"provider": "cloud", "base_url": "https://x/y", "model": "m", "api_key": "k"}
    )
    assert isinstance(resolved, llm.CloudLLM)
    assert resolved.base_url == "https://x/y"


def test_resolve_llm_generic_cloud_missing_fields():
    with pytest.raises(CortexConfigError):
        llm.resolve_llm({"provider": "cloud", "model": "m"})


def test_resolve_llm_unknown_provider():
    with pytest.raises(CortexConfigError, match="unknown llm provider"):
        llm.resolve_llm({"provider": "not-a-thing"})


def test_resolve_llm_callable_and_object():
    fn = lambda s, u: "r"  # noqa: E731
    assert llm.resolve_llm(fn).generate("s", "u") == "r"

    class Obj:
        def generate(self, system, user):
            return "obj"

    obj = Obj()
    assert llm.resolve_llm(obj) is obj
