"""Ported verbatim from the Cortex backend suite (test_compression.py) — proves the copied engine
behaves identically. Only the imports changed."""

import httpx
import pytest

from cortexlayer._engine import compression


def _stub_chat(content: str):
    def chat(prompt: str, model: str) -> str:
        chat.last_prompt = prompt
        chat.last_model = model
        return content

    return chat


PASSAGES = [
    {"page_id": "aaa", "text": "The director of Inception is Christopher Nolan."},
    {"page_id": "bbb", "text": "Christopher Nolan was born on July 30, 1970."},
]


def test_compress_parses_answer_and_ids():
    chat = _stub_chat('{"answer": "1970", "source_page_ids": ["bbb"]}')
    result = compression.compress("When was Nolan born?", PASSAGES, _chat=chat)
    assert result == {"answer": "1970", "source_page_ids": ["bbb"]}
    assert "bbb" in chat.last_prompt and "1970" in chat.last_prompt


def test_include_passage_adds_raw_text():
    chat = _stub_chat('{"answer": "1970", "source_page_ids": ["bbb"]}')
    result = compression.compress(
        "When was Nolan born?", PASSAGES, include_passage=True, _chat=chat
    )
    assert "1970" in result["source_passage"]
    assert "Inception" in result["source_passage"]


def test_garbage_response_falls_back_gracefully():
    chat = _stub_chat("not json at all")
    result = compression.compress("When was Nolan born?", PASSAGES, _chat=chat)
    assert result["answer"] == "not json at all"
    assert result["source_page_ids"] == ["aaa", "bbb"]


def test_build_prompt_includes_anti_hedge_instruction():
    """Task 0102: no "not specified" / "cannot be determined" style non-answers
    when the passages contain any relevant information."""
    prompt = compression.build_prompt("When was Nolan born?", PASSAGES)
    assert "NEVER hedge" in prompt
    assert "not specified" in prompt
    assert "cannot be determined" in prompt
    assert "Commit to the single best-supported answer" in prompt


def test_build_prompt_includes_causal_reasoning_template():
    """Task 0102: mem0-inspired counterfactual/causal template for
    judgment-style questions ("would X still do Y if Z hadn't happened")."""
    prompt = compression.build_prompt("When was Nolan born?", PASSAGES)
    assert "reason step by step" in prompt
    assert "likely yes" in prompt
    assert "likely no" in prompt
    assert "because of" in prompt


def test_llm_spec_routes_through_resolve_llm():
    """Task 0099: llm= bypasses ollama_chat entirely, going through
    _engine.llm.resolve_llm instead — proven with a fake generate()."""
    calls = []

    class FakeLLM:
        def generate(self, system: str, user: str) -> str:
            calls.append((system, user))
            return '{"answer": "1970", "source_page_ids": ["bbb"]}'

    result = compression.compress("When was Nolan born?", PASSAGES, llm=FakeLLM())
    assert result == {"answer": "1970", "source_page_ids": ["bbb"]}
    assert len(calls) == 1
    system, user = calls[0]
    assert system == ""
    assert "bbb" in user and "Nolan" in user


def test_chat_takes_precedence_over_llm():
    """_chat (test injection) wins even when llm= is also passed."""
    chat = _stub_chat('{"answer": "chat wins", "source_page_ids": []}')

    class ExplodingLLM:
        def generate(self, system: str, user: str) -> str:
            raise AssertionError("llm.generate should not be called when _chat is set")

    result = compression.compress(
        "When was Nolan born?", PASSAGES, llm=ExplodingLLM(), _chat=chat
    )
    assert result["answer"] == "chat wins"


def test_live_ollama_call():
    """End-to-end against local Ollama. Skipped when the server is down."""
    try:
        result = compression.compress("When was Nolan born?", PASSAGES)
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout):
        pytest.skip("Ollama server not reachable")
    assert result["answer"]
    assert result["source_page_ids"]
