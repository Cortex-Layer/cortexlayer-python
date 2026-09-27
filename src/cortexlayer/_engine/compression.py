"""Compression pass ("librarian" step): shrink retrieved passages to an answer.

A cheap/small LLM — qwen3.5:9b via local Ollama by default (reader consolidated
onto the extractor 2026-09-20, Miguel: keeps two models resident instead of
three) — reads the raw passages and extracts a short direct answer plus source
page IDs. Pure summarization: no reasoning, link-following, or relevance judgment.

Transport is plain HTTP to Ollama's /api/chat via httpx (already in the dep tree
through chromadb). Host override: OLLAMA_HOST env var, default localhost:11434.

Task 0099: ``compress(..., llm=...)`` can point this step at a cloud provider
instead (same ``_engine.llm.resolve_llm`` spec the facts extractor uses — a
dict like ``{"provider": "muse"}``, a callable, or an object with
``generate()``) — ``llm=None`` (the default) is unchanged local-Ollama
behavior via ``ollama_chat``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import httpx

from .llm import resolve_llm

DEFAULT_MODEL = "qwen3.5:9b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def build_prompt(query: str, passages: list[dict]) -> str:
    numbered = "\n\n".join(
        f"[Passage {i + 1} | page_id={p['page_id']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )
    return (
        "Answer the question using ONLY the passages below. Be concise and direct.\n"
        "Passage lines may start with a timestamp, e.g. \"[3 July, 2023] Name: ...\". If "
        "the answer involves a relative time word (yesterday, today, last week, this "
        "month, etc.) ANYWHERE in the passage text — even inside an already-extracted "
        "fact sentence, not just raw dialogue — you MUST resolve it to an absolute date "
        "using the nearest timestamp in that same passage as the anchor. Your answer "
        "must contain ONLY the resolved absolute date; a relative word in your answer "
        "is always wrong, with no exceptions, even if the passage itself still contains "
        "one. Example: passage \"[8 May, 2023] Caroline moved to Lisbon last week\" and "
        "question \"When did Caroline move to Lisbon?\" -> answer \"around 1 May 2023\", "
        "never \"last week\".\n"
        "NEVER hedge with \"not specified\", \"cannot be determined\", \"unclear\", or any "
        "similar non-answer if the passages contain ANY information relevant to the "
        "question. Commit to the single best-supported answer you can construct from "
        "what is there, even if it takes a small inference — a confident, reasonable "
        "guess beats a refusal to answer. Only reply with an empty answer if the "
        "passages contain NO information relevant to the question at all.\n"
        "Some questions ask you to judge whether something is true, likely, or would "
        "happen, based on a cause described in the passages (e.g. \"would X still do Y "
        "if Z hadn't happened\", \"is X likely to do Y\", \"does X support Y\"). For "
        "these, reason step by step instead of defaulting to a non-answer: (1) find the "
        "passage stating X did/does Y because of, due to, or as a result of Z; (2) check "
        "whether the question's scenario removes or contradicts Z; (3) if Z is removed "
        "or contradicted, answer with the OPPOSITE of what the passage states (usually "
        "\"likely no\"); if Z still holds, answer AGREES with the passage (usually "
        "\"likely yes\"). Give your answer as \"likely yes\" or \"likely no\" plus a "
        "short reason grounded in the passage. Example: passage \"Caroline decided to "
        "skip the marathon because of her knee injury\" and question \"Would Caroline "
        "have run the marathon if she hadn't been injured?\" -> her decision to skip was "
        "caused by the injury; removing the injury removes the cause -> answer \"likely "
        "yes, the injury was the only reason she skipped it\".\n"
        "Reply with a single JSON object: "
        '{"answer": "<short direct answer>", '
        '"source_page_ids": ["<page_id that supports the answer>", ...]}.\n'
        "If the passages contain no information relevant to the question, reply with an "
        'empty answer and an empty source_page_ids list.\n\n'
        f"Question: {query}\n\nPassages:\n{numbered}"
    )


def ollama_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    """Single non-streaming chat call. Returns the raw response content.

    ``num_ctx`` overrides the model's runtime context window (Ollama reloads
    the model if this differs from what's currently resident). Needed for the
    full-context benchmark arm, where a whole conversation transcript can
    exceed the default 32768-token window — see run_locomo.py --backend
    full_context.

    ``think`` (task 0085, falsifiability test in progress): thinking models
    (e.g. qwen3.5) otherwise spend hundreds of hidden reasoning tokens on a
    one-line answer (~11s vs ~0.6s measured) — hence the default off. Under
    investigation as a fix for reader-miss empty answers on questions needing
    one small resolution step (relative-date math, cross-referencing two
    adjacent facts). Exposed as a parameter (not hardcoded) so the benchmark
    harness can A/B it before any default changes.
    """
    options = {"temperature": 0}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    with httpx.Client(base_url=OLLAMA_HOST, timeout=120.0) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "think": think,
                "options": options,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def parse_response(content: str, fallback_ids: list[str]) -> dict:
    """Parse the model's JSON; fall back to raw text + all IDs on failure."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        parsed = json.loads(content[start:end])
        answer = str(parsed.get("answer", "")).strip()
        ids = [str(i) for i in parsed.get("source_page_ids", []) or []]
        return {"answer": answer, "source_page_ids": ids or fallback_ids}
    except (ValueError, AttributeError, TypeError):
        return {"answer": content.strip(), "source_page_ids": fallback_ids}


def compress(
    query: str,
    passages: list[dict],
    model: str = DEFAULT_MODEL,
    include_passage: bool = False,
    num_ctx: int | None = None,
    think: bool = False,
    llm: Any = None,
    _chat: Callable[[str, str], str] | None = None,
) -> dict:
    """Compress passages into {answer, source_page_ids, [source_passage]}.

    ``think`` (task 0085): see :func:`ollama_chat`.
    ``llm`` (task 0099): an ``_engine.llm.resolve_llm`` spec — ``None``
    (default) keeps today's local-Ollama ``ollama_chat`` path (honoring
    ``model``/``num_ctx``/``think``); anything else (a provider dict like
    ``{"provider": "muse"}``, a callable, or an object with ``generate()``)
    routes through that provider instead, ignoring ``model``/``num_ctx``.
    ``_chat`` is injectable for tests and takes precedence over ``llm``
    (defaults to a live Ollama call).
    """
    if _chat is not None:
        chat = _chat
    elif llm is not None:
        provider = resolve_llm(llm)
        chat = lambda p, m: provider.generate("", p)  # noqa: E731
    else:
        chat = lambda p, m: ollama_chat(p, m, num_ctx=num_ctx, think=think)  # noqa: E731
    page_ids = [p["page_id"] for p in passages]
    result = parse_response(chat(build_prompt(query, passages), model), page_ids)
    if include_passage:
        result["source_passage"] = "\n\n".join(p["text"] for p in passages)
    return result
