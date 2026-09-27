"""Compression pass ("librarian" step): shrink retrieved passages to an answer.

A cheap/small LLM — qwen3.5:9b via local Ollama (reader consolidated onto the
extractor 2026-09-20, Miguel: keeps two models resident instead of three) — reads the raw passages and extracts a short direct answer plus source
page IDs. Pure summarization: no reasoning, link-following, or relevance judgment.

Transport is plain HTTP to Ollama's /api/chat via httpx (already in the dep tree
through chromadb). Host override: OLLAMA_HOST env var, default localhost:11434.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import httpx

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
        "month, etc.), resolve it to an absolute date using that line's timestamp — "
        "answer with the resolved date, never the relative word itself.\n"
        "Reply with a single JSON object: "
        '{"answer": "<short direct answer>", '
        '"source_page_ids": ["<page_id that supports the answer>", ...]}.\n'
        "If the passages do not contain the answer, reply with an empty answer "
        'and an empty source_page_ids list.\n\n'
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
    _chat: Callable[[str, str], str] | None = None,
) -> dict:
    """Compress passages into {answer, source_page_ids, [source_passage]}.

    ``think`` (task 0085): see :func:`ollama_chat`.
    ``_chat`` is injectable for tests (defaults to a live Ollama call).
    """
    chat = _chat or (lambda p, m: ollama_chat(p, m, num_ctx=num_ctx, think=think))
    page_ids = [p["page_id"] for p in passages]
    result = parse_response(chat(build_prompt(query, passages), model), page_ids)
    if include_passage:
        result["source_passage"] = "\n\n".join(p["text"] for p in passages)
    return result
