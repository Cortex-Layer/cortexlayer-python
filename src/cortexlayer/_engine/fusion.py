"""Rank fusion for retrieval: Reciprocal Rank Fusion (RRF) + MMR diversity re-rank.

Task 0093. Competitor research (Mnemosyne) surfaced the pattern — fuse heterogeneous
signals by RANK POSITION instead of blending scores on a common numeric scale, then
drop near-duplicates with a diversity re-rank — but that research was an automated
fetch-and-summarize, not a byte-level read of their source, so this is our OWN
implementation of the general techniques (Cormack et al. 2009 for RRF; Carbonell &
Goldstein 1998 for MMR), not a port. Constants below (``DEFAULT_RRF_K``,
``DEFAULT_MMR_LAMBDA``) are the commonly-cited textbook defaults, tuned against our
own data in the task 0093 paired eval rather than copied from Mnemosyne's file.

Three "voices" retrieval.retrieve_fused combines:
  - seed rank (this backend's own seed scoring: plain vector distance for the raw
    backend, or 0079's fused semantic+keyword+entity score for facts);
  - link rank (0077's query-relevance ranking of a seed's linked neighbors, widened
    here to the FULL neighbor pool instead of a per-seed top-1 pick);
  - recency rank (exponential decay over ``created_at``) — only included when the
    query itself has a temporal cue word, so non-temporal queries are never biased
    toward newer pages.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

DEFAULT_RRF_K = 60
DEFAULT_MMR_LAMBDA = 0.7
RECENCY_HALF_LIFE_DAYS = 30.0

_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Deliberately small and dependency-free (no date-parsing library) — a query
# containing any of these, or a bare four-digit year, is treated as temporal.
# False positives (e.g. "day" in "day trip") just mean the recency voice joins
# the fusion for a query it doesn't help; false negatives just mean it sits
# out for one it might have helped. Neither corrupts the other two voices.
_TEMPORAL_CUES = frozenset(
    "when before after first last next previously earlier later recent recently "
    "yesterday today tomorrow tonight ago since until during then now currently "
    "date year years month months week weeks day days old age born birthday "
    "anniversary schedule scheduled upcoming past monday tuesday wednesday "
    "thursday friday saturday sunday january february march april may june july "
    "august september october november december".split()
)


def _words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def has_temporal_cue(query: str) -> bool:
    """Whether ``query`` looks like it's asking about time/order/recency —
    gates the recency voice so non-temporal queries never get a recency bias."""
    if _YEAR_RE.search(query):
        return True
    return bool(_words(query) & _TEMPORAL_CUES)


def reciprocal_rank_fusion(
    voices: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K
) -> Dict[str, float]:
    """``score(d) = sum over voices ranking d of 1 / (k + rank)`` (rank 1-based).

    A voice is a list of ids, best first; an id absent from a voice simply
    contributes nothing from it (no need for every voice to rank every
    candidate). ``k=60`` is RRF's standard constant (Cormack et al. 2009) —
    large enough that a single voice's rank-1 pick doesn't dominate a
    candidate two other voices agree is better."""
    scores: Dict[str, float] = {}
    for voice in voices:
        for rank, doc_id in enumerate(voice, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _age_days(created_at: str, now: datetime) -> Optional[float]:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() / 86400.0


def recency_rank(
    passages: Sequence[dict],
    now: Optional[datetime] = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> List[str]:
    """Passage ids ranked newest-first by exponential decay over
    ``created_at`` (``0.5 ** (age_days / half_life_days)``). A missing or
    unparseable timestamp sorts last — never dropped, just never wins this
    voice. (In the LOCOMO benchmark harness every page's ``created_at`` is
    the real insertion instant, not the conversation's session date — but
    pages are inserted session-by-session in chronological order, so
    insertion order is a faithful proxy for conversation recency there, same
    as it is a real production signal for :meth:`Memory.add` call order.)"""
    now = now or datetime.now(timezone.utc)

    def _score(p: dict) -> float:
        age = _age_days(p.get("created_at", ""), now)
        return -1.0 if age is None else 0.5 ** (max(age, 0.0) / half_life_days)

    return [p["page_id"] for p in sorted(passages, key=_score, reverse=True)]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def mmr_rerank(
    pool: Sequence[dict], limit: int, lam: float = DEFAULT_MMR_LAMBDA
) -> List[dict]:
    """Greedy diversity re-rank over an already-fused ``pool`` (each dict
    needs ``page_id``, ``text``, ``score``), picking up to ``limit``:
    ``argmax(lam * relevance - (1 - lam) * max_similarity_to_already_picked)``
    each step, similarity = word-Jaccard (no extra embedding calls).

    ``relevance`` is ``score`` min-max normalised to ``[0, 1]`` first —
    raw RRF scores are all clustered near ``1/k`` (~0.016 for `k=60`), so
    without normalising, the Jaccard term (0..1) would swamp ``relevance``
    for any ``lam`` short of ~1.0 and the pass would just be similarity
    minimisation, not a relevance/diversity tradeoff.
    """
    if limit <= 0 or not pool:
        return []
    scores = [p["score"] for p in pool]
    lo, hi = min(scores), max(scores)
    span = hi - lo or 1.0
    relevance = {p["page_id"]: (p["score"] - lo) / span for p in pool}
    words = {p["page_id"]: _words(p["text"]) for p in pool}

    remaining = list(pool)
    selected: List[dict] = []
    while remaining and len(selected) < limit:
        if not selected:
            best = max(remaining, key=lambda p: relevance[p["page_id"]])
        else:
            def _value(p: dict) -> float:
                rel = relevance[p["page_id"]]
                sim = max(_jaccard(words[p["page_id"]], words[s["page_id"]]) for s in selected)
                return lam * rel - (1 - lam) * sim

            best = max(remaining, key=_value)
        selected.append(best)
        remaining.remove(best)
    return selected
