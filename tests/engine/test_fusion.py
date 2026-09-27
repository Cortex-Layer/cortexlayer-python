"""Unit tests for task 0093's rank-fusion primitives (no Chroma needed here —
retrieve_fused's integration with storage/links is covered in test_retrieval.py)."""

from datetime import datetime, timedelta, timezone

from cortexlayer._engine import fusion


# --- has_temporal_cue ---------------------------------------------------


def test_temporal_cue_words_detected():
    assert fusion.has_temporal_cue("When did they get married?")
    assert fusion.has_temporal_cue("What happened last week?")
    assert fusion.has_temporal_cue("How old is her dog?")


def test_bare_year_counts_as_temporal_cue():
    assert fusion.has_temporal_cue("What did she do in 2019?")


def test_non_temporal_query_has_no_cue():
    assert not fusion.has_temporal_cue("What is her dog's name?")
    assert not fusion.has_temporal_cue("Where does he work?")


# --- reciprocal_rank_fusion ----------------------------------------------


def test_rrf_agreement_beats_single_voice_top_pick():
    """A doc ranked #1 by both voices should outscore one ranked #1 by only
    one voice and #3 by the other (agreement beats a single strong opinion)."""
    voice_a = ["x", "y", "z"]
    voice_b = ["y", "z", "x"]
    scores = fusion.reciprocal_rank_fusion([voice_a, voice_b])
    assert scores["y"] > scores["x"]


def test_rrf_ignores_voices_that_dont_rank_a_candidate():
    scores = fusion.reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(scores) == {"a", "b", "c"}
    assert scores["a"] > scores["b"]  # a is rank 1 in its voice, b is rank 2


def test_rrf_empty_voices_yield_no_scores():
    assert fusion.reciprocal_rank_fusion([]) == {}
    assert fusion.reciprocal_rank_fusion([[], []]) == {}


# --- recency_rank ---------------------------------------------------------


def _iso(days_ago: float, now: datetime) -> str:
    return (now - timedelta(days=days_ago)).isoformat()


def test_recency_rank_newest_first():
    now = datetime.now(timezone.utc)
    passages = [
        {"page_id": "old", "created_at": _iso(100, now)},
        {"page_id": "new", "created_at": _iso(1, now)},
        {"page_id": "mid", "created_at": _iso(10, now)},
    ]
    assert fusion.recency_rank(passages, now=now) == ["new", "mid", "old"]


def test_recency_rank_missing_timestamp_sorts_last():
    now = datetime.now(timezone.utc)
    passages = [
        {"page_id": "dateless", "created_at": ""},
        {"page_id": "dated", "created_at": _iso(5, now)},
    ]
    assert fusion.recency_rank(passages, now=now) == ["dated", "dateless"]


def test_recency_rank_unparseable_timestamp_does_not_raise():
    now = datetime.now(timezone.utc)
    passages = [
        {"page_id": "garbage", "created_at": "not-a-date"},
        {"page_id": "dated", "created_at": _iso(1, now)},
    ]
    assert fusion.recency_rank(passages, now=now) == ["dated", "garbage"]


# --- mmr_rerank ------------------------------------------------------------


def test_mmr_prefers_diversity_over_a_near_duplicate():
    """Two near-identical high-score passages, one distinct mid-score one, and
    a low-score distractor (so min-max normalisation doesn't zero out the
    distinct passage's relevance by making it the pool minimum): MMR should
    pick the distinct one over the second near-duplicate."""
    pool = [
        {"page_id": "a", "text": "the cat sat on the mat", "score": 1.0},
        {"page_id": "b", "text": "the cat sat on the mat today", "score": 0.95},
        {"page_id": "c", "text": "completely different content here", "score": 0.6},
        {"page_id": "d", "text": "filler filler filler filler", "score": 0.1},
    ]
    selected = fusion.mmr_rerank(pool, limit=2, lam=0.5)
    ids = [p["page_id"] for p in selected]
    assert ids[0] == "a"          # top relevance always wins the first pick
    assert "c" in ids             # diversity beats picking the near-duplicate "b"


def test_mmr_lambda_1_ignores_diversity():
    """lam=1.0 degenerates to plain top-score selection (no diversity term)."""
    pool = [
        {"page_id": "a", "text": "same same same", "score": 1.0},
        {"page_id": "b", "text": "same same same", "score": 0.9},
        {"page_id": "c", "text": "totally unrelated", "score": 0.5},
    ]
    selected = fusion.mmr_rerank(pool, limit=2, lam=1.0)
    assert [p["page_id"] for p in selected] == ["a", "b"]


def test_mmr_respects_limit_and_empty_pool():
    pool = [{"page_id": "a", "text": "x", "score": 1.0}]
    assert fusion.mmr_rerank(pool, limit=0) == []
    assert fusion.mmr_rerank([], limit=5) == []
    assert len(fusion.mmr_rerank(pool, limit=5)) == 1


def test_mmr_handles_equal_scores_without_dividing_by_zero():
    pool = [
        {"page_id": "a", "text": "one", "score": 0.5},
        {"page_id": "b", "text": "two", "score": 0.5},
    ]
    selected = fusion.mmr_rerank(pool, limit=2)
    assert {p["page_id"] for p in selected} == {"a", "b"}
