"""Retrieval pipeline: vector search + link-expansion.

Searches the whole collection (no vault/cluster routing — deliberately excluded).
Links only *add* candidates, never restrict. Expansion heuristic: pull the top
``top_n`` linked page(s) per seed, ranked by embedding similarity to the query
(task 0077 — a seed's links are stored sorted by page id, a random uuid4 hex,
so picking ``links[0]`` picked an arbitrary neighbor, not the most relevant one).
"""

from __future__ import annotations

from typing import Optional

from chromadb.api.models.Collection import Collection

from . import fusion, storage

DEFAULT_K = 50
# Raised from 4 -> 50 (task 0101): judged LOCOMO open-domain subset showed
# +0.231 Muse-judge accuracy / +0.095 F1 (0.385/0.189 -> 0.615/0.284, n=13,
# same ingest) from budget alone, no ranking/architecture change. Still a
# single-conversation subset, not the full suite — see task 0101 notes.


def retrieve(collection: Collection, query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return combined seed + link-expanded passages.

    Each passage is ``{page_id, text, score, via, [linked_from]}`` — ``score``
    is always a real Chroma distance (lower = closer), for both the seed
    (``via="direct"``, distance to the query) and its expansions
    (``via="link"``, distance from the neighbor-ranking query, with the seed
    id in ``linked_from``). Seeds come first, then expanded pages,
    deduplicated by page ID (a page that is both seed and expansion keeps
    ``direct``).
    """
    return expand_links(collection, storage.query(collection, query, n_results=k), query=query)


def _rank_neighbors(
    collection: Collection,
    neighbor_ids: list[str],
    top_n: int,
    query: str | None = None,
    query_embedding: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Neighbor ids ranked by relevance to the query (nearest first), capped at
    ``top_n``, paired with the Chroma distance from that same ranking query
    (lower is closer). Filters out dangling links (deleted pages) first —
    Chroma's ``query(ids=...)`` errors if any id in the filter doesn't exist.
    Falls back to stored (arbitrary, id-sorted) order with a placeholder
    ``0.0`` score if neither ``query`` nor ``query_embedding`` is given, since
    that order isn't a real similarity ranking.

    ``query_embedding`` (not raw text) is required for collections whose
    vectors come from a caller-supplied embedder (fact memory) rather than
    Chroma's own default text embedding function — passing ``query_texts``
    there would re-embed with the wrong (default) function and error on the
    dimension mismatch, or silently rank on the wrong vector space.
    """
    existing = collection.get(ids=neighbor_ids)["ids"]
    if not existing:
        return []
    n_results = min(top_n, len(existing))
    if query_embedding is not None:
        result = collection.query(
            query_embeddings=[query_embedding], ids=existing, n_results=n_results
        )
    elif query:
        result = collection.query(
            query_texts=[query], ids=existing, n_results=n_results
        )
    else:
        return [(pid, 0.0) for pid in existing[:top_n]]
    ranked = result["ids"][0] if result["ids"] else []
    if not ranked:
        return [(pid, 0.0) for pid in existing[:top_n]]
    # Bug fix (found 2026-09-26 testing the Playground: link-expansion rows
    # always showed score=0): this query's own distances are the real
    # per-neighbor scores. The old code discarded them here and re-derived
    # a "score" from a later storage.get_page() call, which never carries
    # distances (it's a plain Chroma `get`, not a `query`) — so it silently
    # fell back to 0.0 every time.
    distances = result.get("distances", [[]])[0] if result.get("distances") else []
    return list(zip(ranked, distances)) if distances else [(pid, 0.0) for pid in ranked]


def expand_links(
    collection: Collection,
    seeds: list[dict],
    query: str | None = None,
    top_n: int = 1,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Seeds (storage-shaped page dicts with ``score``) -> passages, adding
    each seed's ``top_n`` most-query-relevant linked page(s). Split from
    :func:`retrieve` so other seed sources (fact memory's entity-boosted
    search) reuse the same expansion. ``query`` (or ``query_embedding``, for
    collections with a non-default embedder) ranks the neighbors; omit both
    to keep the old arbitrary (id-sorted) pick."""
    seen: set[str] = set()
    passages: list[dict] = []

    def _add(page_id: str, text: str, score: float, via: str,
             linked_from: str | None = None, tags: dict | None = None) -> None:
        if page_id not in seen:
            seen.add(page_id)
            passage: dict = {"page_id": page_id, "text": text,
                             "score": score, "via": via}
            if linked_from is not None:
                passage["linked_from"] = linked_from
            if tags:
                passage["tags"] = tags
            passages.append(passage)

    for seed in seeds:
        _add(seed["id"], seed["text"], seed.get("score", 0.0), "direct", tags=seed.get("tags"))
    for seed in seeds:
        if not seed["links"]:
            continue
        targets = _rank_neighbors(
            collection, seed["links"], top_n, query=query, query_embedding=query_embedding
        )
        for target_id, score in targets:
            target = storage.get_page(collection, target_id)
            if target is not None:
                _add(target["id"], target["text"], score,
                     "link", linked_from=seed["id"], tags=target.get("tags"))
    return passages


def retrieve_fused_multi(
    pools: list[tuple[Collection, Optional[str], list[dict]]],
    query: str,
    top_n: int = 1,
    query_embedding: list[float] | None = None,
    rrf_k: int = fusion.DEFAULT_RRF_K,
    use_mmr: bool = True,
    mmr_lambda: float = fusion.DEFAULT_MMR_LAMBDA,
) -> list[dict]:
    """Task 0109 / arch §9.4: :func:`retrieve_fused` generalized to fan out
    across several Chroma collections at once — exactly two in production
    (the caller's personal pool + one active org pool, never every org the
    caller belongs to), though this function itself places no limit on
    ``pools`` and doesn't enforce that business rule — the caller does.

    ``pools`` is ``[(collection, pool_label, seeds), ...]`` — ``seeds`` are
    already fetched per-collection by the caller (this backend's own seed
    scoring: vector distance for raw, 0079's fused score for facts), exactly
    like :func:`retrieve_fused`'s ``seeds`` parameter, just once per pool.
    Link-expansion and page hydration both run against each candidate's OWN
    originating collection (tracked through the merge via closures over that
    pool's ``collection`` — a candidate id is never looked up in a different
    pool's collection than the one it was found in). Each pool contributes
    its own seed-rank voice and link-rank voice to one shared RRF pass
    (fusing by rank position *within* each pool's own ranking, never a
    cross-pool score comparison — pools can use different scoring scales),
    plus the usual single recency voice computed once over the combined
    candidate pool when ``query`` has a temporal cue, then one MMR re-rank
    over everything.

    ``pool_label`` is stamped onto each resulting passage's ``tags["pool"]``
    (0090 provenance) so a merged result still shows which pool it came
    from — e.g. ``"personal"`` / ``"org:<org_id>"``. ``None`` omits the key
    entirely, which is what makes a single ``(collection, None, seeds)``
    pool reduce to :func:`retrieve_fused` bit-for-bit.
    """
    by_id: dict[str, dict] = {}
    voices: list[list[str]] = []
    total_seeds = 0

    for collection, pool_label, seeds in pools:
        seed_ids = [s["id"] for s in seeds]
        total_seeds += len(seed_ids)
        for s in seeds:
            tags = dict(s.get("tags") or {})
            if pool_label is not None:
                tags["pool"] = pool_label
            by_id[s["id"]] = {
                "page_id": s["id"], "text": s["text"], "via": "direct",
                "created_at": s.get("created_at", ""), "tags": tags or None,
            }
        voices.append(seed_ids)

        neighbor_ids = sorted({nid for s in seeds for nid in s.get("links", [])} - set(seed_ids))
        link_from: dict[str, str] = {}
        for s in seeds:
            for nid in s.get("links", []):
                if nid in neighbor_ids:
                    link_from.setdefault(nid, s["id"])

        link_ranked = (
            _rank_neighbors(collection, neighbor_ids, len(neighbor_ids),
                             query=query, query_embedding=query_embedding)
            if neighbor_ids else []
        )
        link_ids = [nid for nid, _ in link_ranked]
        for nid in link_ids:
            target = storage.get_page(collection, nid)
            if target is not None:
                tags = dict(target.get("tags") or {})
                if pool_label is not None:
                    tags["pool"] = pool_label
                by_id[nid] = {
                    "page_id": nid, "text": target["text"], "via": "link",
                    "linked_from": link_from.get(nid),
                    "created_at": target.get("created_at", ""), "tags": tags or None,
                }
        voices.append(link_ids)

    if not by_id:
        return []

    if fusion.has_temporal_cue(query):
        voices.append(fusion.recency_rank(list(by_id.values())))
    scores = fusion.reciprocal_rank_fusion(voices, k=rrf_k)

    pool = [{**passage, "score": scores.get(pid, 0.0)} for pid, passage in by_id.items()]
    pool.sort(key=lambda p: p["score"], reverse=True)

    target_count = min(total_seeds + top_n * total_seeds, len(pool))
    if use_mmr:
        selected = fusion.mmr_rerank(pool, target_count, lam=mmr_lambda)
    else:
        selected = pool[:target_count]
    for p in selected:
        p.pop("created_at", None)  # internal to fusion, not part of the public passage shape
        if p.get("linked_from") is None:
            p.pop("linked_from", None)
        if not p.get("tags"):
            p.pop("tags", None)
    return selected


def retrieve_fused(
    collection: Collection,
    seeds: list[dict],
    query: str,
    top_n: int = 1,
    query_embedding: list[float] | None = None,
    rrf_k: int = fusion.DEFAULT_RRF_K,
    use_mmr: bool = True,
    mmr_lambda: float = fusion.DEFAULT_MMR_LAMBDA,
) -> list[dict]:
    """Task 0093: seeds + link-expansion, unified by Reciprocal Rank Fusion
    instead of :func:`expand_links`'s two-stage "rank seeds, then separately
    rank+append top-``top_n`` links per seed" pipeline, plus a post-fusion MMR
    diversity re-rank. Same passage shape as :func:`expand_links`
    (``page_id``/``text``/``score``/``via``/``linked_from``/``tags``); ``score``
    here is the fused RRF score (higher = better), not a Chroma distance.

    Voices fused (see ``fusion`` module):
      - **seed** — the order ``seeds`` already arrives in (this backend's own
        scoring: vector distance for raw, 0079's fused score for facts);
      - **link** — ALL of the seeds' linked neighbors (not just each seed's
        top-``top_n``), ranked together by relevance to ``query`` in one pass,
        so fusion can pick a strong neighbor of seed 3 over a weak one of
        seed 1 — the per-seed cap in :func:`expand_links` can't do that;
      - **recency** — only when ``fusion.has_temporal_cue(query)`` is true.

    The final passage count matches the upper bound :func:`expand_links` with
    the same ``top_n`` would produce (``len(seeds) + top_n * len(seeds)``,
    capped by however many distinct candidates actually exist) — fusion
    chooses FROM a wider pool, but the OUTPUT is the same size, so a paired
    eval against :func:`expand_links` isolates fusion+MMR from a change in
    retrieval breadth.

    Task 0109: a thin single-pool call into :func:`retrieve_fused_multi`
    (``pool_label=None``, so no ``tags["pool"]`` gets added) — bit-for-bit
    the same selection/order/scores as before that generalization.
    """
    return retrieve_fused_multi(
        [(collection, None, seeds)], query, top_n=top_n,
        query_embedding=query_embedding, rrf_k=rrf_k,
        use_mmr=use_mmr, mmr_lambda=mmr_lambda,
    )
