"""Retrieval pipeline: vector search + link-expansion.

Searches the whole collection (no vault/cluster routing — deliberately excluded).
Links only *add* candidates, never restrict. Expansion heuristic: pull the top
``top_n`` linked page(s) per seed, ranked by embedding similarity to the query
(task 0077 — a seed's links are stored sorted by page id, a random uuid4 hex,
so picking ``links[0]`` picked an arbitrary neighbor, not the most relevant one).
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from . import storage

DEFAULT_K = 4


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
