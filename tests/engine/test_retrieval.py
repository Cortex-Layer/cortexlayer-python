"""Ported verbatim from the Cortex backend suite (test_retrieval.py, mem0 test omitted) — proves the copied engine
behaves identically. Only the imports changed."""

from cortexlayer._engine import ingestion as _ingestion
from cortexlayer._engine import nlp as _nlp


class ingestion:  # noqa: N801 — shim: keeps the ported test bodies byte-identical
    _nlp = _nlp.SpacyNLP()

    @staticmethod
    def ingest_text(collection, text):
        return _ingestion.add_text(collection, text, ingestion._nlp)

from cortexlayer._engine import linking
from cortexlayer._engine import retrieval
from cortexlayer._engine import storage


def _fresh_collection(tmp_path):
    client = storage.get_client(persist_dir=str(tmp_path / "chroma"))
    return storage.get_collection(client)


def _seed_two_hop(col):
    """S (about the film/director) <-> A (about the director's age)."""
    ingestion.ingest_text(
        col,
        "The director of Inception is Christopher Nolan.\n"
        "Christopher Nolan was born on July 30, 1970.\n",
    )
    linking.run_linking_pass(col)


def test_expansion_pulls_linked_page(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Who directed Inception?", k=1)
    texts = [p["text"] for p in passages]
    # Seed answers who; the linked page (birth date) rides along via expansion.
    assert any("Christopher Nolan" in t and "director" in t for t in texts)
    assert any("1970" in t for t in texts)


def test_expansion_is_robust_to_which_seed_wins(tmp_path):
    """Whatever the top seed is, its top-1 link must be in the results."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    for query in ["Who directed Inception?", "When was Nolan born?", "Inception film"]:
        passages = retrieval.retrieve(col, query, k=1)
        ids = [p["page_id"] for p in passages]
        assert len(ids) == len(set(ids)), "duplicates leaked through"
        seed = storage.query(col, query, n_results=1)[0]
        if seed["links"]:
            assert seed["links"][0] in ids


def test_dedupe_with_mutual_links(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Inception Nolan", k=5)
    ids = [p["page_id"] for p in passages]
    assert len(ids) == len(set(ids))
    assert len(passages) == 2  # both pages, no duplicates despite mutual links


def test_empty_collection(tmp_path):
    col = _fresh_collection(tmp_path)
    assert retrieval.retrieve(col, "anything", k=4) == []


def test_provenance_marks_direct_vs_link(tmp_path):
    """0044: seeds are via=direct (with a score), expansions via=link + linked_from."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Who directed Inception?", k=1)
    by_id = {p["page_id"]: p for p in passages}
    assert len(passages) == 2
    seed = storage.query(col, "Who directed Inception?", n_results=1)[0]
    assert "score" in seed
    direct = by_id[seed["id"]]
    assert direct["via"] == "direct"
    assert "linked_from" not in direct
    assert direct["score"] == seed["score"]
    linked = [p for p in passages if p["page_id"] != seed["id"]][0]
    assert linked["via"] == "link"
    assert linked["linked_from"] == seed["id"]


def test_linked_score_is_a_real_distance_not_zero(tmp_path):
    """Regression for the 2026-09-26 bug: link-expansion rows always reported
    score=0.0 because _rank_neighbors computed real distances via its own
    ranking query but only returned ids, and expand_links then re-fetched
    the page with storage.get_page (a plain Chroma `get`, which never
    carries distances) and fell back to the 0.0 default."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Who directed Inception?", k=1)
    linked = [p for p in passages if p["via"] == "link"][0]
    assert linked["score"] != 0.0


def test_expansion_ranks_by_relevance_not_arbitrary_id_order(tmp_path):
    """0077: expansion used to take ``seed["links"][0]`` — links are stored
    sorted by page id (uuid4 hex), so the pick was arbitrary, not the most
    relevant neighbor. Give the irrelevant neighbor the alphabetically-first
    id (the old code would have picked it) and confirm ranking beats id order."""
    col = _fresh_collection(tmp_path)
    ingestion.ingest_text(col, "The director of Inception is Christopher Nolan.")
    storage.insert_page(
        col, "Christopher Nolan directed Dunkirk in 2017.",
        ["Christopher Nolan"], page_id="a" * 32,
    )
    storage.insert_page(
        col, "Christopher Nolan was born on July 30, 1970.",
        ["Christopher Nolan"], page_id="z" * 32,
    )
    linking.run_linking_pass(col)
    passages = retrieval.retrieve(col, "When was the director of Inception born?", k=1)
    linked_texts = [p["text"] for p in passages if p["via"] == "link"]
    assert linked_texts == ["Christopher Nolan was born on July 30, 1970."]


def test_expansion_without_query_falls_back_to_id_order(tmp_path):
    """expand_links(query=None) keeps the old (arbitrary but deterministic)
    id-sorted pick — callers that don't have a query still work."""
    col = _fresh_collection(tmp_path)
    seed_id = ingestion.ingest_text(col, "The director of Inception is Christopher Nolan.")[0]
    storage.insert_page(
        col, "Christopher Nolan directed Dunkirk in 2017.",
        ["Christopher Nolan"], page_id="a" * 32,
    )
    storage.insert_page(
        col, "Christopher Nolan was born on July 30, 1970.",
        ["Christopher Nolan"], page_id="z" * 32,
    )
    linking.run_linking_pass(col)
    seed = storage.get_page(col, seed_id)
    passages = retrieval.expand_links(col, [{**seed, "score": 0.0}])
    linked_texts = [p["text"] for p in passages if p["via"] == "link"]
    assert linked_texts == ["Christopher Nolan directed Dunkirk in 2017."]


def test_provenance_dedupe_keeps_direct(tmp_path):
    """A page that is both seed and expansion stays via=direct."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    passages = retrieval.retrieve(col, "Inception Nolan", k=5)
    by_id = {p["page_id"]: p for p in passages}
    assert all(p["via"] == "direct" for p in passages if "linked_from" not in p)
    # Mutual links: each page links the other, both are seeds — no link rows survive.
    assert all("linked_from" not in p for p in passages), by_id


# --- retrieve_fused (task 0093: RRF fusion + MMR) --------------------------


def test_retrieve_fused_pulls_linked_page(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    seeds = storage.query(col, "Who directed Inception?", n_results=1)
    passages = retrieval.retrieve_fused(col, seeds, "Who directed Inception?")
    texts = [p["text"] for p in passages]
    assert any("director" in t for t in texts)
    assert any("1970" in t for t in texts)


def test_retrieve_fused_dedupes_and_marks_provenance(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    seeds = storage.query(col, "Who directed Inception?", n_results=1)
    passages = retrieval.retrieve_fused(col, seeds, "Who directed Inception?")
    ids = [p["page_id"] for p in passages]
    assert len(ids) == len(set(ids))
    seed_id = seeds[0]["id"]
    direct = next(p for p in passages if p["page_id"] == seed_id)
    assert direct["via"] == "direct"
    assert "linked_from" not in direct
    linked = next(p for p in passages if p["page_id"] != seed_id)
    assert linked["via"] == "link"
    assert linked["linked_from"] == seed_id
    assert "created_at" not in linked  # internal fusion field, stripped from output


def test_retrieve_fused_ranks_the_full_neighbor_pool_not_just_per_seed_top1(tmp_path):
    """A seed with three linked candidates: fusion should be able to surface
    the most query-relevant one even though expand_links(top_n=1) would only
    ever have looked at each seed's own single best pick — here there's only
    one seed, so this also proves the ranking-over-the-whole-pool wiring.

    Links are wired directly (not via the entity-overlap linking pass, whose
    0011 denoising would drop a name shared by 4+ pages below its minimum
    link weight) — this test is about neighbor RANKING, not linking itself.
    """
    col = _fresh_collection(tmp_path)
    seed_id = ingestion.ingest_text(col, "The director of Inception is Christopher Nolan.")[0]
    for pid, text in (
        ("a" * 32, "Christopher Nolan directed Dunkirk in 2017."),
        ("b" * 32, "Christopher Nolan enjoys long walks."),
        ("z" * 32, "Christopher Nolan was born on July 30, 1970."),
    ):
        storage.insert_page(col, text, ["Christopher Nolan"], page_id=pid)
        storage.append_link(col, seed_id, pid)
    seeds = [storage.get_page(col, seed_id)]
    seeds[0]["score"] = 0.0
    passages = retrieval.retrieve_fused(col, seeds, "When was the director of Inception born?")
    linked_texts = [p["text"] for p in passages if p["via"] == "link"]
    assert any("1970" in t for t in linked_texts)


def test_retrieve_fused_empty_seeds(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    assert retrieval.retrieve_fused(col, [], "anything") == []


def test_retrieve_fused_no_mmr_still_dedupes(tmp_path):
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    seeds = storage.query(col, "Inception Nolan", n_results=5)
    passages = retrieval.retrieve_fused(col, seeds, "Inception Nolan", use_mmr=False)
    ids = [p["page_id"] for p in passages]
    assert len(ids) == len(set(ids))
    assert len(passages) == 2  # both pages, mutual links, no duplicates


def test_retrieve_fused_temporal_query_does_not_crash_or_drop_results(tmp_path):
    """The recency voice only joins fusion for temporal-looking queries — just
    confirm it wires in without breaking dedup/count when it does."""
    col = _fresh_collection(tmp_path)
    _seed_two_hop(col)
    seeds = storage.query(col, "When was the director born?", n_results=5)
    passages = retrieval.retrieve_fused(col, seeds, "When was the director born?")
    assert len({p["page_id"] for p in passages}) == len(passages)
    assert len(passages) == 2
