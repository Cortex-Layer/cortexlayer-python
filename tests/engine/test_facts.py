"""Fact-memory engine (Mem0-style LLM extraction + entity-boosted search).

No Ollama needed: the LLM is a script, the embedder a deterministic hash of
words. Algorithm parity with real Mem0 is checked separately (see task 0078)."""

import hashlib
import json
import math
import re

import httpx
import pytest

from cortexlayer import (
    CortexConfigError, InvalidRequestError, LLMError, Memory, NotFoundError,
)
from cortexlayer._engine import storage
from cortexlayer._engine.facts import backends, entities, prompts, scoring
from cortexlayer._engine.facts.engine import FactEngine, parse_extraction


# --- doubles ----------------------------------------------------------------------


class FakeEmbedder:
    name = "fake:hash128"

    def __init__(self):
        self.calls = []

    @staticmethod
    def vec(text):
        v = [0.0] * 128
        for w in re.findall(r"[a-z0-9']+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_batch(self, texts, action="add"):
        self.calls.append((action, list(texts)))
        return [self.vec(t) for t in texts]


def new_block(user_prompt):
    return user_prompt.split("## New Messages\n")[1].split("\n\n## Observation Date")[0]


class ScriptedLLM:
    """One fact per sentence of the new message, unless a scripted reply is queued."""

    def __init__(self):
        self.calls = []
        self.queue = []

    def generate(self, system, user):            # the LLM protocol
        return self(system, user)

    def __call__(self, system, user):
        self.calls.append((system, user))
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        facts = []
        for line in new_block(user).replace("user: ", "", 1).strip().splitlines():
            for sent in re.split(r"(?<=[.!?])\s+", line.strip()):
                if len(sent) > 12:
                    facts.append({"id": str(len(facts)), "text": sent, "attributed_to": "user"})
        return json.dumps({"memory": facts})


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def emb():
    return FakeEmbedder()


@pytest.fixture
def eng(tmp_path, spacy_nlp, llm, emb):
    client = storage.get_client(str(tmp_path / "s"))
    return FactEngine(client, str(tmp_path / "s"), backends.resolve_llm(llm), emb, spacy_nlp)


# --- reply parsing -------------------------------------------------------------------


@pytest.mark.parametrize("reply,n", [
    ('{"memory": [{"id": "0", "text": "A fact."}]}', 1),
    ('```json\n{"memory": [{"text": "A fact."}]}\n```', 1),
    ('<think>hmm</think>{"memory": [{"text": "A fact."}, {"text": "B fact."}]}', 2),
    ('Sure! Here you go: {"memory": [{"text": "A fact."}]} Hope that helps', 1),
    ('{"memory": []}', 0), ('', 0), (None, 0), ('not json at all', 0),
    ('{"memory": "nope"}', 0), ('{"other": 1}', 0), ('[1, 2]', 0),
    ('{"memory": [1, "x", {"text": "kept"}]}', 1),                     # non-dict entries dropped
    (['{"memory": [{"text": "A fact."}]}'], 1),                       # list-of-strings content
])
def test_parse_extraction(reply, n):
    assert len(parse_extraction(reply)) == n


# --- scoring math ---------------------------------------------------------------------


def test_distance_to_score_and_entity_boost():
    assert scoring.distance_to_score(0.0) == 1.0
    assert scoring.distance_to_score(1.0) == 0.5
    assert scoring.entity_boost(1.0, 1) == 0.5                          # sim * 0.5
    assert scoring.entity_boost(0.8, 1) == pytest.approx(0.4)
    assert scoring.entity_boost(1.0, 100) < scoring.entity_boost(1.0, 2) < 0.5   # crowded entities count less
    assert scoring.entity_boost(1.0, 0) == 0.5                          # n floored at 1


def test_score_and_rank_divisor_threshold_and_topk():
    cands = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.6}, {"id": "c", "score": 0.05}]
    plain = scoring.score_and_rank(cands, {}, threshold=0.1, top_k=5)
    assert [r["id"] for r in plain] == ["a", "b"]                       # c gated by the threshold
    assert plain[0]["score"] == pytest.approx(0.9)                      # divisor 1.0
    boosted = scoring.score_and_rank(cands, {"b": 0.5, "c": 0.5}, threshold=0.1, top_k=5)
    assert [r["id"] for r in plain] == ["a", "b"] and [r["id"] for r in boosted] == ["b", "a"]   # the boost reorders
    assert {r["id"] for r in boosted} == {"a", "b"}                     # the boost never rescues a sub-threshold hit
    b = next(r for r in boosted if r["id"] == "b")
    assert b["score"] == pytest.approx((0.6 + 0.5) / 1.5)               # divisor 1.5 once any boost exists
    assert len(scoring.score_and_rank(cands, {}, 0.0, top_k=1)) == 1
    assert all(r["score"] <= 1.0 for r in scoring.score_and_rank([{"id": "x", "score": 1.0}], {"x": 0.5}, 0, 1))
    assert scoring.score_and_rank([{"id": None, "score": 1}], {}, 0, 5) == []


# --- add -------------------------------------------------------------------------------


def test_add_extracts_and_stores_facts_with_metadata(eng):
    out = eng.add("u", "Caroline adopted a dog named Max. Melanie visited Paris.")
    assert [f["text"] for f in out] == ["Caroline adopted a dog named Max.", "Melanie visited Paris."]
    col = eng.collection("u")
    page = storage.get_page(col, out[0]["id"])
    assert page["text"] == out[0]["text"] and "Caroline" in page["entities"]
    meta = col.get(ids=[out[0]["id"]], include=["metadatas"])["metadatas"][0]
    assert meta["hash"] == hashlib.md5(out[0]["text"].encode()).hexdigest()
    assert meta["attributed_to"] == "user" and meta["updated_at"]


def test_add_with_tags_stamps_every_extracted_fact(eng):
    """Task 0090: caller-supplied tags must not clobber, and must not be
    clobbered by, the engine's own extra metadata (hash/updated_at/attributed_to)."""
    out = eng.add(
        "u", "Caroline adopted a dog named Max. Melanie visited Paris.",
        tags={"agent_id": "planner"},
    )
    col = eng.collection("u")
    for f in out:
        page = storage.get_page(col, f["id"])
        assert page["tags"] == {"agent_id": "planner"}
        meta = col.get(ids=[f["id"]], include=["metadatas"])["metadatas"][0]
        assert meta["attributed_to"] == "user" and meta["hash"]         # untouched by tags


def test_timestamp_is_prefixed_and_stored(eng, llm):
    out = eng.add("u", "Caroline adopted a dog named Max.", timestamp="8 May, 2023")
    assert out[0]["text"].startswith("[8 May, 2023]")
    assert storage.get_page(eng.collection("u"), out[0]["id"])["created_at"] == "8 May, 2023"
    assert "[8 May, 2023] Caroline adopted" in llm.calls[0][1]


def test_nothing_extracted_returns_empty_but_still_remembers_the_message(eng, llm):
    llm.queue.append('{"memory": []}')
    assert eng.add("u", "Hi there, how are you doing today?") == []
    assert eng.collection("u").count() == 0
    eng.add("u", "Caroline adopted a dog named Max.")
    assert "Hi there, how are you doing today?" in llm.calls[1][1]       # in "Last k Messages"


def test_prompt_carries_context_and_hides_real_ids(eng, llm):
    first = eng.add("u", "Caroline adopted a dog named Max.")
    eng.add("u", "Max learned to swim in Boston harbour.")
    system, user = llm.calls[1]
    assert system == prompts.extraction_system_prompt()
    existing = json.loads(user.split("## Existing Memories\n")[1].split("\n\n## New Messages")[0])
    assert existing == [{"id": "0", "text": "Caroline adopted a dog named Max."}]
    assert first[0]["id"] not in user                                   # only "0".."9"
    assert "user: Caroline adopted a dog named Max." in user            # last-k messages
    assert user.endswith("# Output:")


def test_custom_instructions_and_observation_date(tmp_path, spacy_nlp, llm, emb):
    client = storage.get_client(str(tmp_path / "s"))
    plain = FactEngine(client, str(tmp_path / "s"), llm, emb, spacy_nlp, custom_instructions="Favor recall.")
    plain.add("u", "Caroline adopted a dog named Max.", timestamp="8 May, 2023")
    user = llm.calls[-1][1]
    assert "## Custom Instructions\nFavor recall." in user
    assert not user.split("## Observation Date\n")[1].startswith("8 May")   # default = mem0: today

    dated = FactEngine(client, str(tmp_path / "s"), llm, emb, spacy_nlp, observation_date_from_timestamp=True)
    dated.add("v", "Melanie visited Paris with Sarah.", timestamp="8 May, 2023")
    assert llm.calls[-1][1].split("## Observation Date\n")[1].startswith("8 May, 2023")


def test_duplicates_are_dropped_by_hash(eng, llm):
    fact = "Caroline adopted a dog named Max."
    a = eng.add("u", fact)
    assert len(a) == 1
    assert eng.add("u", fact) == []                                     # nearest existing fact has the same hash
    assert eng.collection("u").count() == 1
    llm.queue.append(json.dumps({"memory": [{"text": "Same fact twice."}, {"text": "Same fact twice."}]}))
    assert len(eng.add("u", "irrelevant")) == 1                         # within-batch dedup


def test_entity_and_embedding_context_use_the_right_actions(eng, emb):
    eng.add("u", "Caroline adopted a dog named Max.")
    eng.add("u", "Melanie visited Paris.")
    actions = [a for a, _ in emb.calls]
    assert "add" in actions and "search" in actions                     # facts+entities embed as 'add'; context query as 'search'


# --- failure modes ------------------------------------------------------------------------


def test_llm_outage_raises_and_stores_nothing(eng, llm):
    llm.queue.append(LLMError("ollama down"))
    with pytest.raises(LLMError, match="ollama down"):
        eng.add("u", "Caroline adopted a dog named Max.")
    llm.queue.append(RuntimeError("boom"))
    with pytest.raises(LLMError, match="boom"):
        eng.add("u", "Caroline adopted a dog named Max.")
    assert eng.collection("u").count() == 0
    assert eng._messages.last("u") == []                                 # a failed add is not "seen"


def test_garbled_reply_is_an_empty_result_not_an_error(eng, llm):
    llm.queue.append("I refuse to answer in JSON")
    assert eng.add("u", "Caroline adopted a dog named Max.") == []


def test_embedder_outage_raises_instead_of_dropping_facts(tmp_path, spacy_nlp, llm):
    class Down:
        name = "down"

        def embed_batch(self, texts, action="add"):
            raise ConnectionError("no route")

    client = storage.get_client(str(tmp_path / "s"))
    e = FactEngine(client, str(tmp_path / "s"), llm, Down(), spacy_nlp)
    with pytest.raises(LLMError, match="embedding failed"):
        e.add("u", "Caroline adopted a dog named Max.")


def test_partial_embedding_failure_skips_only_that_fact(tmp_path, spacy_nlp, llm):
    class Picky(FakeEmbedder):
        def embed_batch(self, texts, action="add"):
            if len(texts) > 1 and action == "add":
                raise ConnectionError("batch failed")
            if any("Paris" in t for t in texts) and action == "add":
                raise ConnectionError("this one fails")
            return super().embed_batch(texts, action)

    client = storage.get_client(str(tmp_path / "s"))
    e = FactEngine(client, str(tmp_path / "s"), llm, Picky(), spacy_nlp)
    out = e.add("u", "Caroline adopted a dog named Max. Melanie visited Paris.")
    assert [f["text"] for f in out] == ["Caroline adopted a dog named Max."]


def test_a_store_refuses_a_different_embedder(tmp_path, spacy_nlp, llm, emb):
    client = storage.get_client(str(tmp_path / "s"))
    FactEngine(client, str(tmp_path / "s"), llm, emb, spacy_nlp).add("u", "Caroline adopted a dog named Max.")
    other = FakeEmbedder()
    other.name = "fake:other"
    with pytest.raises(CortexConfigError, match="different embedders"):
        FactEngine(client, str(tmp_path / "s"), llm, other, spacy_nlp).collection("u")


# --- search --------------------------------------------------------------------------------


def _seed(eng):
    eng.add("u", "Christopher Nolan directed Inception. Christopher Nolan was born in London. "
                 "Melanie visited Paris with Sarah. Sarah moved to Berlin. Caroline adopted a dog named Max.")


def test_search_ranks_by_similarity_with_fused_scores(eng):
    _seed(eng)
    hits = eng.seeds("u", "Who directed Inception?", 3)
    assert hits[0]["text"] == "Christopher Nolan directed Inception."
    assert all(0 < h["score"] <= 1 for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert set(hits[0]) >= {"id", "text", "entities", "links", "created_at", "score"}
    assert eng.seeds("u", "anything", 3) != [] and eng.seeds("nobody", "anything", 3) == []


def test_entity_boosts_point_at_the_facts_linked_to_the_query_entity(tmp_path, spacy_nlp, llm, emb):
    """Only the fact linked (via the entity index) to the entity named in the query is
    boosted, by ~0.5 for an exact entity match."""
    client = storage.get_client(str(tmp_path / "s"))
    with_ent = FactEngine(client, str(tmp_path / "s"), llm, emb, spacy_nlp)
    llm.queue.append(json.dumps({"memory": [
        {"text": "Boston has a support group."},           # mentions Boston
        {"text": "Zebra runs quickly downtown."}]}))
    with_ent.add("u", "x")
    col = with_ent.collection("u")
    ids = {storage.get_page(col, i)["text"]: i for i in col.get()["ids"]}
    boosts = with_ent._entity_boosts("u", [("PROPER", "Boston")])
    assert ids["Boston has a support group."] in boosts and ids["Zebra runs quickly downtown."] not in boosts
    assert boosts[ids["Boston has a support group."]] == pytest.approx(0.5, abs=0.01)   # exact entity match ~ sim 1.0


def test_no_spacy_means_plain_semantic_search(tmp_path, llm, emb):
    from cortexlayer._engine.nlp import RegexNLP

    client = storage.get_client(str(tmp_path / "s"))
    e = FactEngine(client, str(tmp_path / "s"), llm, emb, RegexNLP())
    e.add("u", "Caroline adopted a dog named Max.")
    assert e._entities("u").count() == 0                                # no entity index without spaCy
    assert e.seeds("u", "Max the dog", 2)[0]["text"].startswith("Caroline")


# --- entity index -----------------------------------------------------------------------------


def test_entities_are_merged_across_facts_and_adds(eng):
    eng.add("u", "Caroline adopted a dog named Max. Caroline moved to Boston.")
    eng.add("u", "Caroline visited Paris.")
    col = eng._entities("u")
    got = col.get(include=["documents", "metadatas"])
    rows = {d.lower(): json.loads(m["linked_memory_ids"]) for d, m in zip(got["documents"], got["metadatas"])}
    assert len(rows["caroline"]) == 3                                   # one entity row, three facts
    assert len(got["documents"]) == len({d.lower() for d in got["documents"]})   # no duplicate entity rows


# --- edit / isolation -----------------------------------------------------------------------------


def test_update_reembeds_and_refreshes_hash(eng):
    pid = eng.add("u", "Caroline adopted a dog named Max.")[0]["id"]
    eng.update("u", pid, "Zebra crossing the Serengeti plains quickly.")
    page = storage.get_page(eng.collection("u"), pid)
    assert page["text"] == "Zebra crossing the Serengeti plains quickly."
    meta = eng.collection("u").get(ids=[pid], include=["metadatas"])["metadatas"][0]
    assert meta["hash"] == hashlib.md5(page["text"].encode()).hexdigest()
    assert eng.seeds("u", "Serengeti zebra", 1)[0]["id"] == pid


def test_update_preserves_tags(eng):
    pid = eng.add("u", "Caroline adopted a dog named Max.", tags={"source": "hermes"})[0]["id"]
    eng.update("u", pid, "Caroline adopted a cat named Max.")
    assert storage.get_page(eng.collection("u"), pid)["tags"] == {"source": "hermes"}


def test_users_are_isolated_including_context_and_entities(eng, llm):
    eng.add("alice", "Caroline adopted a dog named Max.")
    eng.add("bob", "Melanie visited Paris.")
    assert eng.collection("alice").count() == 1 and eng.collection("bob").count() == 1
    assert eng.seeds("bob", "dog named Max", 3)[0]["text"] == "Melanie visited Paris."   # only bob's facts exist for bob
    user = llm.calls[-1][1]                                             # bob's prompt
    assert "Max" not in user.split("## New Messages")[0]                # alice's fact/message never leak in
    eng.delete_all("alice")
    assert eng._messages.last("alice") == [] and eng._messages.last("bob") != []
    assert eng.collection("bob").count() == 1


# --- adapters (mock Ollama) --------------------------------------------------------------------------


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_llm_request_shape_and_reply():
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"memory": []}'}})

    out = backends.OllamaLLM("m1", "http://h:1", http_client=_mock(handler)).generate("SYS", "USER")
    assert out == '{"memory": []}' and seen["url"] == "http://h:1/api/chat"
    b = seen["body"]
    assert b["model"] == "m1" and b["format"] == "json" and b["stream"] is False and b["think"] is False
    assert b["options"] == {"temperature": 0.1, "top_p": 0.1, "num_predict": 2000}
    assert b["messages"][0] == {"role": "system", "content": "SYS"}
    assert b["messages"][1]["content"] == "USER\n\nPlease respond with valid JSON only."


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="boom"),
    lambda r: httpx.Response(200, json={"nope": 1}),
    lambda r: httpx.Response(200, text="not json"),
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")),
])
def test_ollama_llm_errors_become_llm_error(handler):
    with pytest.raises(LLMError):
        backends.OllamaLLM("m", "http://h", http_client=_mock(handler)).generate("s", "u")


def test_ollama_embedder_shape_errors_and_empty():
    def ok(request):
        body = json.loads(request.content)
        assert str(request.url) == "http://h/api/embed" and body["model"] == "e1"
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]] * len(body["input"])})

    e = backends.OllamaEmbedder("e1", "http://h", http_client=_mock(ok))
    assert e.name == "ollama:e1" and e.embed_batch(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]] and e.embed_batch([]) == []
    short = backends.OllamaEmbedder("e1", "http://h", http_client=_mock(lambda r: httpx.Response(200, json={"embeddings": [[1.0]]})))
    with pytest.raises(LLMError, match="1 embeddings for 2"):
        short.embed_batch(["a", "b"])
    with pytest.raises(LLMError):
        backends.OllamaEmbedder("e1", "http://h", http_client=_mock(lambda r: httpx.Response(500))).embed_batch(["a"])


def test_resolve_llm_and_embedder_specs(monkeypatch):
    assert isinstance(backends.resolve_llm(None), backends.OllamaLLM)
    assert backends.resolve_llm({"model": "x", "host": "http://h"}).model == "x"
    fn = lambda s, u: "r"
    assert backends.resolve_llm(fn).generate("s", "u") == "r"
    obj = type("O", (), {"generate": lambda self, s, u: "o"})()
    assert backends.resolve_llm(obj) is obj
    for bad in (5, "ollama", {"provider": "openai"}):
        with pytest.raises(CortexConfigError):
            backends.resolve_llm(bad)
    assert isinstance(backends.resolve_embedder(None), backends.ChromaEmbedder)
    assert isinstance(backends.resolve_embedder("ollama"), backends.OllamaEmbedder)
    assert backends.resolve_embedder({"provider": "ollama", "model": "m"}).name == "ollama:m"
    for bad in ("nope", 5, {"provider": "x"}):
        with pytest.raises(CortexConfigError):
            backends.resolve_embedder(bad)
    monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:9/")
    assert backends.OllamaLLM().host == "http://elsewhere:9"


# --- prompt + entity extractor ------------------------------------------------------------------------


def test_extraction_prompt_is_mems_unchanged_and_builder_matches_its_layout():
    sysprompt = prompts.extraction_system_prompt()
    assert sysprompt.startswith("\n\n# ROLE") and "ADD" in sysprompt and len(sysprompt) > 30000
    built = prompts.build_extraction_prompt(new_messages="user: hi\n", timestamp="8 May, 2023",
                                            current_date="2026-01-02", custom_instructions="X")
    assert "## New Messages\nuser: hi\n" in built and "## Observation Date\n8 May, 2023" in built
    assert "## Current Date\n2026-01-02" in built and "## Custom Instructions\nX" in built
    assert built.endswith("# Output:") and built.count("## ") >= 8
    long = prompts.build_extraction_prompt(last_k_messages=[{"role": "user", "content": "x" * 400}])
    assert "x" * 300 + "..." in long and "x" * 301 not in long        # last-k truncated to 300 chars


def test_entity_extractor_is_typed_and_optional(spacy_nlp):
    got = entities.extract_entities("Christopher Nolan directed Inception.", nlp=spacy_nlp._nlp)
    assert ("PROPER", "Christopher Nolan") in got and ("PROPER", "Inception") in got
    assert entities.extract_entities_batch([], nlp=spacy_nlp._nlp) == []
    class NoNLP:  # spaCy unavailable -> no entities, never an error (as mem0)
        pass
    entities._default_nlp_cache.update(nlp=None, tried=True)
    try:
        assert entities.extract_entities("Anything Goes") == []
        assert entities.extract_entities_batch(["a", "b"]) == [[], []]
    finally:
        entities._default_nlp_cache.update(nlp=None, tried=False)


# --- through Memory(backend="facts") ----------------------------------------------------------------------


@pytest.fixture
def fmem(tmp_path, spacy_nlp, llm, emb):
    return Memory(str(tmp_path / "store"), backend="facts", llm=llm, embedder=emb, entity_extractor=spacy_nlp)


def test_memory_facts_round_trip(fmem, llm):
    ids = fmem.add("Christopher Nolan directed Inception. Christopher Nolan was born in London.",
                   user_id="u").page_ids
    assert len(ids) == 2 and "facts" in repr(fmem)
    assert fmem.count(user_id="u") == 2
    assert fmem.relink(user_id="u")["links_written"] == 2               # entity-overlap linking runs on facts unchanged
    hits = fmem.search("Who directed Inception?", user_id="u", limit=1)
    assert [h.via for h in hits] == ["direct", "link"] and hits[1].linked_from == hits[0].id
    assert 0 < hits[0].score <= 1
    # Regression (found 2026-09-26 testing the Playground): the link hit's
    # score used to be a raw, unconverted Chroma distance (unbounded, and on
    # the opposite "lower is better" scale from the seed's fused [0, 1]
    # similarity) because expand_links() doesn't know this backend fuses and
    # rescales its seed scores. Same range check as the seed above.
    assert 0 < hits[1].score <= 1
    assert [h.via for h in fmem.search("Who directed Inception?", user_id="u", limit=1, expand_links=False)] == ["direct"]
    page = fmem.get(hits[0].id, user_id="u")
    assert page.degree == 1 and page.content == "Christopher Nolan directed Inception."
    fmem.update(page.id, "Christopher Nolan directed Interstellar.", user_id="u")
    assert fmem.get(page.id, user_id="u").content.endswith("Interstellar.")
    assert fmem.search("Interstellar", user_id="u", limit=1)[0].id == page.id
    fmem.delete(page.id, user_id="u")
    with pytest.raises(NotFoundError):
        fmem.get(page.id, user_id="u")
    assert fmem.delete_all(user_id="u") == 1 and fmem.count(user_id="u") == 0
    fmem.add("Melanie visited Paris.", user_id="u")                    # usable again after delete_all
    assert fmem.count(user_id="u") == 1


def test_memory_facts_tags_round_trip_through_get_and_search(fmem):
    ids = fmem.add("Caroline adopted a dog named Max.", user_id="u", tags={"source": "hermes"}).page_ids
    assert fmem.get(ids[0], user_id="u").tags == {"source": "hermes"}
    assert fmem.search("dog Max", user_id="u")[0].tags == {"source": "hermes"}


def test_memory_facts_answer_uses_the_fact_seeds(fmem):
    fmem.add("Caroline adopted a dog named Max.", user_id="u")
    out = fmem.answer("What is the dog called?", user_id="u",
                      chat=lambda p, m: '{"answer": "Max", "source_page_ids": []}')
    assert out.answer == "Max" and len(out.source_page_ids) == 1


def test_memory_facts_isolation_and_persistence(tmp_path, spacy_nlp, llm, emb):
    def make():
        return Memory(str(tmp_path / "p"), backend="facts", llm=llm, embedder=emb, entity_extractor=spacy_nlp)

    m = make()
    pid = m.add("Caroline adopted a dog named Max.", user_id="alice").page_ids[0]
    with pytest.raises(NotFoundError):
        m.get(pid, user_id="bob")
    assert m.search("dog Max", user_id="bob") == []
    again = make()
    assert again.get(pid, user_id="alice").content == "Caroline adopted a dog named Max."
    again.add("Max learned to swim in Boston harbour.", user_id="alice")
    assert "Caroline adopted a dog named Max." in llm.calls[-1][1]      # last-k context survives a restart


def test_memory_facts_config_errors_and_defaults(tmp_path, spacy_nlp, emb):
    with pytest.raises(CortexConfigError, match="backend"):
        Memory(str(tmp_path), backend="nope")
    m = Memory.from_config({"data_dir": str(tmp_path / "c"), "backend": "facts", "entity_extractor": spacy_nlp,
                            "embedder": emb, "llm": lambda s, u: '{"memory": []}',
                            "custom_instructions": "Be brief.", "observation_date_from_timestamp": True})
    assert m.add("hello there friend", user_id="u").page_ids == []
    # Constructing with defaults needs no Ollama; using it does, and says so via LLMError.
    d = Memory(str(tmp_path / "d"), backend="facts", entity_extractor=spacy_nlp, embedder=emb,
               llm={"host": "http://127.0.0.1:9", "model": "x"})
    with pytest.raises(LLMError, match="127.0.0.1:9"):
        d.add("Caroline adopted a dog named Max.")
    assert d.count() == 0


def test_raw_and_facts_stores_do_not_collide(tmp_path, spacy_nlp, llm, emb):
    store = str(tmp_path / "both")
    raw = Memory(store, entity_extractor=spacy_nlp)
    facts = Memory(store, backend="facts", llm=llm, embedder=emb, entity_extractor=spacy_nlp)
    raw.add("Raw note about Caroline.", user_id="u")
    facts.add("Caroline adopted a dog named Max.", user_id="u")
    assert raw.count(user_id="u") == 1 and facts.count(user_id="u") == 1
    assert raw.get_all(user_id="u").pages[0].content.startswith("Raw note")
    assert facts.get_all(user_id="u").pages[0].content.startswith("Caroline adopted")


def test_facts_validation_is_shared_with_raw(fmem):
    for call in (lambda: fmem.add(""), lambda: fmem.search("q", limit=0), lambda: fmem.get(""),
                 lambda: fmem.add("x", user_id="../bad")):
        with pytest.raises(InvalidRequestError):
            call()


# --- bulk import (migration from a Mem0 store) ----------------------------------------


def test_import_facts_keeps_ids_is_idempotent_and_searchable(eng, emb):
    facts = [
        {"id": "f1", "text": "Miguel lives in Dallas.", "vector": emb.vec("Miguel lives in Dallas."),
         "created_at": "2026-09-20T18:15:06+00:00", "hash": "h1", "attributed_to": "user"},
        {"id": "f2", "text": "Miguel has a daily news brief.", "vector": emb.vec("Miguel has a daily news brief.")},
    ]
    ents = [
        {"id": "e1", "text": "Miguel", "type": "PROPER", "vector": emb.vec("Miguel"),
         "linked_memory_ids": ["f1", "f2", "gone"]},
        {"id": "e2", "text": "Nobody", "type": "PROPER", "vector": emb.vec("Nobody"),
         "linked_memory_ids": ["gone"]},   # points only at a deleted memory: dropped
    ]
    assert eng.import_facts("u", facts, ents) == {
        "facts": 2, "facts_skipped": 0, "entities": 1, "entities_skipped": 1}
    col = eng.collection("u")
    page = storage.get_page(col, "f1")
    assert page["text"] == "Miguel lives in Dallas." and page["created_at"] == "2026-09-20T18:15:06+00:00"
    assert col.get(ids=["f1"])["metadatas"][0]["hash"] == "h1"
    ecol = eng._entities("u")
    assert ecol.count() == 1
    assert storage._decode_list(ecol.get(ids=["e1"])["metadatas"][0]["linked_memory_ids"]) == ["f1", "f2"]

    # A second run changes nothing.
    again = eng.import_facts("u", facts, ents)
    assert again["facts"] == 0 and again["facts_skipped"] == 2 and again["entities"] == 0
    assert col.count() == 2 and ecol.count() == 1

    # Imported facts are ordinary facts: searchable, entity-boosted.
    top = eng.seeds("u", "Where does Miguel live in Dallas?", 2)
    assert top[0]["id"] == "f1"
