"""``Memory`` — the embedded engine behind the mem0-style API."""

import inspect
import sys

import pytest

from cortexlayer import (
    AddResult,
    Answer,
    CortexClient,
    CortexConfigError,
    InvalidRequestError,
    LocalDependencyError,
    Memory,
    NotFoundError,
    Page,
    PageList,
    SearchResult,
)


# --- round trip -----------------------------------------------------------------


def test_add_get_search_update_delete_round_trip(mem):
    ids = mem.add("Alice moved to Lisbon in March.").page_ids
    assert len(ids) == 1 and isinstance(mem.add("x y z"), AddResult)

    page = mem.get(ids[0])
    assert isinstance(page, Page)
    assert page.content == "Alice moved to Lisbon in March."
    assert page.title == "Alice moved to Lisbon in March."

    hits = mem.search("where did Alice move?")
    assert isinstance(hits[0], SearchResult) and hits[0].id == ids[0]
    assert hits[0].text == "Alice moved to Lisbon in March."       # full text, not just a snippet
    assert hits[0].via == "direct" and hits[0].linked_from is None

    assert mem.update(ids[0], "Alice moved to Porto.") is None
    assert mem.get(ids[0]).content == "Alice moved to Porto."
    assert mem.search("Porto")[0].id == ids[0]

    assert mem.delete(ids[0]) is None
    with pytest.raises(NotFoundError):
        mem.get(ids[0])
    assert mem.count() == 1


def test_data_persists_across_instances(tmp_path, spacy_nlp):
    store = str(tmp_path / "s")
    pid = Memory(store, entity_extractor=spacy_nlp).add("Alice likes tea.").page_ids[0]
    again = Memory(store, entity_extractor=spacy_nlp)
    assert again.get(pid).content == "Alice likes tea."
    assert again.search("tea")[0].id == pid


def test_search_on_an_empty_store_is_empty(mem):
    assert mem.search("anything") == []
    assert mem.get_all().total == 0 and mem.count() == 0
    assert mem.answer("anything", chat=lambda p, m: "{}") == Answer("", [])


def test_search_returns_the_full_text_of_long_memories(mem):
    long = "Alice " + "really ".join(f"likes thing{i} " for i in range(60)) + "a lot."
    assert len(long) > 400
    assert len(mem.add(long).page_ids) == 1            # one long sentence stays one page
    hit = mem.search("Alice likes thing3")[0]
    assert len(hit.snippet) <= 200 and len(hit.text) > 200 and hit.text.startswith("Alice")


def test_timestamp_and_chunking(mem):
    pid = mem.add("Alice moved to Lisbon.", timestamp="8 May, 2023").page_ids[0]
    page = mem.get(pid)
    assert page.created_at == "8 May, 2023" and page.content.startswith("[8 May, 2023]")
    long_text = "\n".join(f"Fact {i} about Bob and his travels." for i in range(12))
    assert len(mem.add(long_text).page_ids) == 12


# --- links ----------------------------------------------------------------------


def _seed(mem, **kw):
    mem.add("Christopher Nolan directed Inception.", **kw)
    mem.add("Christopher Nolan was born in London in 1970.", **kw)
    mem.add("Bob likes hiking.", **kw)


def test_relink_creates_links_and_search_expands_through_them(mem):
    _seed(mem)
    assert all(p.degree == 0 for p in mem.get_all().pages)
    stats = mem.relink()
    assert stats["pages"] == 3 and stats["links_written"] == 2

    pages = {p.title: p for p in mem.get_all().pages}
    a, b = pages["Christopher Nolan directed Inception."], pages["Christopher Nolan was born in London in 1970."]
    assert a.links == [b.id] and a.linked_from == [b.id] and a.degree == 1
    assert pages["Bob likes hiking."].degree == 0

    hits = mem.search("Who directed Inception?", limit=1)
    assert [h.via for h in hits] == ["direct", "link"]
    assert hits[1].id == b.id and hits[1].linked_from == a.id      # provenance
    assert [h.via for h in mem.search("Who directed Inception?", limit=1, expand_links=False)] == ["direct"]


def test_relink_is_idempotent_and_auto_relink_works(tmp_path, spacy_nlp):
    m = Memory(str(tmp_path), entity_extractor=spacy_nlp, auto_relink=True)
    _seed(m)
    assert all(p.degree == 1 for p in m.get_all().pages if "Nolan" in p.content)
    assert m.relink()["links_written"] == 0


def test_delete_leaves_no_dangling_links_after_relink(mem):
    _seed(mem)
    mem.relink()
    victim = next(p for p in mem.get_all().pages if "directed" in p.content)
    mem.delete(victim.id)
    survivor = next(p for p in mem.get_all().pages if "born" in p.content)
    assert survivor.links == [] and survivor.linked_from == []     # unknown ids filtered
    assert mem.search("Nolan born")[0].id == survivor.id            # no crash on stale link
    mem.relink()
    assert mem.get(survivor.id).degree == 0


# --- tags (provenance metadata, task 0090) ---------------------------------------


def test_add_with_tags_round_trips_through_get_search_and_get_all(mem):
    tagged = {"agent_id": "planner", "session_id": "abc123", "priority": 2, "verified": True}
    pid = mem.add("Alice moved to Lisbon in March.", tags=tagged).page_ids[0]

    assert mem.get(pid).tags == tagged
    assert mem.search("Alice Lisbon")[0].tags == tagged
    page = next(p for p in mem.get_all().pages if p.id == pid)
    assert page.tags == tagged


def test_omitting_tags_behaves_exactly_as_today(mem):
    pid = mem.add("Bob likes tea.").page_ids[0]
    assert mem.get(pid).tags == {}
    assert mem.search("Bob tea")[0].tags == {}
    assert mem.get_all().pages[0].tags == {}


def test_tags_are_provenance_not_a_filter(mem):
    """No tag-based filtering exists anywhere — search/get_all keep spanning
    the whole collection whether or not a page (or the query) mentions tags."""
    a = mem.add("Carol wrote a report.", tags={"source": "agent-a"}).page_ids[0]
    b = mem.add("Carol wrote another report.").page_ids[0]  # no tags at all

    ids = {p.id for p in mem.get_all().pages}
    assert ids == {a, b}
    hits = {h.id for h in mem.search("Carol report", limit=2)}
    assert hits == {a, b}


def test_tags_are_carried_through_link_expansion(mem):
    _seed(mem, tags={"source": "import"})
    mem.relink()
    hits = mem.search("Who directed Inception?", limit=1)
    assert [h.via for h in hits] == ["direct", "link"]
    assert hits[0].tags == {"source": "import"} and hits[1].tags == {"source": "import"}


def test_bad_tags_raise_invalid_request(mem):
    for bad in (
        "nope", 1, ["a"],
        {1: "int key"}, {"": "empty key"},
        {"nested": {"a": 1}}, {"list": [1, 2]}, {"none": None},
    ):
        with pytest.raises(InvalidRequestError):
            mem.add("x", tags=bad)


def test_empty_tags_dict_is_treated_like_no_tags(mem):
    pid = mem.add("x", tags={}).page_ids[0]
    assert mem.get(pid).tags == {}


# --- browse ---------------------------------------------------------------------


def test_get_all_pagination_and_filter(mem):
    for i in range(7):
        mem.add(f"Item number {i} about Carol.")
    mem.add("Something else entirely.")
    first = mem.get_all(limit=3)
    assert isinstance(first, PageList) and len(first.pages) == 3 and first.total == 8
    assert first.limit == 3 and first.offset == 0
    last = mem.get_all(limit=3, offset=6)
    assert len(last.pages) == 2
    filtered = mem.get_all(query="CAROL", limit=50)
    assert filtered.total == 7 and all("Carol" in p.content for p in filtered.pages)
    assert mem.get_all(query="nomatch").total == 0
    seen = {p.id for p in mem.get_all(limit=3).pages} | {p.id for p in mem.get_all(limit=3, offset=3).pages} \
        | {p.id for p in mem.get_all(limit=3, offset=6).pages}
    assert len(seen) == 8                                            # windows don't overlap


# --- users ----------------------------------------------------------------------


def test_users_cannot_see_or_touch_each_other(mem):
    a = mem.add("Alice's vault code is 4471.", user_id="alice").page_ids[0]
    mem.add("Bob likes tea.", user_id="bob")

    assert mem.search("vault code", user_id="bob")[0].id != a
    assert all("4471" not in h.snippet for h in mem.search("vault code", user_id="bob"))
    with pytest.raises(NotFoundError):
        mem.get(a, user_id="bob")
    with pytest.raises(NotFoundError):
        mem.update(a, "pwned", user_id="bob")
    with pytest.raises(NotFoundError):
        mem.delete(a, user_id="bob")
    assert mem.get(a, user_id="alice").content.endswith("4471.")
    assert mem.count(user_id="alice") == 1 and mem.count(user_id="bob") == 1
    assert mem.count() == 0                                          # default user is separate


def test_delete_all_only_clears_one_user(mem):
    mem.add("one", user_id="alice")
    mem.add("Two things.\nThree things.", user_id="alice")
    mem.add("keep me", user_id="bob")
    assert mem.delete_all(user_id="alice") == 3
    assert mem.count(user_id="alice") == 0 and mem.count(user_id="bob") == 1
    mem.add("fresh start", user_id="alice")                          # usable again afterwards
    assert mem.count(user_id="alice") == 1
    assert mem.delete_all(user_id="nobody") == 0


def test_default_user_id_and_validation(tmp_path, spacy_nlp):
    m = Memory(str(tmp_path), entity_extractor=spacy_nlp, default_user_id="carol")
    m.add("Carol's note.")
    assert m.count(user_id="carol") == 1 and m.count(user_id="default") == 0
    for bad in ("", "../etc", "a b", "x" * 65, "-lead"):
        with pytest.raises(InvalidRequestError):
            m.add("x", user_id=bad)
        with pytest.raises(InvalidRequestError):
            Memory(str(tmp_path / "z"), entity_extractor=spacy_nlp, default_user_id=bad)


# --- validation -----------------------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda m: m.add(""), lambda m: m.add(None), lambda m: m.add("x", timestamp=""),
    lambda m: m.search(""), lambda m: m.search("q", limit=0), lambda m: m.search("q", limit=51),
    lambda m: m.search("q", limit=True), lambda m: m.search("q", expand_links="yes"),
    lambda m: m.get(""), lambda m: m.update("p", ""), lambda m: m.update("", "x"),
    lambda m: m.delete(""), lambda m: m.get_all(limit=501), lambda m: m.get_all(offset=-1),
    lambda m: m.get_all(query=""), lambda m: m.answer(""), lambda m: m.answer("q", limit=0),
])
def test_bad_arguments_raise_invalid_request(mem, call):
    with pytest.raises(InvalidRequestError):
        call(mem)


def test_unknown_ids_are_not_found(mem):
    for call in (lambda: mem.get("nope"), lambda: mem.update("nope", "x"), lambda: mem.delete("nope")):
        with pytest.raises(NotFoundError) as ei:
            call()
        assert ei.value.status == 404 and "nope" in str(ei.value)


# --- answer (LLM step) ----------------------------------------------------------


def test_answer_uses_the_supplied_chat_function(mem):
    pid = mem.add("Alice moved to Lisbon in March.").page_ids[0]
    seen = {}

    def chat(prompt, model):
        seen["prompt"], seen["model"] = prompt, model
        return f'{{"answer": "Lisbon", "source_page_ids": ["{pid}"]}}'

    out = mem.answer("Where did Alice move?", chat=chat, model="my-model")
    assert out == Answer(answer="Lisbon", source_page_ids=[pid])
    assert "Alice moved to Lisbon" in seen["prompt"] and "Where did Alice move?" in seen["prompt"]
    assert seen["model"] == "my-model"


def test_answer_falls_back_to_raw_text_and_all_ids_on_bad_json(mem):
    pid = mem.add("Alice moved to Lisbon.").page_ids[0]
    out = mem.answer("where?", chat=lambda p, m: "Lisbon, I think")
    assert out.answer == "Lisbon, I think" and out.source_page_ids == [pid]


def test_answer_propagates_llm_failures(mem):
    mem.add("Alice moved to Lisbon.")

    def down(prompt, model):
        raise ConnectionError("ollama is down")

    with pytest.raises(ConnectionError):
        mem.answer("where?", chat=down)


# --- config ---------------------------------------------------------------------


def test_from_config(tmp_path, spacy_nlp):
    m = Memory.from_config({
        "data_dir": str(tmp_path / "cfg"), "entity_extractor": spacy_nlp,
        "default_user_id": "dora", "auto_relink": True, "backend": "raw",
    })
    m.add("Dora likes Paris.")
    assert m.count(user_id="dora") == 1


def test_from_config_rejects_bad_config(tmp_path):
    with pytest.raises(CortexConfigError, match="unknown config keys: nope"):
        Memory.from_config({"nope": 1})
    with pytest.raises(CortexConfigError, match="mem0"):
        Memory.from_config({"backend": "mem0"})
    with pytest.raises(CortexConfigError):
        Memory.from_config("not a dict")


def test_data_dir_env_var_and_default(tmp_path, monkeypatch, spacy_nlp):
    monkeypatch.setenv("CORTEXLAYER_DATA_DIR", str(tmp_path / "envstore"))
    m = Memory(entity_extractor=spacy_nlp)
    m.add("hello there")
    assert (tmp_path / "envstore").is_dir() and str(tmp_path / "envstore") in repr(m)
    monkeypatch.delenv("CORTEXLAYER_DATA_DIR")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert ".cortexlayer'" in repr(Memory(entity_extractor=spacy_nlp))


# --- parity with the hosted client ----------------------------------------------


def test_memory_and_client_share_the_same_memory_api():
    shared = ("add", "search", "get", "get_all", "update", "delete", "relink")
    for name in shared:
        assert hasattr(Memory, name) and hasattr(CortexClient, name), name
        local = inspect.signature(getattr(Memory, name)).parameters
        hosted = inspect.signature(getattr(CortexClient, name)).parameters
        assert set(hosted) - {"self"} <= set(local) - {"self"} , (name, set(hosted) ^ set(local))


# --- extras are optional --------------------------------------------------------


def test_importing_cortexlayer_does_not_load_the_engine():
    import subprocess

    code = (
        "import sys, cortexlayer; "
        "print(any(m.startswith(('chromadb','spacy')) for m in sys.modules), "
        "'cortexlayer._engine' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_memory_without_the_local_extra_explains_the_fix(monkeypatch):
    for name in [n for n in sys.modules if n.startswith(("cortexlayer._engine", "chromadb", "spacy"))]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "chromadb", None)          # makes `import chromadb` fail
    with pytest.raises(LocalDependencyError, match=r"cortexlayer\[local\]") as ei:
        Memory("/tmp/never-created")
    assert isinstance(ei.value, ImportError)
