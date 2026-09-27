"""Supersede-on-write for the fact engine (task 0094). Fakes only — no models,
following 0079's reproduce-the-failing-case pattern (see test_facts_keyword.py)."""

import json

import pytest

from cortexlayer import Memory
from cortexlayer._engine import storage
from cortexlayer._engine.facts import backends
from cortexlayer._engine.facts.engine import FactEngine, _is_current, _parse_iso

from test_facts import FakeEmbedder, ScriptedLLM  # noqa: E402  (same doubles as the 0078 tests)

T1 = "2023-01-01T00:00:00+00:00"   # "User prefers Python" written
T2 = "2023-06-01T00:00:00+00:00"   # "User prefers Rust" written, later
BETWEEN = "2023-03-01T00:00:00+00:00"


def _engine(tmp_path, spacy_nlp, name, *, supersede):
    llm = ScriptedLLM()
    client = storage.get_client(str(tmp_path / name))
    eng = FactEngine(
        client, str(tmp_path / name), backends.resolve_llm(llm), FakeEmbedder(),
        spacy_nlp, supersede=supersede,
    )
    eng._test_llm = llm  # stash for queueing replies below
    return eng


def _write_preference(eng, text, timestamp):
    eng._test_llm.queue.append(json.dumps({
        "memory": [{"id": "0", "text": text, "subject": "User", "predicate": "prefers"}]
    }))
    return eng.add("u", text, timestamp=timestamp)[0]["id"]


# --- the target failure mode + the fix -------------------------------------------------


def test_reproduces_the_failure_and_supersede_fixes_it(tmp_path, spacy_nlp):
    """A preference stated twice at different times: append-only returns both
    (ambiguous — which is current?); supersede-on-write returns only the
    current one by default, and the old one under `as_of` history."""
    off = _engine(tmp_path, spacy_nlp, "off", supersede=False)
    _write_preference(off, "User prefers Python.", T1)
    _write_preference(off, "User prefers Rust.", T2)
    off_texts = {p["text"] for p in off.seeds("u", "What does the user prefer?", 10)}
    assert off_texts == {"User prefers Python.", "User prefers Rust."}   # BUG: both, ambiguous

    on = _engine(tmp_path, spacy_nlp, "on", supersede=True)
    old_id = _write_preference(on, "User prefers Python.", T1)
    new_id = _write_preference(on, "User prefers Rust.", T2)

    current = on.seeds("u", "What does the user prefer?", 10)
    assert [p["text"] for p in current] == ["User prefers Rust."]        # FIX: only current by default
    assert current[0]["id"] == new_id

    history = on.seeds("u", "What does the user prefer?", 10, as_of=BETWEEN)
    assert [p["text"] for p in history] == ["User prefers Python."]      # old fact still reachable via as_of

    old_page = storage.get_page(on.collection("u"), old_id)
    assert old_page["created_at"] == T1
    meta = on.collection("u").get(ids=[old_id], include=["metadatas"])["metadatas"][0]
    assert meta["valid_until"] == T2                                    # closed exactly at the new fact's valid_from


def test_unrelated_predicate_is_not_superseded(tmp_path, spacy_nlp):
    """Same subject, different predicate: no supersession — 0094 must not
    over-trigger just because two facts are about the same entity."""
    on = _engine(tmp_path, spacy_nlp, "distinct", supersede=True)
    on._test_llm.queue.append(json.dumps({"memory": [
        {"id": "0", "text": "User prefers Python.", "subject": "User", "predicate": "prefers"}
    ]}))
    on.add("u", "User prefers Python.", timestamp=T1)
    on._test_llm.queue.append(json.dumps({"memory": [
        {"id": "0", "text": "User lives in Lisbon.", "subject": "User", "predicate": "lives_in"}
    ]}))
    on.add("u", "User lives in Lisbon.", timestamp=T2)
    texts = {p["text"] for p in on.seeds("u", "user facts", 10)}
    assert texts == {"User prefers Python.", "User lives in Lisbon."}   # both current, no false supersession


def test_subject_match_is_case_and_whitespace_insensitive(tmp_path, spacy_nlp):
    on = _engine(tmp_path, spacy_nlp, "norm", supersede=True)
    on._test_llm.queue.append(json.dumps({"memory": [
        {"id": "0", "text": "User prefers Python.", "subject": "  User ", "predicate": "Prefers"}
    ]}))
    on.add("u", "User prefers Python.", timestamp=T1)
    on._test_llm.queue.append(json.dumps({"memory": [
        {"id": "0", "text": "User prefers Rust.", "subject": "User", "predicate": "prefers"}
    ]}))
    on.add("u", "User prefers Rust.", timestamp=T2)
    assert [p["text"] for p in on.seeds("u", "prefers", 10)] == ["User prefers Rust."]


def test_within_batch_chaining_closes_the_middle_fact_not_just_the_first(tmp_path, spacy_nlp):
    """Two facts sharing (subject, predicate) extracted in the SAME add() call
    (e.g. a message that states a preference twice) must chain correctly:
    the earlier of the two closes, not the pre-existing one twice."""
    on = _engine(tmp_path, spacy_nlp, "chain", supersede=True)
    _write_preference(on, "User prefers Python.", T1)
    on._test_llm.queue.append(json.dumps({"memory": [
        {"id": "0", "text": "User prefers Go.", "subject": "User", "predicate": "prefers"},
        {"id": "1", "text": "User prefers Rust.", "subject": "User", "predicate": "prefers"},
    ]}))
    on.add("u", "User prefers Go, then changed their mind to Rust.", timestamp=T2)
    assert [p["text"] for p in on.seeds("u", "prefers", 10)] == ["User prefers Rust."]


def test_supersede_off_never_closes_anything(tmp_path, spacy_nlp):
    off = _engine(tmp_path, spacy_nlp, "off2", supersede=False)
    _write_preference(off, "User prefers Python.", T1)
    _write_preference(off, "User prefers Rust.", T2)
    metas = off.collection("u").get(include=["metadatas"])["metadatas"]
    assert all("valid_until" not in (m or {}) for m in metas)
    # valid_from is still stamped regardless — schema is always present.
    assert all((m or {}).get("valid_from") for m in metas)


def test_missing_subject_or_predicate_is_never_superseded(tmp_path, spacy_nlp):
    """A fact the extractor didn't tag with subject/predicate (model
    non-compliance, or a store using a custom_instructions path that skips
    it) must never be silently closed or silently close another fact."""
    on = _engine(tmp_path, spacy_nlp, "notag", supersede=True)
    on._test_llm.queue.append(json.dumps({"memory": [{"id": "0", "text": "User prefers Python."}]}))
    on.add("u", "User prefers Python.", timestamp=T1)
    _write_preference(on, "User prefers Rust.", T2)
    texts = {p["text"] for p in on.seeds("u", "prefers", 10)}
    assert texts == {"User prefers Python.", "User prefers Rust."}   # untagged fact never closed


def test_supersede_failure_does_not_lose_the_new_fact(tmp_path, spacy_nlp, monkeypatch):
    on = _engine(tmp_path, spacy_nlp, "fail", supersede=True)
    monkeypatch.setattr(
        FactEngine, "_supersede_facts",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = _write_preference(on, "User prefers Python.", T1)
    assert out  # add() still returns the fact despite the supersede step blowing up


# --- current-truth filter (unit-level) --------------------------------------------------


def test_is_current_no_versioning_info_is_always_current():
    assert _is_current({}, None) is True
    assert _is_current(None, None) is True


def test_is_current_open_window_is_current_forever():
    assert _is_current({"valid_from": T1}, None) is True
    assert _is_current({"valid_from": T1}, "2099-01-01T00:00:00+00:00") is True


def test_is_current_closed_window_hides_after_valid_until():
    meta = {"valid_from": T1, "valid_until": T2}
    assert _is_current(meta, BETWEEN) is True     # still open at BETWEEN
    assert _is_current(meta, T2) is False         # closed exactly at valid_until
    assert _is_current(meta, "2024-01-01T00:00:00+00:00") is False


def test_is_current_before_valid_from_is_not_yet_current():
    assert _is_current({"valid_from": T2}, T1) is False


def test_is_current_fails_open_on_unparseable_timestamps():
    assert _is_current({"valid_from": "not-a-date"}, None) is True
    assert _is_current({"valid_from": T1, "valid_until": "not-a-date"}, "2099-01-01T00:00:00+00:00") is True


def test_parse_iso_handles_missing_and_bad_input():
    assert _parse_iso(None) is None
    assert _parse_iso("") is None
    assert _parse_iso("garbage") is None
    assert _parse_iso(T1) is not None


# --- through Memory ----------------------------------------------------------------------


def test_memory_supersede_option(tmp_path, spacy_nlp):
    def make(supersede, name):
        llm = ScriptedLLM()
        return Memory(str(tmp_path / name), backend="facts", llm=llm, embedder=FakeEmbedder(),
                      entity_extractor=spacy_nlp, supersede=supersede), llm

    m, llm = make(True, "on")
    assert m._facts._supersede is True
    off, _ = make(False, "off")
    assert off._facts._supersede is False

    cfg = Memory.from_config({"data_dir": str(tmp_path / "cfg"), "backend": "facts",
                              "entity_extractor": spacy_nlp, "embedder": FakeEmbedder(),
                              "llm": ScriptedLLM(), "supersede": False})
    assert cfg._facts._supersede is False


def test_memory_search_as_of_ignored_for_raw_backend(tmp_path, spacy_nlp):
    m = Memory(str(tmp_path / "raw"), backend="raw", entity_extractor=spacy_nlp)
    m.add("The sky is blue.", user_id="u")
    # Raw pages have no validity window — as_of must not error or filter anything.
    assert len(m.search("sky", user_id="u", as_of="2020-01-01T00:00:00+00:00")) == 1


def test_memory_search_rejects_non_string_as_of(tmp_path, spacy_nlp):
    m = Memory(str(tmp_path / "raw2"), backend="raw", entity_extractor=spacy_nlp)
    m.add("The sky is blue.", user_id="u")
    with pytest.raises(Exception):
        m.search("sky", user_id="u", as_of=123)


def test_memory_answer_as_of_reaches_history(tmp_path, spacy_nlp):
    """End-to-end through Memory.answer()'s as_of param, not just FactEngine.seeds()."""
    llm = ScriptedLLM()
    m = Memory(str(tmp_path / "answer"), backend="facts", llm=llm, embedder=FakeEmbedder(),
               entity_extractor=spacy_nlp)
    llm.queue.append(json.dumps({
        "memory": [{"id": "0", "text": "User prefers Python.", "subject": "User", "predicate": "prefers"}]
    }))
    m.add("User prefers Python.", user_id="u", timestamp=T1)
    llm.queue.append(json.dumps({
        "memory": [{"id": "0", "text": "User prefers Rust.", "subject": "User", "predicate": "prefers"}]
    }))
    m.add("User prefers Rust.", user_id="u", timestamp=T2)

    seen_prompts = []

    def capture(prompt, model):
        seen_prompts.append(prompt)
        return json.dumps({"answer": "ok", "source_page_ids": []})

    m.answer("What does the user prefer?", user_id="u", chat=capture)
    assert "Rust" in seen_prompts[-1] and "Python" not in seen_prompts[-1]   # default: current only

    m.answer("What does the user prefer?", user_id="u", chat=capture, as_of=BETWEEN)
    assert "Python" in seen_prompts[-1] and "Rust" not in seen_prompts[-1]   # as_of: history
