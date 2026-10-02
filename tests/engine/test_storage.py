"""Ported verbatim from the Cortex backend suite (test_storage.py) — proves the copied engine
behaves identically. Only the imports changed."""

import pytest

from cortexlayer._engine import storage


def _fresh_collection(tmp_path):
    client = storage.get_client(persist_dir=str(tmp_path / "chroma"))
    return storage.get_collection(client)


def test_insert_get_round_trip(tmp_path):
    col = _fresh_collection(tmp_path)
    pid = storage.insert_page(
        col, "Christopher Nolan directed Inception.", ["Christopher Nolan", "Inception"]
    )
    page = storage.get_page(col, pid)
    assert page is not None
    assert page["text"] == "Christopher Nolan directed Inception."
    assert sorted(page["entities"]) == ["Christopher Nolan", "Inception"]
    assert page["links"] == []
    assert page["created_at"]


def test_insert_get_round_trip_with_tags(tmp_path):
    col = _fresh_collection(tmp_path)
    pid = storage.insert_page(
        col, "Note.", ["Note"], tags={"agent_id": "planner", "n": 3, "ok": True}
    )
    page = storage.get_page(col, pid)
    assert page["tags"] == {"agent_id": "planner", "n": 3, "ok": True}


def test_tags_default_to_empty_dict_when_omitted_or_empty(tmp_path):
    col = _fresh_collection(tmp_path)
    a = storage.insert_page(col, "No tags.", [])
    b = storage.insert_page(col, "Empty tags.", [], tags={})
    assert storage.get_page(col, a)["tags"] == {}
    assert storage.get_page(col, b)["tags"] == {}


def test_text_update_preserves_tags(tmp_path):
    col = _fresh_collection(tmp_path)
    pid = storage.insert_page(col, "Old text.", ["Entity"], tags={"source": "import"})
    storage.update_page_text(col, pid, "New text.")
    assert storage.get_page(col, pid)["tags"] == {"source": "import"}


def test_links_update_and_append(tmp_path):
    col = _fresh_collection(tmp_path)
    a = storage.insert_page(col, "Page A about Nolan.", ["Nolan"])
    b = storage.insert_page(col, "Page B about Nolan too.", ["Nolan"])
    storage.update_links(col, a, [b])
    assert storage.get_page(col, a)["links"] == [b]
    storage.append_link(col, a, b)  # idempotent — no duplicate
    assert storage.get_page(col, a)["links"] == [b]


def test_text_update_preserves_links_and_entities(tmp_path):
    col = _fresh_collection(tmp_path)
    pid = storage.insert_page(col, "Old text.", ["Entity"])
    storage.update_links(col, pid, ["other-id"])
    storage.update_page_text(col, pid, "New text.")
    page = storage.get_page(col, pid)
    assert page["text"] == "New text."
    assert page["entities"] == ["Entity"]
    assert page["links"] == ["other-id"]


def test_query_and_delete(tmp_path):
    col = _fresh_collection(tmp_path)
    pid = storage.insert_page(
        col, "The director of Inception is Christopher Nolan.", ["Inception"]
    )
    assert storage.count(col) == 1
    hits = storage.query(col, "Who directed Inception?", n_results=1)
    assert hits and hits[0]["id"] == pid
    storage.delete_page(col, pid)
    assert storage.count(col) == 0
    assert storage.get_page(col, pid) is None


def test_list_pages_pagination(tmp_path):
    col = _fresh_collection(tmp_path)
    for i in range(3):
        storage.insert_page(col, f"Fact number {i}.", [f"entity-{i}"])
    assert len(storage.list_pages(col, limit=2, offset=0)) == 2
    assert len(storage.list_pages(col, limit=2, offset=2)) == 1


def _fresh_client(tmp_path):
    return storage.get_client(persist_dir=str(tmp_path / "chroma"))


def test_default_user_keeps_legacy_collection_name(tmp_path):
    client = _fresh_client(tmp_path)
    assert storage.get_collection(client).name == storage.COLLECTION_NAME
    assert storage.get_collection(client, "default").name == "cortex_pages"


def test_per_user_collections_are_isolated(tmp_path):
    client = _fresh_client(tmp_path)
    alice = storage.get_collection(client, "alice")
    bob = storage.get_collection(client, "bob")
    default = storage.get_collection(client, "default")
    assert alice.name == "cortex_pages__alice"
    assert bob.name == "cortex_pages__bob"

    leaked = storage.insert_page(alice, "Alice secret.", ["Alice"])
    # Leaked cross-user ID fails closed on every read path.
    assert storage.get_page(bob, leaked) is None
    assert storage.get_page(default, leaked) is None
    assert storage.query(bob, "Alice secret", n_results=5) == []
    assert storage.count(bob) == 0
    assert storage.count(default) == 0
    # Same handle still reads its own data.
    assert storage.get_page(alice, leaked)["text"] == "Alice secret."

    # Same explicit page_id in two collections denotes different pages.
    storage.insert_page(bob, "Bob page.", ["Bob"], page_id="shared-id")
    storage.insert_page(alice, "Alice page.", ["Alice"], page_id="shared-id")
    assert storage.get_page(bob, "shared-id")["text"] == "Bob page."
    assert storage.get_page(alice, "shared-id")["text"] == "Alice page."


def test_user_id_validation_rejects_bad_input(tmp_path):
    client = _fresh_client(tmp_path)
    for bad in ["", "has space", "a/b", "..", "trail-", "semi;colon", "x" * 65, None]:
        with pytest.raises((ValueError, TypeError)):
            storage.get_collection(client, bad)


# --- org collections (task 0107, arch §9.2) ---

ORG_A = "11111111111111111111111111111111"
ORG_B = "22222222222222222222222222222222"


def test_collection_name_for_org(tmp_path):
    assert storage.collection_name_for_org(ORG_A) == f"cortex_org_pages__{ORG_A}"


def test_org_id_validation_rejects_bad_input(tmp_path):
    client = _fresh_client(tmp_path)
    for bad in ["", "not-hex!!", "x" * 31, "x" * 33, "ABCDEF00" * 4, None, 123]:
        with pytest.raises((ValueError, TypeError)):
            storage.get_org_collection(client, bad)


def test_org_collections_are_isolated_from_each_other(tmp_path):
    client = _fresh_client(tmp_path)
    org_a = storage.get_org_collection(client, ORG_A)
    org_b = storage.get_org_collection(client, ORG_B)
    assert org_a.name == f"cortex_org_pages__{ORG_A}"
    assert org_b.name == f"cortex_org_pages__{ORG_B}"

    leaked = storage.insert_page(org_a, "Org A secret.", ["A"])
    assert storage.get_page(org_b, leaked) is None
    assert storage.query(org_b, "Org A secret", n_results=5) == []
    assert storage.count(org_b) == 0
    assert storage.get_page(org_a, leaked)["text"] == "Org A secret."


def test_org_and_user_collections_never_collide(tmp_path):
    """An org_id and a user_id chosen to collide under the OLD shared-prefix
    scheme (same raw id string) must still land in disjoint collections,
    because the org prefix shares no characters with the user prefix."""
    client = _fresh_client(tmp_path)
    shared_id = ORG_A  # also a legal user_id under _USER_ID_RE
    user_col = storage.get_collection(client, shared_id)
    org_col = storage.get_org_collection(client, shared_id)
    assert user_col.name != org_col.name
    assert user_col.name == f"cortex_pages__{shared_id}"
    assert org_col.name == f"cortex_org_pages__{shared_id}"

    leaked = storage.insert_page(user_col, "User secret.", ["U"])
    # Cross-namespace get_page fails closed exactly like the per-user case.
    assert storage.get_page(org_col, leaked) is None
    assert storage.count(org_col) == 0


def test_adversarial_org_user_id_pair_cannot_collide(tmp_path):
    """Even an org_id built to literally equal an existing user's full
    collection name can't produce the same Chroma collection name, since the
    org prefix is prepended on top of it."""
    client = _fresh_client(tmp_path)
    user_col = storage.get_collection(client, "alice")
    assert user_col.name == "cortex_pages__alice"
    # "cortex_pages__alice" itself is not a valid org_id (not 32 hex chars),
    # so no org can even be named to attempt the collision in the first
    # place — validation rejects it before a collection is ever opened.
    with pytest.raises(ValueError):
        storage.get_org_collection(client, "cortex_pages__alice")
