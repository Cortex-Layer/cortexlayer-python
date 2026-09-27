"""Chroma storage wrapper — the single source of truth for Cortex pages.

Schema per entry:
    id          string    unique page ID
    document    string    raw page text
    embedding   vector    generated at ingestion (Chroma default embedding function)
    entities    metadata  list of extracted entities/key terms
    links       metadata  list of linked page IDs (populated by the linking pass)
    created_at  metadata  ISO-8601 timestamp
    tags        metadata  optional dict of page-level provenance (task 0090), e.g.
                          agent_id/source/session_id — JSON-encoded, decoded back to
                          a dict by this module. Deliberately NOT an isolation
                          mechanism: every query still spans the whole per-user
                          collection by default; a tag just rides along as metadata
                          you can look at (or opt-in filter on) later. Per-user
                          isolation is (and stays) via separate Chroma collections
                          (task 0026) — no default filter here means no
                          forgotten-filter failure mode either.

Implementation note: Chroma metadata values must be scalars (str/int/float/bool),
so ``entities``, ``links`` and ``tags`` are stored as JSON-encoded strings and
transparently decoded by this module. Callers always work with plain Python lists
/ dicts.

Multi-user isolation: the raw backend keeps one Chroma collection per user in
the shared persist dir. ``get_collection`` takes a
``user_id`` and returns that user's scoped handle; every other function in this
module operates on a passed-in handle, so scoping is by construction — there is
no unscoped full-store scan anywhere. ``default`` keeps the legacy
``cortex_pages`` name (zero migration); other users get
``cortex_pages__<user>``. Page IDs are uuid4 hex (globally unique), so a leaked
cross-user ID simply does not exist in the caller's collection: ``get_page``
returns None (fails closed).
"""

from __future__ import annotations

import datetime
import json
import re
import uuid

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

COLLECTION_NAME = "cortex_pages"
DEFAULT_PERSIST_DIR = "./data/chroma"

DEFAULT_USER_ID = "default"
USER_COLLECTION_PREFIX = "cortex_pages__"
MAX_USER_ID_LENGTH = 64
# §8.4 charset, tightened to require leading/trailing alphanumerics: Chroma
# collection names must start AND end with [a-zA-Z0-9], so a user_id ending in
# "-" or "_" would otherwise fail at get_or_create time. Flagged for Miguel
# in the 0027 notes (0026 proposed bare ^[A-Za-z0-9_-]{1,64}$).
_USER_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?")


def validate_user_id(user_id: str) -> str:
    """Validate-never-mangle per §8.4: return ``user_id`` or raise ValueError.

    Invalid input is an error, never a silent fallback/mangling (mangling can
    map two distinct inputs onto one store → cross-user collision).
    """
    if (
        not isinstance(user_id, str)
        or not user_id
        or len(user_id) > MAX_USER_ID_LENGTH
        or _USER_ID_RE.fullmatch(user_id) is None
    ):
        raise ValueError(
            "invalid user_id: must match "
            f"[A-Za-z0-9][A-Za-z0-9_-]{{0,62}}[A-Za-z0-9]? (1-{MAX_USER_ID_LENGTH} chars); "
            f"got {user_id!r}"
        )
    return user_id


def collection_name_for_user(user_id: str = DEFAULT_USER_ID) -> str:
    """Map a validated user_id to its Chroma collection name.

    ``default`` keeps the legacy ``cortex_pages`` name (zero migration);
    every other user gets ``cortex_pages__<user>``.
    """
    validate_user_id(user_id)
    if user_id == DEFAULT_USER_ID:
        return COLLECTION_NAME
    return f"{USER_COLLECTION_PREFIX}{user_id}"


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_page_id() -> str:
    return uuid.uuid4().hex


def _encode_list(values: list[str]) -> str:
    return json.dumps(list(values))


def _decode_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in decoded] if isinstance(decoded, list) else []


def _encode_tags(tags: dict | None) -> str | None:
    """``None``/``{}`` -> ``None`` (no key written at all — identical to a
    store that predates task 0090, no schema migration needed)."""
    return json.dumps(tags) if tags else None


def _decode_tags(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.PersistentClient:
    """Open (creating if needed) the persistent embedded Chroma client.

    Chroma's anonymous product telemetry is switched off: a library should not
    phone home by default.
    """
    return chromadb.PersistentClient(
        path=persist_dir, settings=Settings(anonymized_telemetry=False)
    )


def get_collection(
    client: chromadb.PersistentClient,
    user_id: str = DEFAULT_USER_ID,
    name: str | None = None,
    embedding_function=None,
) -> Collection:
    """Get-or-create the calling user's pages collection (user-scoped handle).

    ``user_id`` selects the collection per §8: ``default`` → legacy
    ``cortex_pages`` (zero migration), others → ``cortex_pages__<user>``.
    ``name`` is a deprecated escape hatch kept for old call sites/tests; if
    given it wins, otherwise the name derives from ``user_id``.

    ``embedding_function`` defaults to None, which means Chroma's built-in
    default (ONNX MiniLM-L6-v2). Pass an
    explicit function here only to override it (e.g. in tests).
    """
    resolved = name if name is not None else collection_name_for_user(user_id)
    kwargs: dict = {"name": resolved}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client.get_or_create_collection(**kwargs)


def _page_from_result(
    page_id: str, document: str | None, metadata: dict | None
) -> dict:
    metadata = metadata or {}
    return {
        "id": page_id,
        "text": document or "",
        "entities": _decode_list(metadata.get("entities")),
        "links": _decode_list(metadata.get("links")),
        "created_at": metadata.get("created_at", ""),
        "tags": _decode_tags(metadata.get("tags")),
    }


def insert_page(
    collection: Collection,
    text: str,
    entities: list[str],
    page_id: str | None = None,
    created_at: str | None = None,
    embedding: list[float] | None = None,
    extra: dict | None = None,
    tags: dict | None = None,
) -> str:
    """Insert one page into the caller's user-scoped collection. Returns the page ID.

    IDs are uuid4 hex (globally unique); the same hex in two users'
    collections denotes different pages, and a leaked cross-user ID fails
    closed on ``get_page`` (returns None — the ID is absent in your store).
    ``links`` starts empty until the per-user linking pass runs.

    ``tags`` (task 0090) is optional page-level provenance metadata (e.g.
    ``{"agent_id": "planner"}``) — kept separate from ``extra``, which is a
    raw passthrough callers use for their own scalar metadata (e.g. the facts
    engine's ``hash``/``updated_at``/``attributed_to``), so the two can never
    clobber each other. Reserved keys (``entities``/``links``/``created_at``/
    ``tags``) always win over anything of the same name in ``extra``."""
    pid = page_id or new_page_id()
    kwargs: dict = {}
    if embedding is not None:
        # Caller-supplied vector (fact memory embeds with its own model);
        # without it Chroma embeds the document with its default function.
        kwargs["embeddings"] = [embedding]
    encoded_tags = _encode_tags(tags)
    collection.add(
        ids=[pid],
        documents=[text],
        metadatas=[
            {
                **(extra or {}),
                "entities": _encode_list(entities),
                "links": _encode_list([]),
                "created_at": created_at or utc_now_iso(),
                **({"tags": encoded_tags} if encoded_tags else {}),
            }
        ],
        **kwargs,
    )
    return pid


def get_page(collection: Collection, page_id: str) -> dict | None:
    """Fetch one page by ID from the caller's user-scoped collection.

    Returns None if the ID does not exist *in this user's collection* —
    including a leaked ID from another user (fails closed, no ownership
    re-check needed since collections are disjoint)."""
    result = collection.get(ids=[page_id])
    if not result["ids"]:
        return None
    return _page_from_result(
        result["ids"][0],
        result["documents"][0] if result["documents"] else None,
        result["metadatas"][0] if result["metadatas"] else None,
    )


def list_pages(
    collection: Collection, limit: int = 100, offset: int = 0
) -> list[dict]:
    """Paginated scan of the caller's *user-scoped* collection.

    Used by the per-user linking pass (arch §2.3/§8: per-user passes, never
    global). ``collection`` is always a user-scoped handle from
    ``get_collection(client, user_id)``, so there is no unscoped
    full-store scan — isolation is by collection handle, not by filter.
    """
    result = collection.get(limit=limit, offset=offset)
    return [
        _page_from_result(pid, doc, meta)
        for pid, doc, meta in zip(
            result["ids"],
            result["documents"] or [],
            result["metadatas"] or [],
        )
    ]


def update_page_text(
    collection: Collection,
    page_id: str,
    text: str,
    entities: list[str] | None = None,
) -> None:
    """Replace a page's text (re-embeds via the collection's embedding function).

    If ``entities`` is given, replace those too; otherwise keep existing ones.
    ``links`` are left untouched.
    """
    metadata_update: dict = {}
    if entities is not None:
        metadata_update["entities"] = _encode_list(entities)
    collection.update(
        ids=[page_id],
        documents=[text],
        **({"metadatas": [metadata_update]} if metadata_update else {}),
    )


def update_links(collection: Collection, page_id: str, links: list[str]) -> None:
    """Overwrite a page's ``links`` metadata field."""
    collection.update(
        ids=[page_id],
        metadatas=[{"links": _encode_list(links)}],
    )


def append_link(collection: Collection, page_id: str, target_id: str) -> None:
    """Add one link to a page (no-op if already present)."""
    page = get_page(collection, page_id)
    if page is None:
        raise KeyError(f"page not found: {page_id}")
    if target_id not in page["links"]:
        update_links(collection, page_id, [*page["links"], target_id])


def delete_page(collection: Collection, page_id: str) -> None:
    collection.delete(ids=[page_id])


def count(collection: Collection) -> int:
    return collection.count()


def query(
    collection: Collection, query_text: str, n_results: int = 5
) -> list[dict]:
    """Vector similarity search over the caller's user-scoped collection.

    Returns page dicts plus ``score`` (raw Chroma distance, lower = closer).
    Never spans collections."""
    result = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )
    ids = result["ids"][0] if result["ids"] else []
    docs = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    distances = result.get("distances", [[]])[0] if result.get("distances") else []
    pages = [
        _page_from_result(pid, doc, meta)
        for pid, doc, meta in zip(ids, docs, metas)
    ]
    for page, dist in zip(pages, distances):
        page["score"] = dist
    for page in pages[len(distances):]:
        page["score"] = 0.0
    return pages
