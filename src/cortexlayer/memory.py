"""``Memory`` — the embedded, in-process Cortex engine (no server).

>>> from cortexlayer import Memory                                   # doctest: +SKIP
>>> m = Memory()                                                     # doctest: +SKIP
>>> m.add("I moved to Lisbon in March", user_id="alice")             # doctest: +SKIP
>>> m.search("where does alice live?", user_id="alice")              # doctest: +SKIP

Same method names and result types as :class:`~cortexlayer.CortexClient`, so
switching between local and hosted is a change of constructor. Memories live in
an embedded Chroma store under ``data_dir``; each ``user_id`` gets its own
collection, so users can never see each other's pages.

Needs ``pip install "cortexlayer[local]"``. Importing this module does not —
the engine is only loaded when a ``Memory`` is constructed.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional

from ._validate import need_int, need_str, need_tags
from .errors import CortexConfigError, InvalidRequestError, LocalDependencyError, NotFoundError
from .types import Answer, AddResult, Page, PageList, SearchResult

DEFAULT_USER_ID = "default"
DATA_DIR_ENV = "CORTEXLAYER_DATA_DIR"
MAX_SEARCH_LIMIT = 50  # raised from 20 alongside DEFAULT_K (task 0101)
MAX_LIST_LIMIT = 500
BACKENDS = ("raw", "facts")
_CONFIG_KEYS = {
    "backend", "data_dir", "entity_extractor", "spacy_model",
    "default_user_id", "auto_relink", "llm", "embedder", "custom_instructions",
    "observation_date_from_timestamp", "keyword_scoring", "supersede",
}


def _default_data_dir() -> str:
    return os.environ.get(DATA_DIR_ENV) or os.path.join(
        os.path.expanduser("~"), ".cortexlayer"
    )


def _load_engine(backend: str = "raw") -> Any:
    """Import the engine, or explain how to install it."""
    try:
        from ._engine import compression, ingestion, linking, nlp, retrieval, shaping, storage
        if backend == "facts":
            from ._engine.facts import backends as fact_backends
            from ._engine.facts import engine as fact_engine
            from ._engine.facts import scoring as fact_scoring
    except ImportError as e:
        raise LocalDependencyError(
            "The embedded engine needs extra packages. Run: "
            'pip install "cortexlayer[local]"  '
            f"(missing: {getattr(e, 'name', None) or e})."
        ) from e

    class _Engine:
        pass

    eng = _Engine()
    for mod in (compression, ingestion, linking, nlp, retrieval, shaping, storage):
        setattr(eng, mod.__name__.rsplit(".", 1)[-1], mod)
    if backend == "facts":
        eng.fact_backends, eng.fact_engine, eng.fact_scoring = (
            fact_backends, fact_engine, fact_scoring,
        )
    return eng


class Memory:
    """Embedded memory store.

    Args:
        data_dir: where the embedded store lives (default ``~/.cortexlayer``,
            or ``$CORTEXLAYER_DATA_DIR``). It is created on first use.
        entity_extractor: ``"auto"`` (default: spaCy if available, else a
            simpler regex extractor with a one-time warning), ``"spacy"``,
            ``"regex"``, or your own object with ``entities(text)`` and
            ``sentences(text)`` methods. Entities drive linking, so a weaker
            extractor means fewer links.
        spacy_model: spaCy model name (default ``en_core_web_sm``).
        default_user_id: used when a call omits ``user_id``.
        auto_relink: re-run the linking pass after every ``add``. Off by
            default — linking is a batch pass over all pages, so prefer
            calling :meth:`relink` after adding several memories.
        backend: ``"raw"`` (default) stores your text as small pages — adding
            never needs an LLM. ``"facts"`` asks an LLM to distil each ``add``
            into self-contained facts (Mem0-style), stores those, and boosts
            search by shared entities; it needs an LLM (``llm=``, Ollama by
            default) and an embedder (``embedder=``, Chroma's built-in by
            default).
        llm: (``"facts"``) ``None`` = Ollama at ``$OLLAMA_HOST``; a dict such
            as ``{"model": "qwen3.5:9b", "host": "..."}``; a callable
            ``fn(system, user) -> str``; or any object with ``generate()``.
        embedder: (``"facts"``) ``None``/``"chroma"`` (default), ``"ollama"``,
            a dict like ``{"provider": "ollama", "model": "qwen3-embedding:8b"}``,
            or an object with ``embed_batch(texts, action)`` and ``name``. A
            store remembers its embedder and refuses a different one.
        custom_instructions: (``"facts"``) extra extraction rules appended to
            the prompt.
        observation_date_from_timestamp: (``"facts"``) pass ``add(timestamp=)``
            to the extractor as the conversation's Observation Date, so
            "last week" resolves against that date. Off by default, which
            matches Mem0 (it resolves relative dates against today).
        keyword_scoring: (``"facts"``) fuse a BM25 keyword score into search so
            a fact that literally contains a query word is not outranked by
            generic facts with slightly closer vectors. ``True`` (weight 1, as
            Mem0's design; the default) or a float weight; ``False`` is plain
            semantic + entity scoring, identical to Mem0 on a Chroma store.
        supersede: (``"facts"``, task 0094) when a new fact shares a
            (subject, predicate) with an existing one still in effect (e.g.
            "User prefers X" later "User prefers Y"), close the old one's
            validity window instead of leaving both — :meth:`search`/
            :meth:`answer` then default to current-truth-only (pass
            ``as_of`` to see history). ``True`` (default) also asks the
            extractor for the subject/predicate pair it needs; ``False`` is
            pure append-only, identical to pre-0094 Cortex (and to Mem0,
            which has no equivalent).
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        *,
        entity_extractor: Any = "auto",
        spacy_model: str = "en_core_web_sm",
        default_user_id: str = DEFAULT_USER_ID,
        auto_relink: bool = False,
        backend: str = "raw",
        llm: Any = None,
        embedder: Any = None,
        custom_instructions: Optional[str] = None,
        observation_date_from_timestamp: bool = False,
        keyword_scoring: Any = True,
        supersede: bool = True,
    ) -> None:
        if backend not in BACKENDS:
            raise CortexConfigError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self._backend = backend
        self._e = _load_engine(backend)
        self._data_dir = os.path.abspath(data_dir or _default_data_dir())
        self._default_user = self._check_uid(default_user_id)
        self._auto_relink = bool(auto_relink)
        self._nlp = self._e.nlp.resolve_nlp(entity_extractor, spacy_model=spacy_model)
        self._client = self._e.storage.get_client(self._data_dir)
        self._collections: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._facts = None
        if backend == "facts":
            self._facts = self._e.fact_engine.FactEngine(
                self._client, self._data_dir,
                self._e.fact_backends.resolve_llm(llm),
                self._e.fact_backends.resolve_embedder(embedder),
                self._nlp, custom_instructions=custom_instructions,
                observation_date_from_timestamp=observation_date_from_timestamp,
                keyword_weight=float(keyword_scoring),
                supersede=supersede,
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Memory":
        """Build from a plain dict (keys: ``data_dir``, ``entity_extractor``,
        ``spacy_model``, ``default_user_id``, ``auto_relink``, ``backend``,
        ``llm``, ``embedder``, ``custom_instructions``).
        """
        if not isinstance(config, dict):
            raise CortexConfigError("config must be a dict")
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise CortexConfigError(f"unknown config keys: {', '.join(sorted(unknown))}")
        backend = config.get("backend", "raw")
        if backend not in BACKENDS:
            raise CortexConfigError(
                f"backend {backend!r} is not available; choose one of {BACKENDS}"
            )
        return cls(**config)

    def __repr__(self) -> str:
        return f"Memory(data_dir={self._data_dir!r}, backend={self._backend!r})"

    # --- internals ---

    def _check_uid(self, uid: str) -> str:
        try:
            return self._e.storage.validate_user_id(uid)
        except ValueError as e:
            raise InvalidRequestError(str(e)) from e

    def _check_org(self, org_id: str) -> str:
        try:
            return self._e.storage.validate_org_id(org_id)
        except ValueError as e:
            raise InvalidRequestError(str(e)) from e

    def _uid(self, user_id: Optional[str]) -> str:
        """``None`` means the instance's default user."""
        return self._check_uid(self._default_user if user_id is None else user_id)

    def _col(self, user_id: Optional[str]):
        uid = self._uid(user_id)
        with self._lock:
            col = self._collections.get(uid)
            if col is None:
                col = self._collections[uid] = (
                    self._facts.collection(uid) if self._facts is not None
                    else self._e.storage.get_collection(self._client, uid)
                )
            return col

    def collection(self, user_id: Optional[str] = None) -> Any:
        """The user's underlying Chroma collection (raw pages, or extracted
        facts for ``backend="facts"``; both share one page schema).

        For embedders that serve or browse the store directly, e.g. the Cortex
        server. Treat it as read-only: writing through it bypasses embedding,
        entity extraction and the facts pipeline."""
        return self._col(user_id)

    def _org_col(self, org_id: str):
        org_id = self._check_org(org_id)
        key = ("org", org_id)
        with self._lock:
            col = self._collections.get(key)
            if col is None:
                col = self._collections[key] = (
                    self._facts.collection_for_org(org_id) if self._facts is not None
                    else self._e.storage.get_org_collection(self._client, org_id)
                )
            return col

    def org_collection(self, org_id: str) -> Any:
        """An org pool's underlying Chroma collection (task 0108, arch §9.2) —
        parallel to :meth:`collection`, same read-only caveat."""
        return self._org_col(org_id)

    def _passages(
        self, user_id, col, query: str, limit: int, expand_links: bool,
        as_of: Optional[str] = None,
    ) -> List[dict]:
        """Seeds (+ optional link expansion) as ``{page_id, text, score, via, ...}``."""
        if self._facts is not None:
            seeds = self._facts.seeds(self._uid(user_id), query, limit, as_of=as_of)
        else:
            seeds = self._e.storage.query(col, query, n_results=limit)
        if expand_links:
            if self._facts is not None:
                # Facts collections hold vectors from the configured embedder,
                # not Chroma's default text function — rank with a matching
                # query vector (task 0077).
                passages = self._e.retrieval.expand_links(
                    col, seeds, query_embedding=self._facts.embed_query(query)
                )
                # expand_links's score is always a raw Chroma distance
                # (lower=better, unbounded) — right for the raw backend, but
                # seeds() above already fused this backend's score into a
                # [0, 1] similarity where *higher* is better (semantic +
                # keyword + entity boost). Left unconverted, link rows landed
                # on a different scale and direction than seeds (found
                # 2026-09-26 testing the Playground: linked results looked
                # unsorted relative to direct ones). Put link rows on the
                # same scale so score is comparable across the whole result.
                for p in passages:
                    if p["via"] == "link":
                        p["score"] = self._e.fact_scoring.distance_to_score(p["score"])
                return passages
            return self._e.retrieval.expand_links(col, seeds, query=query)
        return [
            {"page_id": p["id"], "text": p["text"], "score": p.get("score", 0.0),
             "via": "direct", "tags": p.get("tags")}
            for p in seeds
        ]

    def _all_pages(self, col) -> List[dict]:
        return self._e.shaping.scan_all(col, self._e.storage.list_pages)

    # --- memory ---

    def add(
        self,
        text: str,
        *,
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        org_id: Optional[str] = None,
    ) -> AddResult:
        """Save ``text`` as memory. Long text is chunked into several small
        pages (one fact each works best). ``timestamp`` (e.g. ``"8 May, 2023"``)
        is stored as the date.

        ``tags`` (task 0090) is optional provenance metadata — a flat dict of
        str/int/float/bool values, e.g. ``{"agent_id": "planner", "session_id":
        "abc123"}`` — stamped onto every page this call creates and returned
        as-is on :meth:`get`/:meth:`get_all`/:meth:`search`. It is deliberately
        **not** a scoping or isolation mechanism: ``search``/``answer``/
        ``get_all`` keep spanning this user's whole collection by default,
        exactly as without ``tags``. It exists so several agents/sessions can
        share one person's memory (the "hive mind") while still recording,
        for later inspection, which one wrote a given fact.

        ``org_id`` (task 0108, arch §9.2/§9.3): write into that org's shared
        pool instead of ``user_id``'s personal one (same extraction/embedding
        pipeline either way — ``backend="facts"`` runs the same LLM
        distillation for an org pool as for a personal one). Every page this
        call creates is additionally stamped ``tags["author"] = user_id``
        (resolved the same way ``user_id`` always is — explicit param, else
        this instance's default) — the author tag task 0110's supersede
        logic and 0105 §9.1's admin-visibility reasoning both depend on.
        Validating that ``user_id`` actually belongs to ``org_id`` is the
        caller's job (e.g. the server checks ``org_members`` before this
        call) — this method only validates ``org_id``'s own shape.

        Does not relink unless ``auto_relink`` is on."""
        need_str(text, "text")
        if timestamp is not None:
            need_str(timestamp, "timestamp")
        tags = need_tags(tags, "tags")
        if org_id is not None:
            org_id = self._check_org(org_id)
            tags = {**(tags or {}), "author": self._uid(user_id)}
            col = self._org_col(org_id)
        else:
            col = self._col(user_id)
        if self._facts is not None:
            ids = [
                f["id"] for f in self._facts.add(
                    self._uid(user_id), text, timestamp, tags, org_id=org_id
                )
            ]
        else:
            ids = self._e.ingestion.add_text(col, text, self._nlp, timestamp=timestamp, tags=tags)
        if self._auto_relink and ids:
            self._e.linking.run_linking_pass(col)
        return AddResult(page_ids=ids)

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,  # raised from 4 (task 0101) — see MAX_SEARCH_LIMIT
        expand_links: bool = True,
        as_of: Optional[str] = None,
    ) -> List[SearchResult]:
        """Semantic search. With ``expand_links`` (default) related pages are
        pulled in via links — those results have ``via == "link"``. An empty
        store returns ``[]``.

        ``as_of`` (``backend="facts"`` only, task 0094): retrieve fact
        history as it stood at that ISO timestamp instead of the current
        truth — a fact superseded before ``as_of`` stays hidden, one
        superseded after it (or never) is included. Ignored for
        ``backend="raw"`` (pages there have no validity window)."""
        need_str(query, "query")
        need_int(limit, "limit", 1, MAX_SEARCH_LIMIT)
        if not isinstance(expand_links, bool):
            raise InvalidRequestError("expand_links must be True or False")
        if as_of is not None:
            need_str(as_of, "as_of")
        col = self._col(user_id)
        if col.count() == 0:
            return []
        passages = self._passages(user_id, col, query, limit, expand_links, as_of=as_of)
        return [self._e.shaping.search_result(p) for p in passages]

    def _seeds(
        self, col, query: str, limit: int, *,
        user_id: Optional[str] = None, org_id: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> List[dict]:
        """Raw seed search against one already-resolved collection (personal
        or org) — this backend's own scoring, pre-link-expansion. Shared by
        :meth:`search_multi_pool` (task 0109) so facts-backend seed search
        (entity boosts, current-truth filtering) works identically for an
        org pool as for a personal one."""
        if self._facts is not None:
            return self._facts.seeds(user_id, query, limit, as_of=as_of, org_id=org_id)
        return self._e.storage.query(col, query, n_results=limit)

    def search_multi_pool(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        org_id: str,
        limit: int = 50,
    ) -> List[SearchResult]:
        """Task 0109 / arch §9.4: fan out seed search + link-expansion across
        exactly two pools — the caller's personal pool and one org pool —
        and unify them with 0093's RRF fusion instead of running each pool
        independently. Every result's ``tags`` carries ``"pool"``
        (``"personal"`` or ``"org:<org_id>"``) in addition to whatever was
        already there (e.g. 0108's ``"author"``), so provenance survives the
        merge.

        This method does not check organization membership — the caller
        (the server) validates ``org_id`` against ``org_members`` before
        calling it, same division of labor as :meth:`add`. It only ever
        touches these two pools, never every org the caller belongs to.

        An empty combined pool (both collections empty, or org_id's
        collection does not exist yet) returns ``[]``, same as
        :meth:`search` on an empty store."""
        need_str(query, "query")
        need_int(limit, "limit", 1, MAX_SEARCH_LIMIT)
        org_id = self._check_org(org_id)
        personal_col = self._col(user_id)
        org_col = self._org_col(org_id)
        if personal_col.count() == 0 and org_col.count() == 0:
            return []
        uid = self._uid(user_id)
        personal_seeds = (
            self._seeds(personal_col, query, limit, user_id=uid) if personal_col.count() else []
        )
        org_seeds = (
            self._seeds(org_col, query, limit, org_id=org_id) if org_col.count() else []
        )
        q_emb = self._facts.embed_query(query) if self._facts is not None else None
        passages = self._e.retrieval.retrieve_fused_multi(
            [
                (personal_col, "personal", personal_seeds),
                (org_col, f"org:{org_id}", org_seeds),
            ],
            query,
            query_embedding=q_emb,
        )
        return [self._e.shaping.search_result(p) for p in passages]

    def answer(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,  # raised from 4 (task 0101) — see MAX_SEARCH_LIMIT
        model: Optional[str] = None,
        think: bool = False,
        chat: Optional[Callable[[str, str], str]] = None,
        llm: Any = None,
        as_of: Optional[str] = None,
    ) -> Answer:
        """Retrieve, then have an LLM distil a short direct answer plus the ids
        of the pages that support it.

        Needs an LLM: by default a local Ollama at ``$OLLAMA_HOST``
        (``http://localhost:11434``). Pass ``chat=fn(prompt, model) -> str`` to
        use any other model, or ``llm=`` for a provider spec (task 0099) —
        ``None``, a dict like ``{"provider": "muse"}``/``{"provider":
        "deepseek"}``, a callable ``fn(system, user) -> str``, or an object
        with ``generate()``; same spec as the ``Memory(llm=...)`` extractor
        config. ``chat`` takes precedence over ``llm`` if both are given. An
        unreachable LLM raises (this call has no degraded mode — use
        :meth:`search` for raw passages).

        ``think`` (task 0085): let the model reason before answering (~11s vs
        ~0.6s per call) — off by default, under evaluation as a fix for
        reader-miss empty answers on questions needing one small resolution
        step. Affects this call only, never extraction or linking. Ignored
        when ``llm``/``chat`` is set (local-Ollama-only knob).

        ``as_of`` (``backend="facts"`` only, task 0094): answer from fact
        history as it stood at that ISO timestamp instead of current truth.
        """
        need_str(query, "query")
        need_int(limit, "limit", 1, MAX_SEARCH_LIMIT)
        if as_of is not None:
            need_str(as_of, "as_of")
        col = self._col(user_id)
        if col.count() == 0:
            return Answer(answer="", source_page_ids=[])
        passages = self._passages(user_id, col, query, limit, True, as_of=as_of)
        kwargs: Dict[str, Any] = {"_chat": chat, "think": think, "llm": llm}
        if model:
            kwargs["model"] = model
        out = self._e.compression.compress(query, passages, **kwargs)
        return Answer(answer=out["answer"], source_page_ids=list(out["source_page_ids"]))

    def get(self, id: str, *, user_id: Optional[str] = None) -> Page:
        """One page by id. Unknown ids — including another user's — raise
        :class:`NotFoundError`."""
        need_str(id, "id")
        col = self._col(user_id)
        stored = self._e.storage.get_page(col, id)
        if stored is None:
            raise NotFoundError(f"page not found: {id}", status=404)
        out, incoming = self._e.shaping.link_context(self._all_pages(col))
        return self._e.shaping.page(stored, out, incoming)

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PageList:
        """Browse pages; ``query`` is a case-insensitive substring filter (use
        :meth:`search` for semantic search)."""
        need_int(limit, "limit", 1, MAX_LIST_LIMIT)
        need_int(offset, "offset", 0)
        if query is not None:
            need_str(query, "query")
        col = self._col(user_id)
        everything = self._all_pages(col)
        needle = query.strip().lower() if query else None
        matches = [
            p for p in everything if needle is None or needle in p.get("text", "").lower()
        ]
        out, incoming = self._e.shaping.link_context(everything)
        window = matches[offset:offset + limit]
        return PageList(
            pages=[self._e.shaping.page(p, out, incoming) for p in window],
            total=len(matches), limit=limit, offset=offset,
        )

    def update(self, id: str, text: str, *, user_id: Optional[str] = None) -> None:
        """Replace a page's text (re-embeds, re-extracts entities, keeps links).
        Links can go stale — call :meth:`relink` after batch edits."""
        need_str(id, "id")
        need_str(text, "text")
        col = self._col(user_id)
        if self._e.storage.get_page(col, id) is None:
            raise NotFoundError(f"page not found: {id}", status=404)
        if self._facts is not None:
            self._facts.update(self._uid(user_id), id, text)
        else:
            self._e.storage.update_page_text(col, id, text, self._nlp.entities(text))

    def delete(self, id: str, *, user_id: Optional[str] = None) -> None:
        """Delete a page. Links from other pages that pointed at it are
        skipped by retrieval and cleaned up by the next :meth:`relink`."""
        need_str(id, "id")
        col = self._col(user_id)
        if self._e.storage.get_page(col, id) is None:
            raise NotFoundError(f"page not found: {id}", status=404)
        self._e.storage.delete_page(col, id)

    def delete_all(self, *, user_id: Optional[str] = None) -> int:
        """Delete every page of one user. Returns how many were removed."""
        uid = self._uid(user_id)
        col = self._col(uid)
        removed = col.count()
        with self._lock:
            self._collections.pop(uid, None)
            if self._facts is not None:
                self._facts.delete_all(uid)
            else:
                self._client.delete_collection(col.name)
        return removed

    def relink(self, *, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Re-run the batch linking pass (idempotent). Returns
        ``{"pages": n, "links_written": m}``."""
        return self._e.linking.run_linking_pass(self._col(user_id))

    def count(self, *, user_id: Optional[str] = None) -> int:
        """Number of pages stored for one user."""
        return self._col(user_id).count()
