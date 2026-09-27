"""Fact-memory engine: LLM-extracted facts in Chroma, entity-boosted search.

Implements the algorithm of Mem0 2.1's "V3 phased batch pipeline" (Apache
License 2.0; https://github.com/mem0ai/mem0, ``mem0/memory/main.py``) on Cortex's
own storage, so Cortex does not depend on ``mem0ai``. See the repository NOTICE.

``add`` (one call per piece of text):

0. context: the last 10 messages previously added for this user (sqlite);
1. retrieve the 10 most similar existing facts (their ids are replaced by
   "0".."9" so the model cannot hallucinate real ids);
2. ONE LLM call with Mem0's additive-extraction prompt (+ optional custom
   instructions) -> JSON list of self-contained facts;
3. batch-embed the facts;
4. md5 dedup against those 10 existing facts and within the batch;
5. insert each fact as a page (same schema as the raw backend, so the linking
   pass, link-expansion and browse work unchanged) with its entities;
6. index the facts' spaCy entities in a per-user entity collection.

``search``: embed the query, over-fetch semantic candidates, boost the ones
linked to entities found in the query, threshold + rank (see ``scoring``).

Differences from Mem0, all deliberate:
- optional ``observation_date_from_timestamp`` (off = identical to Mem0);
- no history/audit table (only the last-10-messages context table);
- BM25 keyword term is opt-in (``keyword_weight``; Mem0 has it but it is inactive with
  Chroma, so off = identical to Mem0 — see ``scoring`` and task 0079);
- an embedding *outage* raises instead of silently dropping facts;
- ``linked_memory_ids`` in the LLM output is ignored, exactly as Mem0 ignores it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence

from ...errors import CortexConfigError, LLMError
from .. import ingestion, storage
from ..nlp import NLP
from . import entities as entity_extraction
from . import prompts, scoring
from .backends import LLM, Embedder

log = logging.getLogger("cortexlayer.facts")

FACTS_PREFIX = "cortex_facts__"
ENTITIES_PREFIX = "cortex_facts_entities__"
CONTEXT_MESSAGES = 10          # last-k messages shown to the extractor
EXISTING_FACTS_CONTEXT = 10    # similar existing facts shown to the extractor
ENTITY_REUSE_SIMILARITY = 0.95  # an entity this close to an existing one is the same entity
SEARCH_THRESHOLD = 0.1


# --- LLM reply parsing (Mem0's remove_code_blocks / extract_json) -----------------


def _strip_code_blocks(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(
            b if isinstance(b, str) else (b.get("text", "") if isinstance(b, dict) else "")
            for b in content
        )
    if not isinstance(content, str):
        return ""
    stripped = content.strip()
    match = re.match(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$", stripped)
    inner = match.group(1).strip() if match else stripped
    return re.sub(r"<think>.*?</think>", "", inner, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def parse_extraction(reply: Any) -> List[Dict[str, Any]]:
    """The model's reply -> ``[{"text": ..., ...}]``; ``[]`` if unparseable or empty
    (logged, never raised — a garbled reply is not a service outage)."""
    body = _strip_code_blocks(reply)
    if not body:
        return []
    try:
        try:
            parsed = json.loads(body, strict=False)
        except json.JSONDecodeError:
            parsed = json.loads(_extract_json(body), strict=False)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("could not parse extraction reply as JSON: %s", e)
        return []
    memories = parsed.get("memory", []) if isinstance(parsed, dict) else []
    return [m for m in memories if isinstance(m, dict)] if isinstance(memories, list) else []


# --- last-k messages (Mem0's sqlite ``messages`` table) ---------------------------


class _Messages:
    """Recent add() texts per user, shown to the extractor as context."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        with self._connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, "
                "role TEXT, content TEXT)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS messages_scope ON messages (scope, seq)")

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        return sqlite3.connect(self._path)

    def last(self, scope: str, limit: int = CONTEXT_MESSAGES) -> List[Dict[str, str]]:
        with self._lock, self._connect() as c:
            rows = c.execute(
                "SELECT role, content FROM (SELECT seq, role, content FROM messages "
                "WHERE scope = ? ORDER BY seq DESC LIMIT ?) ORDER BY seq ASC",
                (scope, limit),
            ).fetchall()
        return [{"role": r, "content": t} for r, t in rows]

    def save(self, scope: str, messages: Sequence[Dict[str, str]]) -> None:
        with self._lock, self._connect() as c:
            c.executemany(
                "INSERT INTO messages (scope, role, content) VALUES (?, ?, ?)",
                [(scope, m.get("role"), m.get("content")) for m in messages],
            )
            c.execute(  # keep only the newest CONTEXT_MESSAGES per scope
                "DELETE FROM messages WHERE scope = ? AND seq NOT IN "
                "(SELECT seq FROM messages WHERE scope = ? ORDER BY seq DESC LIMIT ?)",
                (scope, scope, CONTEXT_MESSAGES),
            )

    def clear(self, scope: str) -> None:
        with self._lock, self._connect() as c:
            c.execute("DELETE FROM messages WHERE scope = ?", (scope,))


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


class FactEngine:
    """LLM fact extraction + entity-boosted search over per-user Chroma collections."""

    def __init__(
        self,
        client: Any,
        data_dir: str,
        llm: LLM,
        embedder: Embedder,
        nlp: NLP,
        *,
        custom_instructions: Optional[str] = None,
        threshold: float = SEARCH_THRESHOLD,
        observation_date_from_timestamp: bool = False,
        keyword_weight: float = 0.0,
    ) -> None:
        self._client = client
        self._llm = llm
        self._embedder = embedder
        self._nlp = nlp
        # Mem0-style entity extraction needs a real spaCy pipeline; without one
        # the entity index stays empty and search is plain semantic (as Mem0).
        self._spacy = getattr(nlp, "_nlp", None)
        self._custom = custom_instructions or None
        self._threshold = threshold
        # Mem0's pipeline (as the Cortex server uses it) never passes an observation
        # date, so the extractor resolves "yesterday"/"last week" against TODAY. With
        # this on, the add() timestamp is passed as the Observation Date instead.
        self._obs_from_ts = observation_date_from_timestamp
        # 0 = no keyword term (identical to Mem0 on Chroma); >0 fuses BM25 into search.
        self._kw_weight = max(float(keyword_weight), 0.0)
        self._messages = _Messages(os.path.join(data_dir, "cortexlayer_facts.db"))

    # --- collections ---

    def _open(self, name: str):
        col = self._client.get_or_create_collection(
            name=name, metadata={"embedder": self._embedder.name}
        )
        recorded = (col.metadata or {}).get("embedder")
        if recorded and recorded != self._embedder.name:
            raise CortexConfigError(
                f"This store was created with embedder {recorded!r} but {self._embedder.name!r} is "
                "configured; vectors from different embedders cannot be mixed. Use the original "
                "embedder or a new data_dir."
            )
        return col

    def collection(self, user_id: str):
        """The user's facts collection (same page schema as the raw backend)."""
        return self._open(FACTS_PREFIX + user_id)

    def _entities(self, user_id: str):
        return self._open(ENTITIES_PREFIX + user_id)

    def _embed(self, texts: Sequence[str], action: str) -> List[List[float]]:
        try:
            return self._embedder.embed_batch(list(texts), action)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001 — provider errors of any kind
            raise LLMError(f"embedding failed: {e}") from e

    def embed_query(self, query: str) -> List[float]:
        """A query's vector in this store's embedding space (task 0077: lets
        link-expansion rank a seed's neighbors by relevance to the query —
        facts collections hold vectors from this custom embedder, not
        Chroma's default text function, so ranking needs the matching vector)."""
        return self._embed([query], "search")[0]

    # --- add ---

    def add(
        self,
        user_id: str,
        text: str,
        timestamp: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Extract facts from ``text`` and store them. Returns ``[{"id", "text"}]``
        (empty if nothing was worth remembering). Raises :class:`LLMError` if the
        model or embedder is unreachable — that is not "no facts".

        ``tags`` (task 0090): optional provenance metadata, stamped as-is onto
        every fact extracted from this one call (via ``storage.insert_page``)."""
        if timestamp:
            text = ingestion.apply_timestamp(text, timestamp)
        messages = [{"role": "user", "content": text}]
        parsed = f"user: {text}\n"
        facts = self.collection(user_id)

        # Phase 0-1: context + similar existing facts (ids hidden behind "0".."9")
        last = self._messages.last(user_id)
        existing_texts: List[Dict[str, str]] = []
        existing_hashes: set = set()
        total = facts.count()
        if total:
            qvec = self._embed([parsed], "search")[0]
            res = facts.query(
                query_embeddings=[qvec], n_results=min(EXISTING_FACTS_CONTEXT, total),
                include=["documents", "metadatas"],
            )
            for i, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0])):
                existing_texts.append({"id": str(i), "text": doc})
                if meta and meta.get("hash"):
                    existing_hashes.add(meta["hash"])

        # Phase 2: one LLM call
        user_prompt = prompts.build_extraction_prompt(
            existing_memories=existing_texts, new_messages=parsed,
            last_k_messages=last, custom_instructions=self._custom,
            timestamp=timestamp if self._obs_from_ts else None,
        )
        try:
            reply = self._llm.generate(prompts.extraction_system_prompt(), user_prompt)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM extraction failed: {e}") from e
        extracted = parse_extraction(reply)
        if not extracted:
            self._messages.save(user_id, messages)
            return []

        # Phase 3: batch embed (fall back one by one; an outage must not look like "nothing")
        texts = [m["text"] for m in extracted if isinstance(m.get("text"), str) and m["text"]]
        vectors: Dict[str, List[float]] = {}
        try:
            vectors = dict(zip(texts, self._embed(texts, "add")))
        except LLMError:
            failures = 0
            for t in texts:
                try:
                    vectors[t] = self._embed([t], "add")[0]
                except LLMError as e:
                    failures += 1
                    log.warning("could not embed extracted fact, skipping it: %s", e)
            if texts and failures == len(texts):
                raise

        # Phase 4-5: hash dedup (vs the similar existing facts and within the batch)
        stamp = timestamp or storage.utc_now_iso()
        now = storage.utc_now_iso()
        records: List[Dict[str, Any]] = []
        seen: set = set()
        for m in extracted:
            t = m.get("text")
            if not t or t not in vectors:
                continue
            h = hashlib.md5(t.encode()).hexdigest()
            if h in existing_hashes or h in seen:
                continue
            seen.add(h)
            extra: Dict[str, Any] = {"hash": h, "updated_at": now}
            if m.get("attributed_to"):
                extra["attributed_to"] = str(m["attributed_to"])
            records.append({"text": t, "vector": vectors[t], "extra": extra})
        if not records:
            self._messages.save(user_id, messages)
            return []

        # Phase 6: persist
        out: List[Dict[str, str]] = []
        for rec in records:
            pid = storage.insert_page(
                facts, rec["text"], self._nlp.entities(rec["text"]),
                created_at=stamp, embedding=rec["vector"], extra=rec["extra"], tags=tags,
            )
            rec["id"] = pid
            out.append({"id": pid, "text": rec["text"]})

        # Phase 7: entity index (best effort, as Mem0)
        try:
            self._link_entities(user_id, records)
        except Exception as e:  # noqa: BLE001
            log.warning("entity linking failed: %s", e)

        self._messages.save(user_id, messages)
        return out

    def _link_entities(self, user_id: str, records: List[Dict[str, Any]]) -> None:
        if self._spacy is None:
            return
        per_record = entity_extraction.extract_entities_batch(
            [r["text"] for r in records], nlp=self._spacy
        )
        merged: Dict[str, List[Any]] = {}   # norm -> [type, text, {memory ids}]
        for rec, ents in zip(records, per_record):
            for etype, etext in ents:
                key = _norm(etext)
                if key in merged:
                    merged[key][2].add(rec["id"])
                else:
                    merged[key] = [etype, etext, {rec["id"]}]
        if not merged:
            return
        keys = list(merged)
        vecs = self._embed([merged[k][1] for k in keys], "add")

        col = self._entities(user_id)
        exact: Dict[str, Any] = {}
        if col.count():
            listed = col.get(include=["metadatas", "documents"])
            for eid, meta, doc in zip(listed["ids"], listed["metadatas"], listed["documents"]):
                exact.setdefault(_norm(doc or ""), (eid, meta or {}))
        nearest = None
        if col.count():
            nearest = col.query(query_embeddings=vecs, n_results=1, include=["distances"])

        ins_ids, ins_vecs, ins_meta, ins_docs = [], [], [], []
        for j, key in enumerate(keys):
            etype, etext, mem_ids = merged[key]
            hit = exact.get(key)
            if hit is None and nearest is not None and nearest["ids"][j]:
                if scoring.distance_to_score(nearest["distances"][j][0]) >= ENTITY_REUSE_SIMILARITY:
                    eid = nearest["ids"][j][0]
                    got = col.get(ids=[eid], include=["metadatas"])
                    hit = (eid, (got["metadatas"] or [{}])[0] or {})
            if hit is not None:
                eid, meta = hit
                linked = set(storage._decode_list(meta.get("linked_memory_ids"))) | mem_ids
                col.update(ids=[eid], metadatas=[{"linked_memory_ids": storage._encode_list(sorted(linked))}])
            else:
                ins_ids.append(storage.new_page_id())
                ins_vecs.append(vecs[j])
                ins_docs.append(etext)
                ins_meta.append({
                    "entity_type": etype,
                    "linked_memory_ids": storage._encode_list(sorted(mem_ids)),
                })
        if ins_ids:
            col.add(ids=ins_ids, embeddings=ins_vecs, documents=ins_docs, metadatas=ins_meta)

    # --- bulk import (migration from a Mem0 store) ---

    def import_facts(
        self,
        user_id: str,
        facts: Sequence[Dict[str, Any]],
        entities: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, int]:
        """Load already-embedded facts (and their entity index) without an LLM
        or embedder call — used to move a Mem0 store into this engine.

        ``facts``: ``{"id", "text", "vector", "created_at"?, "updated_at"?,
        "hash"?, "attributed_to"?, "tags"?}``. ``entities``: ``{"id", "text", "type",
        "vector", "linked_memory_ids"}``. Ids are preserved, so links between
        the two survive. Idempotent: ids already present are skipped. The
        vectors must come from the embedder this engine is configured with —
        the collection is stamped with its name and cannot be told otherwise.
        Returns ``{"facts", "facts_skipped", "entities", "entities_skipped"}``.
        """
        col = self.collection(user_id)
        have = set(col.get(ids=[f["id"] for f in facts])["ids"]) if facts else set()
        added = 0
        for f in facts:
            if f["id"] in have:
                continue
            text = f["text"]
            extra: Dict[str, Any] = {
                "hash": f.get("hash") or hashlib.md5(text.encode()).hexdigest(),
                "updated_at": f.get("updated_at") or f.get("created_at") or storage.utc_now_iso(),
            }
            if f.get("attributed_to"):
                extra["attributed_to"] = str(f["attributed_to"])
            storage.insert_page(
                col, text, self._nlp.entities(text), page_id=f["id"],
                created_at=f.get("created_at"), embedding=list(f["vector"]), extra=extra,
                tags=f.get("tags"),
            )
            added += 1
        stored = have | {f["id"] for f in facts}

        ecol = self._entities(user_id)
        ehave = set(ecol.get(ids=[e["id"] for e in entities])["ids"]) if entities else set()
        ins = [
            e for e in entities
            if e["id"] not in ehave and any(m in stored for m in e["linked_memory_ids"])
        ]
        if ins:
            ecol.add(
                ids=[e["id"] for e in ins],
                embeddings=[list(e["vector"]) for e in ins],
                documents=[e["text"] for e in ins],
                metadatas=[{
                    "entity_type": e.get("type") or "",
                    # Only ids that exist: mem0 keeps ids of memories since deleted.
                    "linked_memory_ids": storage._encode_list(
                        sorted(m for m in e["linked_memory_ids"] if m in stored)
                    ),
                } for e in ins],
            )
        return {
            "facts": added, "facts_skipped": len(have),
            "entities": len(ins), "entities_skipped": len(entities) - len(ins),
        }

    # --- search ---

    def seeds(self, user_id: str, query: str, limit: int) -> List[dict]:
        """Top ``limit`` facts for ``query`` as storage-shaped page dicts with a
        fused ``score`` (higher = better)."""
        facts = self.collection(user_id)
        total = facts.count()
        if total == 0:
            return []
        query_entities = (
            entity_extraction.extract_entities(query, nlp=self._spacy) if self._spacy else []
        )
        qvec = self.embed_query(query)
        res = facts.query(
            query_embeddings=[qvec], n_results=min(max(limit * 4, 60), total),
            include=["documents", "metadatas", "distances"],
        )
        candidates = [
            {"id": pid, "score": scoring.distance_to_score(dist), "document": doc, "metadata": meta}
            for pid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        boosts = self._entity_boosts(user_id, query_entities) if query_entities else {}
        keyword: Dict[str, float] = {}
        if self._kw_weight > 0:
            keyword = self._keyword_scores(facts, query, qvec, candidates)
        ranked = scoring.score_and_rank(
            candidates, boosts, self._threshold, limit, keyword, self._kw_weight
        )
        pages = []
        for r in ranked:
            page = storage._page_from_result(r["id"], r["document"], r["metadata"])
            page["score"] = r["score"]
            pages.append(page)
        return pages

    def _keyword_scores(
        self, facts: Any, query: str, qvec: Sequence[float], candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """BM25 over all of the user's facts. Keyword hits the semantic over-fetch
        missed are appended to ``candidates`` (with their true semantic score) so
        an exact-term fact can win even when it is not among the nearest vectors.

        Reads every fact's text per query — exact and always in sync with the
        store (no side index to maintain), O(facts); fine to ~10^4 per user."""
        try:
            got = facts.get(include=["documents"])
            scores = scoring.keyword_scores(query, dict(zip(got["ids"], got["documents"])))
            have = {c["id"] for c in candidates}
            missing = sorted((i for i in scores if i not in have), key=scores.get, reverse=True)
            missing = missing[: scoring.KEYWORD_MAX_EXTRA_CANDIDATES]
            if missing:
                extra = facts.get(ids=missing, include=["documents", "metadatas", "embeddings"])
                for pid, doc, meta, vec in zip(
                    extra["ids"], extra["documents"], extra["metadatas"], extra["embeddings"]
                ):
                    # Chroma's default space is squared L2, as in query(); mirror it exactly.
                    dist = sum((a - b) ** 2 for a, b in zip(vec, qvec))
                    candidates.append({
                        "id": pid, "score": scoring.distance_to_score(dist),
                        "document": doc, "metadata": meta,
                    })
            return scores
        except Exception as e:  # noqa: BLE001 — a keyword failure degrades to semantic search
            log.warning("keyword scoring failed: %s", e)
            return {}

    def _entity_boosts(self, user_id: str, query_entities: Sequence[Any]) -> Dict[str, float]:
        seen: set = set()
        deduped: List[str] = []
        for _type, text in list(query_entities)[: scoring.MAX_QUERY_ENTITIES]:
            key = _norm(text)
            if key and key not in seen:
                seen.add(key)
                deduped.append(text)
        col = self._entities(user_id)
        total = col.count()
        if not deduped or total == 0:
            return {}
        boosts: Dict[str, float] = {}
        try:
            vecs = self._embed(deduped, "search")
            for vec in vecs:
                res = col.query(
                    query_embeddings=[vec], n_results=min(500, total),
                    include=["metadatas", "distances"],
                )
                for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
                    sim = scoring.distance_to_score(dist)
                    if sim < scoring.ENTITY_MATCH_MIN_SIMILARITY:
                        continue
                    linked = storage._decode_list((meta or {}).get("linked_memory_ids"))
                    boost = scoring.entity_boost(sim, len(linked))
                    for mid in linked:
                        boosts[mid] = max(boosts.get(mid, 0.0), boost)
        except Exception as e:  # noqa: BLE001 — a boost failure degrades to plain search
            log.warning("entity boost failed: %s", e)
        return boosts

    # --- edit ---

    def update(self, user_id: str, page_id: str, text: str) -> None:
        """Replace a fact's text (re-embed, refresh hash/entities; links kept).
        The entity index is not rewritten (as Mem0)."""
        vec = self._embed([text], "update")[0]
        self.collection(user_id).update(
            ids=[page_id], documents=[text], embeddings=[vec],
            metadatas=[{
                "hash": hashlib.md5(text.encode()).hexdigest(),
                "updated_at": storage.utc_now_iso(),
                "entities": storage._encode_list(self._nlp.entities(text)),
            }],
        )

    def delete_all(self, user_id: str) -> None:
        for name in (FACTS_PREFIX + user_id, ENTITIES_PREFIX + user_id):
            try:
                self._client.delete_collection(name)
            except Exception:  # noqa: BLE001 — already absent
                pass
        self._messages.clear(user_id)
