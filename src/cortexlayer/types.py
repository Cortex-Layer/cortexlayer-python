"""Result types. Plain frozen dataclasses (no pydantic) built tolerantly from
server JSON: unknown fields are ignored so a newer server never breaks an
older client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _s(d: Dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    return v if isinstance(v, str) else default


def _list(d: Dict[str, Any], key: str) -> List[str]:
    v = d.get(key) or []
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


def _dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


@dataclass(frozen=True)
class SearchResult:
    """One hit from :meth:`CortexClient.search`.

    ``text`` is the full memory text when the source provides it (local
    ``Memory`` always does); ``snippet`` is its first ~200 characters.

    ``via`` is ``"direct"`` (vector hit) or ``"link"`` (pulled in by
    link-expansion; ``linked_from`` is the seed page that led to it). ``score``
    semantics depend on the backend (a distance, lower = closer, on raw; the
    fused semantic + keyword + entity score, higher = closer, on facts) —
    compare within one store or server, not across.

    ``tags`` (task 0090) is whatever provenance metadata was passed to
    ``Memory.add(tags=...)`` when this page was written (``{}`` if none) —
    informational only, never a filter.
    """

    id: str
    title: str
    snippet: str
    score: float
    via: str = "direct"
    linked_from: Optional[str] = None
    source: str = "private"
    text: str = ""   # the full memory text (local ``Memory`` fills it; a server may omit it)
    tags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchResult":
        lf = d.get("linked_from")
        return cls(
            id=_s(d, "id"),
            title=_s(d, "title"),
            snippet=_s(d, "snippet"),
            score=float(d.get("score") or 0.0),
            via=_s(d, "via", "direct"),
            linked_from=lf if isinstance(lf, str) else None,
            source=_s(d, "source", "private"),
            text=_s(d, "text"),
            tags=_dict(d, "tags"),
        )


@dataclass(frozen=True)
class Page:
    """A stored memory page. ``links`` point out, ``linked_from`` point in.

    ``tags`` (task 0090) is whatever provenance metadata was passed to
    ``Memory.add(tags=...)`` when this page was written (``{}`` if none) —
    informational only, never a filter.
    """

    id: str
    title: str
    content: str
    snippet: str = ""
    links: List[str] = field(default_factory=list)
    linked_from: List[str] = field(default_factory=list)
    degree: int = 0
    created_at: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Page":
        return cls(
            id=_s(d, "id"),
            title=_s(d, "title"),
            content=_s(d, "content"),
            snippet=_s(d, "snippet"),
            links=_list(d, "links"),
            linked_from=_list(d, "linked_from"),
            degree=int(d.get("degree") or 0),
            created_at=_s(d, "created_at"),
            tags=_dict(d, "tags"),
        )


@dataclass(frozen=True)
class PageList:
    """One window of :meth:`CortexClient.get_all` (``total`` counts all matches)."""

    pages: List[Page]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PageList":
        return cls(
            pages=[Page.from_dict(p) for p in d.get("pages") or []],
            total=int(d.get("total") or 0),
            limit=int(d.get("limit") or 0),
            offset=int(d.get("offset") or 0),
        )


@dataclass(frozen=True)
class AddResult:
    """Ids of the pages created by :meth:`CortexClient.add` (long text is
    chunked into several small pages)."""

    page_ids: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AddResult":
        return cls(page_ids=_list(d, "page_ids"))


@dataclass(frozen=True)
class Account:
    user_id: str
    account_name: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Account":
        return cls(user_id=_s(d, "user_id"), account_name=_s(d, "account_name"))


@dataclass(frozen=True)
class UsageBucket:
    key: str
    calls: int
    errors: int
    avg_latency_ms: Optional[float] = None
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UsageBucket":
        avg = d.get("avg_latency_ms")
        lab = d.get("label")
        return cls(
            key=_s(d, "key"),
            calls=int(d.get("calls") or 0),
            errors=int(d.get("errors") or 0),
            avg_latency_ms=float(avg) if isinstance(avg, (int, float)) else None,
            label=lab if isinstance(lab, str) else None,
        )


@dataclass(frozen=True)
class Usage:
    """Your own API usage, aggregated (``group_by``: day | key | operation)."""

    group_by: str
    since: str
    until: str
    total_calls: int
    total_errors: int
    buckets: List[UsageBucket]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Usage":
        return cls(
            group_by=_s(d, "group_by"),
            since=_s(d, "since"),
            until=_s(d, "until"),
            total_calls=int(d.get("total_calls") or 0),
            total_errors=int(d.get("total_errors") or 0),
            buckets=[UsageBucket.from_dict(b) for b in d.get("buckets") or []],
        )


@dataclass(frozen=True)
class Graph:
    """Raw graph payload (``nodes`` / ``edges`` as the server sends them)."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    truncated: bool = False
    stale: bool = False
    start: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Graph":
        start = d.get("start")
        return cls(
            nodes=list(d.get("nodes") or []),
            edges=list(d.get("edges") or []),
            truncated=bool(d.get("truncated", False)),
            stale=bool(d.get("stale", False)),
            start=start if isinstance(start, str) else None,
        )


@dataclass(frozen=True)
class Answer:
    """A short direct answer distilled from retrieved pages
    (:meth:`Memory.answer`; needs an LLM — Ollama by default)."""

    answer: str
    source_page_ids: List[str]
