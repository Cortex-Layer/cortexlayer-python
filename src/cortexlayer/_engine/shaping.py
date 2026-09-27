"""Shape stored pages / retrieval passages into the public result types.

Mirrors the Cortex server's ``/v1`` shaping so local ``Memory`` and the hosted
``CortexClient`` return identical objects.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from ..types import Page, SearchResult

SCAN_CAP = 5000
_SCAN_CHUNK = 500


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def snippet(text: str) -> str:
    return " ".join(text.split())[:200]


def search_result(passage: dict) -> SearchResult:
    """Retrieval passage -> :class:`SearchResult` (with ``via`` provenance)."""
    text = passage.get("text", "")
    linked = passage.get("linked_from")
    return SearchResult(
        id=passage.get("page_id", ""),
        title=first_line(text),
        snippet=snippet(text),
        score=float(passage.get("score", 0.0) or 0.0),
        via=passage.get("via", "direct"),
        linked_from=linked if isinstance(linked, str) else None,
        source="private",
        text=text,
        tags=passage.get("tags") or {},
    )


def scan_all(collection, list_pages, cap: int = SCAN_CAP) -> List[dict]:
    """Chunked scan of one user's whole collection (capped)."""
    items: List[dict] = []
    seen = 0
    while seen < cap:
        batch = list_pages(collection, limit=_SCAN_CHUNK, offset=seen)
        if not batch:
            break
        seen += len(batch)
        items.extend(batch)
    return items


def link_context(all_pages: Iterable[dict]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """(outgoing, incoming) adjacency over ALL of a user's pages.

    Both are restricted to ids that exist and sorted; self-links are ignored
    for incoming — so the numbers agree with the server's graph view.
    """
    pages = list(all_pages)
    known = {p["id"] for p in pages}
    out = {
        p["id"]: sorted({t for t in p.get("links", []) if t in known})
        for p in pages
    }
    incoming: Dict[str, List[str]] = {}
    for nid, targets in out.items():
        for target in targets:
            if target != nid:
                incoming.setdefault(target, []).append(nid)
    return out, {k: sorted(v) for k, v in incoming.items()}


def page(
    stored: dict,
    out: Optional[Dict[str, List[str]]] = None,
    incoming: Optional[Dict[str, List[str]]] = None,
) -> Page:
    """Stored page dict -> :class:`Page` with link context."""
    pid = stored.get("id", "")
    text = stored.get("text", "")
    links = [t for t in (out or {}).get(pid, stored.get("links", [])) if t != pid]
    back = (incoming or {}).get(pid, [])
    return Page(
        id=pid,
        title=first_line(text),
        content=text,
        snippet=snippet(text),
        links=links,
        linked_from=back,
        degree=len(set(links) | set(back)),
        created_at=stored.get("created_at", "") or "",
        tags=stored.get("tags") or {},
    )
