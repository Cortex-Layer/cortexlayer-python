"""Ingestion: chunk raw text into small pages, extract entities, embed, insert.

Pages are deliberately small — roughly one fact/statement each — because
link-expansion works best on focused pages. Chunking: split on line breaks,
then greedily pack sentences so no page exceeds ``MAX_PAGE_CHARS``. Entity
extraction is delegated to an :class:`~cortexlayer._engine.nlp.NLP` object.

Ingestion never runs the linking pass — linking stays a batch operation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from chromadb.api.models.Collection import Collection

from . import storage
from .nlp import NLP

MAX_PAGE_CHARS = 500

_DATE_PREFIX_RE = re.compile(r"^\[.+?\]")


def apply_timestamp(text: str, timestamp: str) -> str:
    """Prefix ``[timestamp]`` onto undated lines only.

    Lines already starting with a ``[date]``-style bracket prefix keep their
    original date instead of getting a second one stacked on top.
    """

    def prefix_line(line: str) -> str:
        if not line.strip():
            return line
        if _DATE_PREFIX_RE.match(line.lstrip()):
            return line
        return f"[{timestamp}] {line}"

    return "\n".join(prefix_line(line) for line in text.splitlines())


def chunk_text(text: str, nlp: NLP, max_chars: int = MAX_PAGE_CHARS) -> List[str]:
    """Split raw text into page-sized chunks."""
    pages: List[str] = []
    for block in (b.strip() for b in text.splitlines()):
        if not block:
            continue
        if len(block) <= max_chars:
            pages.append(block)
            continue
        # Long turn: split into sentences and greedily repack.
        current: List[str] = []
        current_len = 0
        for s in nlp.sentences(block):
            if current and current_len + 1 + len(s) > max_chars:
                pages.append(" ".join(current))
                current, current_len = [], 0
            current.append(s)
            current_len += (1 if current_len else 0) + len(s)
        if current:
            pages.append(" ".join(current))
    return pages


def add_text(
    collection: Collection,
    text: str,
    nlp: NLP,
    timestamp: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Chunk → entities → insert. Returns the new page ids.

    ``timestamp`` (e.g. ``"8 May, 2023"``) is prefixed to undated lines and
    stored as each page's ``created_at``. ``tags`` (task 0090) is optional
    provenance metadata stored on every resulting page unchanged — see
    ``storage.insert_page``.
    """
    if timestamp:
        text = apply_timestamp(text, timestamp)
    return [
        storage.insert_page(
            collection, chunk, nlp.entities(chunk), created_at=timestamp or None, tags=tags
        )
        for chunk in chunk_text(text, nlp)
    ]
