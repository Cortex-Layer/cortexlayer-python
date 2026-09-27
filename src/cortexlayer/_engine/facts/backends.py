"""Pluggable embedder for fact memory, plus the LLM backend re-exported from
``_engine.llm`` (task 0099 moved it there so ``compression.py``'s reader can
share it without importing the facts subpackage — see that module's docstring).

Defaults mirror the Cortex server's Mem0 configuration (Ollama, ``think`` off,
temperature/top_p/num_predict as Mem0's Ollama client: 0.1 / 0.1 / 2000) so the
extraction call is the same call. Anything with the same method works, so tests
and other providers need no Ollama.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Sequence

import httpx

from ...errors import CortexConfigError, LLMError
from ..llm import (  # noqa: F401 — re-exported for existing importers
    CLOUD_PROVIDERS,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_HOST,
    LLM,
    CloudLLM,
    OllamaLLM,
    _CallableLLM,
    ollama_host,
    resolve_llm,
)

DEFAULT_OLLAMA_EMBED_MODEL = "qwen3-embedding:8b"


class Embedder(Protocol):
    name: str  # recorded on the store: vectors from different embedders can't be mixed

    def embed_batch(self, texts: Sequence[str], action: str = "add") -> List[List[float]]:
        """One vector per text. ``action`` is ``add`` | ``search`` | ``update``."""
        ...


class OllamaEmbedder:
    """Embeddings over Ollama's ``/api/embed``."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        host: Optional[str] = None,
        *,
        timeout: float = 120.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model
        self.host = (host or ollama_host()).rstrip("/")
        self.name = f"ollama:{model}"
        self._timeout = timeout
        self._http = http_client

    def embed_batch(self, texts: Sequence[str], action: str = "add") -> List[List[float]]:
        if not texts:
            return []
        try:
            client = self._http or httpx.Client(timeout=self._timeout)
            try:
                resp = client.post(
                    f"{self.host}/api/embed", json={"model": self.model, "input": list(texts)}
                )
            finally:
                if self._http is None:
                    client.close()
            resp.raise_for_status()
            vectors = resp.json().get("embeddings") or []
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"Embedding call to {self.host} ({self.model}) failed: {e}") from e
        if len(vectors) != len(texts):
            raise LLMError(
                f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts ({self.model})"
            )
        return vectors


class ChromaEmbedder:
    """Chroma's built-in ONNX MiniLM-L6-v2 — no Ollama or network needed after
    the one-time ~80 MB model download. The library's default embedder."""

    name = "chroma:onnx-minilm-l6-v2"

    def __init__(self) -> None:
        self._fn: Any = None

    def embed_batch(self, texts: Sequence[str], action: str = "add") -> List[List[float]]:
        if not texts:
            return []
        if self._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._fn = DefaultEmbeddingFunction()
        return [[float(x) for x in v] for v in self._fn(list(texts))]


def resolve_embedder(spec: Any) -> Embedder:
    """``None``/``"chroma"`` → Chroma ONNX; ``"ollama"`` → Ollama default model;
    dict → ``{"provider": "ollama"|"chroma", ...}``; an object with
    ``embed_batch`` and ``name`` → itself."""
    if spec is None or spec == "chroma":
        return ChromaEmbedder()
    if spec == "ollama":
        return OllamaEmbedder()
    if isinstance(spec, dict):
        cfg = dict(spec)
        provider = cfg.pop("provider", "ollama")
        if provider == "ollama":
            return OllamaEmbedder(**cfg)
        if provider == "chroma":
            return ChromaEmbedder()
        raise CortexConfigError(f"unknown embedder provider {provider!r}")
    if hasattr(spec, "embed_batch") and hasattr(spec, "name"):
        return spec
    raise CortexConfigError(
        f"embedder must be None, 'chroma', 'ollama', a dict, or an object with embed_batch() and name; got {spec!r}"
    )
