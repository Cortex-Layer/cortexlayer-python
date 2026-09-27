"""Sync and async clients for the Cortex HTTP API (``/v1/*``).

Both classes share every request builder and response parser (``_Base``); they
differ only in the transport call and the retry sleep, so the two surfaces
cannot drift apart.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import httpx

from ._validate import need_int, need_str
from ._version import __version__
from .errors import (
    AuthenticationError,
    ConflictError,
    ConnectionError,
    CortexConfigError,
    CortexError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    WritesNotSupportedError,
)
from .types import (
    Account,
    AddResult,
    Graph,
    Page,
    PageList,
    SearchResult,
    Usage,
)

DEFAULT_BASE_URL = "https://api.cortexlayer.net"
API_KEY_ENV = "CORTEX_API_KEY"
BASE_URL_ENV = "CORTEX_BASE_URL"

MAX_SEARCH_LIMIT = 50      # server: top_k 1..50 (raised from 20, task 0101)
MAX_LIST_LIMIT = 500       # server: /v1/pages limit 1..500
USAGE_GROUPS = ("day", "key", "operation")
_RETRY_STATUS = (502, 503, 504)

DateLike = Union[str, _dt.date, _dt.datetime]


class _Op:
    """One prepared request: what to send and how to read the answer."""

    __slots__ = ("method", "path", "params", "json", "parse", "retryable", "write")

    def __init__(
        self,
        method: str,
        path: str,
        parse: Callable[[Any], Any],
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        write: bool = False,
    ) -> None:
        self.method = method
        self.path = path
        self.parse = parse
        self.params = params
        self.json = json
        # Only safe, side-effect-free calls are retried on transport errors and
        # 502/503/504: a retried write could apply twice.
        self.retryable = retryable
        self.write = write


_need_str = need_str
_need_int = need_int


def _iso(value: DateLike, name: str) -> str:
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise InvalidRequestError(f"{name} must be an ISO date/datetime string or a date")


def _page_path(page_id: str, suffix: str = "") -> str:
    _need_str(page_id, "id")
    return f"/v1/pages/{quote(page_id, safe='')}{suffix}"


def _ok_none(_: Any) -> None:
    return None


class _Base:
    """Config + request building + response handling, transport-free."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        url = base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        if key is not None and (not isinstance(key, str) or not key.strip()):
            raise CortexConfigError("api_key must be a non-empty string")
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
            raise CortexConfigError("base_url must start with http:// or https://")
        if key is None and url.rstrip("/") == DEFAULT_BASE_URL:
            raise CortexConfigError(
                f"No API key. Pass api_key=... or set {API_KEY_ENV} "
                "(create one in the Cortex web app under Keys)."
            )
        if max_retries < 0:
            raise CortexConfigError("max_retries must be >= 0")
        self._api_key = key.strip() if key else None
        self._base_url = url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff = max(0.0, retry_backoff)

    def __repr__(self) -> str:
        key = "None" if not self._api_key else f"'…{self._api_key[-4:]}'"
        return f"{type(self).__name__}(base_url={self._base_url!r}, api_key={key})"

    # --- request plumbing ---

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/json",
            "User-Agent": f"cortexlayer-python/{__version__}",
        }
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _url(self, op: _Op) -> str:
        return self._base_url + op.path

    def _delay(self, attempt: int) -> float:
        return min(self._backoff * (2 ** attempt), 4.0)

    def _finish(self, op: _Op, resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            raise self._error_for(op, resp)
        try:
            data = resp.json()
        except ValueError as e:
            raise ServerError(
                "Unexpected non-JSON response from server", status=resp.status_code
            ) from e
        try:
            return op.parse(data)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ServerError(
                f"Unexpected response shape from {op.method} {op.path}",
                status=resp.status_code,
            ) from e

    @staticmethod
    def _error_message(resp: httpx.Response) -> Tuple[str, bool]:
        """(message, came_from_a_json_error_body)."""
        try:
            body = resp.json()
        except ValueError:
            return (resp.text.strip()[:200] or resp.reason_phrase or "error"), False
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return body["error"], True
        return (str(body)[:200], False)

    def _error_for(self, op: _Op, resp: httpx.Response) -> CortexError:
        status = resp.status_code
        msg, from_json = self._error_message(resp)
        if op.write and (status == 405 or (status == 404 and not from_json)):
            return WritesNotSupportedError(
                "This Cortex server doesn't support REST writes "
                f"({op.method} {op.path} → {status}). It is running a version "
                "older than server task 0073 — update/redeploy the server.",
                status=status,
            )
        if status in (400, 422):
            return InvalidRequestError(msg, status=status)
        if status == 401:
            return AuthenticationError(
                f"{msg} — check your API key (it may be missing, wrong or revoked).",
                status=401,
            )
        if status == 403:
            return PermissionDeniedError(msg, status=403)
        if status == 404:
            return NotFoundError(msg, status=404)
        if status == 409:
            return ConflictError(msg, status=409)
        if status == 429:
            ra = resp.headers.get("Retry-After")
            try:
                retry_after: Optional[float] = float(ra) if ra is not None else None
            except ValueError:
                retry_after = None
            return RateLimitError(msg, status=429, retry_after=retry_after)
        if status >= 500:
            return ServerError(msg, status=status)
        return CortexError(msg, status=status)

    # --- operations (transport-free; shared by sync + async) ---

    @staticmethod
    def _op_add(text: str, timestamp: Optional[str]) -> _Op:
        body: Dict[str, Any] = {"text": _need_str(text, "text")}
        if timestamp is not None:
            body["timestamp"] = _need_str(timestamp, "timestamp")
        return _Op("POST", "/v1/pages", AddResult.from_dict, json=body, write=True)

    @staticmethod
    def _op_search(query: str, limit: int, expand_links: bool) -> _Op:
        _need_str(query, "query")
        _need_int(limit, "limit", 1, MAX_SEARCH_LIMIT)
        if not isinstance(expand_links, bool):
            raise InvalidRequestError("expand_links must be True or False")
        return _Op(
            "POST", "/v1/search",
            lambda d: [SearchResult.from_dict(r) for r in d["results"]],
            json={"query": query, "top_k": limit, "expand_links": expand_links},
            retryable=True,
        )

    @staticmethod
    def _op_get(page_id: str) -> _Op:
        return _Op("GET", _page_path(page_id), Page.from_dict, retryable=True)

    @staticmethod
    def _op_get_all(query: Optional[str], limit: int, offset: int) -> _Op:
        _need_int(limit, "limit", 1, MAX_LIST_LIMIT)
        _need_int(offset, "offset", 0)
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if query is not None:
            params["q"] = _need_str(query, "query")
        return _Op("GET", "/v1/pages", PageList.from_dict, params=params, retryable=True)

    @staticmethod
    def _op_update(page_id: str, text: str) -> _Op:
        return _Op(
            "PATCH", _page_path(page_id), _ok_none,
            json={"text": _need_str(text, "text")}, write=True,
        )

    @staticmethod
    def _op_delete(page_id: str) -> _Op:
        return _Op("DELETE", _page_path(page_id), _ok_none, write=True)

    @staticmethod
    def _op_relink() -> _Op:
        return _Op("POST", "/v1/relink", dict, write=True)

    @staticmethod
    def _op_me() -> _Op:
        return _Op("GET", "/v1/me", Account.from_dict, retryable=True)

    @staticmethod
    def _op_usage(
        group_by: str, since: Optional[DateLike], until: Optional[DateLike]
    ) -> _Op:
        if group_by not in USAGE_GROUPS:
            raise InvalidRequestError(f"group_by must be one of {', '.join(USAGE_GROUPS)}")
        params: Dict[str, Any] = {"group_by": group_by}
        if since is not None:
            params["since"] = _iso(since, "since")
        if until is not None:
            params["until"] = _iso(until, "until")
        return _Op("GET", "/v1/usage", Usage.from_dict, params=params, retryable=True)

    @staticmethod
    def _op_graph(limit: Optional[int]) -> _Op:
        params = {} if limit is None else {"limit": _need_int(limit, "limit", 1, 5000)}
        return _Op("GET", "/v1/graph", Graph.from_dict, params=params, retryable=True)

    @staticmethod
    def _op_neighbors(page_id: str, depth: int, limit: Optional[int]) -> _Op:
        _need_int(depth, "depth", 1, 3)
        params: Dict[str, Any] = {"depth": depth}
        if limit is not None:
            params["limit"] = _need_int(limit, "limit", 1)
        return _Op(
            "GET", _page_path(page_id, "/neighbors"), Graph.from_dict,
            params=params, retryable=True,
        )


class CortexClient(_Base):
    """Synchronous client.

    >>> c = CortexClient(api_key="ctx_...")            # doctest: +SKIP
    >>> c.add("I moved to Lisbon in March")            # doctest: +SKIP
    >>> [r.title for r in c.search("where do I live?")]  # doctest: +SKIP

    Use as a context manager (or call :meth:`close`) to release connections.
    Pass ``http_client`` to supply your own ``httpx.Client`` (proxies, custom
    transports, tests); it is never closed for you.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        super().__init__(
            api_key, base_url, timeout=timeout,
            max_retries=max_retries, retry_backoff=retry_backoff,
        )
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "CortexClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _call(self, op: _Op) -> Any:
        attempt = 0
        while True:
            try:
                resp = self._http.request(
                    op.method, self._url(op), headers=self._headers(),
                    params=op.params, json=op.json, timeout=self._timeout,
                )
            except httpx.HTTPError as e:
                if op.retryable and attempt < self._max_retries:
                    time.sleep(self._delay(attempt))
                    attempt += 1
                    continue
                raise ConnectionError(
                    f"Could not reach {self._base_url}: {e.__class__.__name__}: {e}"
                ) from e
            if resp.status_code in _RETRY_STATUS and op.retryable and attempt < self._max_retries:
                time.sleep(self._delay(attempt))
                attempt += 1
                continue
            return self._finish(op, resp)

    # --- memory ---

    def add(self, text: str, *, timestamp: Optional[str] = None) -> AddResult:
        """Save ``text`` as memory (long text is chunked into several pages).
        ``timestamp`` (e.g. ``"8 May, 2023"``) is stored as the date.

        Needs a server with REST writes (Cortex server task 0073); raises
        :class:`WritesNotSupportedError` on older servers."""
        return self._call(self._op_add(text, timestamp))

    def search(
        self, query: str, *, limit: int = 50, expand_links: bool = True
    ) -> List[SearchResult]:
        """Semantic search. With ``expand_links`` (default) related pages are
        pulled in via links — those results have ``via == "link"``.
        ``limit`` default raised 4 -> 50 (task 0101): judged LOCOMO
        open-domain subset showed a substantial accuracy gain from budget
        alone, no ranking change."""
        return self._call(self._op_search(query, limit, expand_links))

    def get(self, id: str) -> Page:
        """One page by id. Unknown ids raise :class:`NotFoundError`."""
        return self._call(self._op_get(id))

    def get_all(
        self, *, query: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> PageList:
        """Browse pages; ``query`` is a case-insensitive substring filter
        (use :meth:`search` for semantic search)."""
        return self._call(self._op_get_all(query, limit, offset))

    def update(self, id: str, text: str) -> None:
        """Replace a page's text. Needs a server with REST writes (task 0073)."""
        self._call(self._op_update(id, text))

    def delete(self, id: str) -> None:
        """Delete a page. Needs a server with REST writes (task 0073)."""
        self._call(self._op_delete(id))

    def relink(self) -> Dict[str, Any]:
        """Re-run the batch linking pass (do it after adding several
        memories). Needs a server with REST writes (task 0073)."""
        return self._call(self._op_relink())

    # --- account ---

    def me(self) -> Account:
        """The account this key belongs to (a cheap way to validate a key)."""
        return self._call(self._op_me())

    def usage(
        self,
        *,
        group_by: str = "day",
        since: Optional[DateLike] = None,
        until: Optional[DateLike] = None,
    ) -> Usage:
        """Your own API usage, aggregated by ``day``, ``key`` or ``operation``."""
        return self._call(self._op_usage(group_by, since, until))

    # --- graph ---

    def graph(self, *, limit: Optional[int] = None) -> Graph:
        """The link graph over your pages (capped; see ``truncated``)."""
        return self._call(self._op_graph(limit))

    def neighbors(self, id: str, *, depth: int = 1, limit: Optional[int] = None) -> Graph:
        """N-hop neighborhood (``depth`` 1–3) of one page."""
        return self._call(self._op_neighbors(id, depth, limit))


class AsyncCortexClient(_Base):
    """Asynchronous twin of :class:`CortexClient` (same methods, ``await`` them)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(
            api_key, base_url, timeout=timeout,
            max_retries=max_retries, retry_backoff=retry_backoff,
        )
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncCortexClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _call(self, op: _Op) -> Any:
        attempt = 0
        while True:
            try:
                resp = await self._http.request(
                    op.method, self._url(op), headers=self._headers(),
                    params=op.params, json=op.json, timeout=self._timeout,
                )
            except httpx.HTTPError as e:
                if op.retryable and attempt < self._max_retries:
                    await asyncio.sleep(self._delay(attempt))
                    attempt += 1
                    continue
                raise ConnectionError(
                    f"Could not reach {self._base_url}: {e.__class__.__name__}: {e}"
                ) from e
            if resp.status_code in _RETRY_STATUS and op.retryable and attempt < self._max_retries:
                await asyncio.sleep(self._delay(attempt))
                attempt += 1
                continue
            return self._finish(op, resp)

    async def add(self, text: str, *, timestamp: Optional[str] = None) -> AddResult:
        return await self._call(self._op_add(text, timestamp))

    async def search(
        self, query: str, *, limit: int = 50, expand_links: bool = True
    ) -> List[SearchResult]:
        return await self._call(self._op_search(query, limit, expand_links))

    async def get(self, id: str) -> Page:
        return await self._call(self._op_get(id))

    async def get_all(
        self, *, query: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> PageList:
        return await self._call(self._op_get_all(query, limit, offset))

    async def update(self, id: str, text: str) -> None:
        await self._call(self._op_update(id, text))

    async def delete(self, id: str) -> None:
        await self._call(self._op_delete(id))

    async def relink(self) -> Dict[str, Any]:
        return await self._call(self._op_relink())

    async def me(self) -> Account:
        return await self._call(self._op_me())

    async def usage(
        self,
        *,
        group_by: str = "day",
        since: Optional[DateLike] = None,
        until: Optional[DateLike] = None,
    ) -> Usage:
        return await self._call(self._op_usage(group_by, since, until))

    async def graph(self, *, limit: Optional[int] = None) -> Graph:
        return await self._call(self._op_graph(limit))

    async def neighbors(
        self, id: str, *, depth: int = 1, limit: Optional[int] = None
    ) -> Graph:
        return await self._call(self._op_neighbors(id, depth, limit))
