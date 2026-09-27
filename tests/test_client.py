"""Client tests: request shapes, parsing, errors, retries, config, sync/async
parity. Everything runs against ``httpx.MockTransport`` — no server."""

import asyncio
import json

import httpx
import pytest

import cortexlayer as cx
from cortexlayer import (
    AsyncCortexClient,
    AuthenticationError,
    ConflictError,
    ConnectionError,
    CortexClient,
    CortexConfigError,
    CortexError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    WritesNotSupportedError,
)

BASE = "https://api.test"
KEY = "ctx_secretkey1234"

PAGE = {
    "id": "p1", "title": "Alice moved to Lisbon", "content": "Alice moved to Lisbon.",
    "snippet": "Alice moved to Lisbon.", "links": ["p2"], "linked_from": ["p3"],
    "degree": 2, "created_at": "2026-01-01T00:00:00Z", "future_field": "ignored",
}
HIT = {"id": "p1", "title": "T", "snippet": "S", "score": 0.4, "source": "private",
       "via": "direct"}
LINK_HIT = {**HIT, "id": "p2", "via": "link", "linked_from": "p1", "score": 0.0}


class Recorder:
    """A MockTransport handler that records requests and replays responses."""

    def __init__(self, *responses):
        self.requests = []
        self._responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def last(self):
        return self.requests[-1]

    def body(self):
        return json.loads(self.last.content)


def J(data, status=200, **kw):
    return httpx.Response(status, json=data, **kw)


def client(rec, **kw):
    kw.setdefault("retry_backoff", 0)
    return CortexClient(
        api_key=KEY, base_url=BASE,
        http_client=httpx.Client(transport=httpx.MockTransport(rec)), **kw,
    )


def aclient(rec, **kw):
    kw.setdefault("retry_backoff", 0)
    return AsyncCortexClient(
        api_key=KEY, base_url=BASE,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(rec)), **kw,
    )


# --- request shapes + parsing -------------------------------------------------


def test_search_request_and_parsing():
    rec = Recorder(J({"results": [HIT, LINK_HIT]}))
    results = client(rec).search("where does alice live?", limit=5)
    assert rec.last.method == "POST" and str(rec.last.url) == f"{BASE}/v1/search"
    assert rec.body() == {"query": "where does alice live?", "top_k": 5,
                          "expand_links": True}
    assert [r.id for r in results] == ["p1", "p2"]
    assert results[0].via == "direct" and results[0].linked_from is None
    assert results[1].via == "link" and results[1].linked_from == "p1"
    assert results[0].score == 0.4
    assert results[0].text == ""                      # servers may omit the full text
    assert cx.SearchResult.from_dict({**HIT, "text": "full text"}).text == "full text"


def test_search_can_disable_link_expansion():
    rec = Recorder(J({"results": []}))
    assert client(rec).search("q", expand_links=False) == []
    assert rec.body()["expand_links"] is False


def test_auth_and_user_agent_headers():
    rec = Recorder(J({"user_id": "alice", "account_name": "alice"}))
    acct = client(rec).me()
    assert rec.last.headers["authorization"] == f"Bearer {KEY}"
    assert rec.last.headers["user-agent"] == f"cortexlayer-python/{cx.__version__}"
    assert (acct.user_id, acct.account_name) == ("alice", "alice")


def test_get_quotes_the_id_and_ignores_unknown_fields():
    rec = Recorder(J(PAGE))
    page = client(rec).get("a/b c")
    assert rec.last.method == "GET"
    assert rec.last.url.raw_path.decode() == "/v1/pages/a%2Fb%20c"
    assert page.id == "p1" and page.links == ["p2"] and page.linked_from == ["p3"]
    assert page.degree == 2 and not hasattr(page, "future_field")


def test_get_all_params_and_pagelist():
    rec = Recorder(J({"pages": [PAGE], "total": 152, "limit": 10, "offset": 20}))
    result = client(rec).get_all(query="lisbon", limit=10, offset=20)
    assert dict(rec.last.url.params) == {"limit": "10", "offset": "20", "q": "lisbon"}
    assert result.total == 152 and result.limit == 10 and result.offset == 20
    assert result.pages[0].title.startswith("Alice")


def test_get_all_defaults_omit_the_filter():
    rec = Recorder(J({"pages": [], "total": 0, "limit": 50, "offset": 0}))
    client(rec).get_all()
    assert dict(rec.last.url.params) == {"limit": "50", "offset": "0"}


def test_add_update_delete_relink_request_shapes():
    rec = Recorder(J({"page_ids": ["a", "b"]}, 201))
    out = client(rec).add("I moved to Lisbon.", timestamp="8 May, 2023")
    assert rec.last.method == "POST" and rec.last.url.path == "/v1/pages"
    assert rec.body() == {"text": "I moved to Lisbon.", "timestamp": "8 May, 2023"}
    assert out.page_ids == ["a", "b"]

    rec = Recorder(J({"page_ids": ["a"]}))
    client(rec).add("x")
    assert rec.body() == {"text": "x"}                       # no timestamp key

    rec = Recorder(J({"ok": True, "page_id": "p1"}))
    assert client(rec).update("p1", "new text") is None
    assert rec.last.method == "PATCH" and rec.last.url.path == "/v1/pages/p1"
    assert rec.body() == {"text": "new text"}

    rec = Recorder(J({"ok": True, "page_id": "p1"}))
    assert client(rec).delete("p1") is None
    assert rec.last.method == "DELETE" and rec.last.url.path == "/v1/pages/p1"

    rec = Recorder(J({"pages": 3, "links": 2}))
    assert client(rec).relink() == {"pages": 3, "links": 2}
    assert rec.last.method == "POST" and rec.last.url.path == "/v1/relink"


def test_usage_params_and_parsing():
    import datetime as dt

    payload = {"group_by": "key", "since": "a", "until": "b", "total_calls": 5,
               "total_errors": 1, "buckets": [
                   {"key": "key_1", "label": "hermes", "calls": 5, "errors": 1,
                    "avg_latency_ms": 12.5}]}
    rec = Recorder(J(payload))
    u = client(rec).usage(group_by="key", since=dt.date(2026, 9, 1), until="2026-09-22")
    assert dict(rec.last.url.params) == {
        "group_by": "key", "since": "2026-09-01", "until": "2026-09-22"}
    assert u.total_calls == 5 and u.buckets[0].label == "hermes"
    assert u.buckets[0].avg_latency_ms == 12.5


def test_graph_and_neighbors():
    g = {"nodes": [{"id": "a"}], "edges": [{"source": "a", "target": "b"}],
         "truncated": True, "stale": False, "start": "a"}
    rec = Recorder(J(g))
    graph = client(rec).graph(limit=10)
    assert dict(rec.last.url.params) == {"limit": "10"}
    assert graph.truncated is True and graph.nodes == [{"id": "a"}]
    n = client(rec).neighbors("a", depth=2, limit=50)
    assert rec.last.url.path == "/v1/pages/a/neighbors"
    assert dict(rec.last.url.params) == {"depth": "2", "limit": "50"}
    assert n.start == "a"


# --- validation (client-side, no request sent) --------------------------------


@pytest.mark.parametrize("call", [
    lambda c: c.search(""),
    lambda c: c.search("   "),
    lambda c: c.search("q", limit=0),
    lambda c: c.search("q", limit=51),
    lambda c: c.search("q", limit=True),
    lambda c: c.search("q", expand_links="yes"),
    lambda c: c.add(""),
    lambda c: c.add(None),
    lambda c: c.add("x", timestamp=""),
    lambda c: c.get(""),
    lambda c: c.update("p1", ""),
    lambda c: c.delete(""),
    lambda c: c.get_all(limit=501),
    lambda c: c.get_all(offset=-1),
    lambda c: c.get_all(query=""),
    lambda c: c.usage(group_by="week"),
    lambda c: c.usage(since=123),
    lambda c: c.neighbors("p1", depth=4),
    lambda c: c.graph(limit=0),
])
def test_bad_arguments_fail_before_any_request(call):
    rec = Recorder(J({}))
    with pytest.raises(InvalidRequestError):
        call(client(rec))
    assert rec.requests == []


def test_invalid_request_is_also_a_value_error():
    with pytest.raises(ValueError):
        client(Recorder(J({}))).search("")


# --- errors ---------------------------------------------------------------------


@pytest.mark.parametrize("status,exc", [
    (400, InvalidRequestError), (422, InvalidRequestError),
    (401, AuthenticationError), (403, PermissionDeniedError),
    (404, NotFoundError), (409, ConflictError),
    (429, RateLimitError), (500, ServerError), (502, ServerError),
    (418, CortexError),
])
def test_error_status_mapping(status, exc):
    rec = Recorder(J({"error": "boom"}, status))
    with pytest.raises(exc) as ei:
        client(rec, max_retries=0).get("p1")
    assert ei.value.status == status
    assert isinstance(ei.value, CortexError)
    assert "boom" in str(ei.value)


def test_permission_error_exposes_the_server_code():
    rec = Recorder(J({"error": "session_required"}, 403))
    with pytest.raises(PermissionDeniedError) as ei:
        client(rec).me()
    assert ei.value.code == "session_required"


def test_rate_limit_carries_retry_after():
    rec = Recorder(J({"error": "slow down"}, 429, headers={"Retry-After": "7"}))
    with pytest.raises(RateLimitError) as ei:
        client(rec).search("q")
    assert ei.value.retry_after == 7.0
    rec = Recorder(J({"error": "slow down"}, 429, headers={"Retry-After": "soon"}))
    with pytest.raises(RateLimitError) as ei:
        client(rec).search("q")
    assert ei.value.retry_after is None


def test_401_message_points_at_the_key():
    rec = Recorder(J({"error": "missing or invalid API key"}, 401))
    with pytest.raises(AuthenticationError, match="API key"):
        client(rec).me()


def test_non_json_error_body_still_maps():
    rec = Recorder(httpx.Response(502, text="<html>bad gateway</html>"))
    with pytest.raises(ServerError, match="bad gateway"):
        client(rec, max_retries=0).get("p1")


def test_success_with_garbage_body_is_a_server_error():
    with pytest.raises(ServerError):
        client(Recorder(httpx.Response(200, text="not json"))).me()
    with pytest.raises(ServerError):
        client(Recorder(J({"unexpected": True}))).search("q")   # no "results"


def test_transport_failure_becomes_connection_error():
    rec = Recorder(httpx.ConnectError("refused"))
    with pytest.raises(ConnectionError, match="refused"):
        client(rec, max_retries=0).me()


def test_write_routes_missing_on_the_server_are_reported_clearly():
    """404 with a plain body (route not found) / 405 on a write route →
    WritesNotSupportedError; a JSON 404 (unknown page id) stays NotFoundError."""
    for resp in (httpx.Response(404, text="Not Found"), J({"error": "x"}, 405),
                 httpx.Response(405, text="Method Not Allowed")):
        for call in (lambda c: c.add("x"), lambda c: c.update("p1", "x"),
                     lambda c: c.delete("p1"), lambda c: c.relink()):
            with pytest.raises(WritesNotSupportedError, match="0073"):
                call(client(Recorder(resp), max_retries=0))
    with pytest.raises(NotFoundError):
        client(Recorder(J({"error": "page not found: p1"}, 404))).delete("p1")
    with pytest.raises(NotFoundError):                 # reads never say "writes"
        client(Recorder(httpx.Response(404, text="Not Found")), max_retries=0).get("p1")


# --- retries --------------------------------------------------------------------


def test_reads_retry_on_5xx_then_succeed():
    rec = Recorder(J({"error": "x"}, 503), J({"error": "x"}, 502), J(PAGE))
    assert client(rec, max_retries=2).get("p1").id == "p1"
    assert len(rec.requests) == 3


def test_reads_retry_on_transport_errors():
    rec = Recorder(httpx.ReadTimeout("slow"), J({"results": []}))
    assert client(rec).search("q") == []
    assert len(rec.requests) == 2


def test_retries_are_bounded():
    rec = Recorder(J({"error": "down"}, 503))
    with pytest.raises(ServerError):
        client(rec, max_retries=2).get("p1")
    assert len(rec.requests) == 3            # 1 try + 2 retries


@pytest.mark.parametrize("call", [
    lambda c: c.add("x"),
    lambda c: c.update("p1", "x"),
    lambda c: c.delete("p1"),
    lambda c: c.relink(),
])
def test_writes_are_never_retried(call):
    rec = Recorder(J({"error": "down"}, 503))
    with pytest.raises(ServerError):
        call(client(rec, max_retries=3))
    assert len(rec.requests) == 1
    rec = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(ConnectionError):
        call(client(rec, max_retries=3))
    assert len(rec.requests) == 1


def test_auth_errors_are_not_retried():
    rec = Recorder(J({"error": "bad"}, 401))
    with pytest.raises(AuthenticationError):
        client(rec, max_retries=3).me()
    assert len(rec.requests) == 1


# --- config ---------------------------------------------------------------------


def test_key_and_url_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("CORTEX_API_KEY", "envkey-9999")
    monkeypatch.setenv("CORTEX_BASE_URL", "http://localhost:8000/")
    rec = Recorder(J({"user_id": "u", "account_name": "u"}))
    c = CortexClient(http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    c.me()
    assert str(rec.last.url) == "http://localhost:8000/v1/me"   # trailing slash trimmed
    assert rec.last.headers["authorization"] == "Bearer envkey-9999"


def test_explicit_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("CORTEX_API_KEY", "envkey")
    monkeypatch.setenv("CORTEX_BASE_URL", "http://env.example")
    rec = Recorder(J({"user_id": "u", "account_name": "u"}))
    CortexClient(api_key="explicit", base_url=BASE,
                 http_client=httpx.Client(transport=httpx.MockTransport(rec))).me()
    assert rec.last.url.host == "api.test"
    assert rec.last.headers["authorization"] == "Bearer explicit"


def test_hosted_default_without_a_key_fails_fast(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    monkeypatch.delenv("CORTEX_BASE_URL", raising=False)
    with pytest.raises(CortexConfigError, match="CORTEX_API_KEY"):
        CortexClient()


def test_self_hosted_open_mode_needs_no_key(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    rec = Recorder(J({"user_id": "default", "account_name": "default"}))
    c = CortexClient(base_url="http://localhost:8000",
                     http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    c.me()
    assert "authorization" not in rec.last.headers


@pytest.mark.parametrize("kw", [
    {"api_key": ""}, {"api_key": "  "}, {"api_key": KEY, "base_url": "ftp://x"},
    {"api_key": KEY, "base_url": "no-scheme"}, {"api_key": KEY, "max_retries": -1},
])
def test_bad_config_is_rejected(kw):
    with pytest.raises(CortexConfigError):
        CortexClient(**kw)


def test_repr_never_leaks_the_key():
    c = CortexClient(api_key=KEY, base_url=BASE)
    assert KEY not in repr(c) and "1234" in repr(c)
    c.close()


def test_context_manager_closes_only_a_client_it_owns():
    injected = httpx.Client(transport=httpx.MockTransport(Recorder(J({}))))
    with CortexClient(api_key=KEY, base_url=BASE, http_client=injected):
        pass
    assert not injected.is_closed                      # yours stays open
    with CortexClient(api_key=KEY, base_url=BASE) as owned:
        http = owned._http
    assert http.is_closed


# --- async parity ---------------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


def test_async_client_has_the_same_public_surface():
    def public(cls):
        return {n for n in dir(cls) if not n.startswith("_")} - {"close", "aclose"}

    assert public(AsyncCortexClient) == public(CortexClient)


def test_async_round_trip_and_headers():
    async def go():
        rec = Recorder(J({"results": [HIT, LINK_HIT]}))
        async with aclient(rec) as c:
            results = await c.search("q", limit=3, expand_links=False)
        return rec, results

    rec, results = run(go())
    assert rec.body() == {"query": "q", "top_k": 3, "expand_links": False}
    assert rec.last.headers["authorization"] == f"Bearer {KEY}"
    assert [r.via for r in results] == ["direct", "link"]


def test_async_errors_validation_and_retries():
    async def go():
        with pytest.raises(InvalidRequestError):
            await aclient(Recorder(J({}))).search("")
        with pytest.raises(NotFoundError):
            await aclient(Recorder(J({"error": "page not found: x"}, 404))).get("x")
        rec = Recorder(J({"error": "x"}, 503), J(PAGE))
        assert (await aclient(rec).get("p1")).id == "p1"
        assert len(rec.requests) == 2
        rec = Recorder(J({"error": "x"}, 503))
        with pytest.raises(ServerError):
            await aclient(rec).add("x")                 # writes: no retry
        assert len(rec.requests) == 1
        with pytest.raises(WritesNotSupportedError):
            await aclient(Recorder(httpx.Response(404, text="Not Found"))).delete("p")
        with pytest.raises(ConnectionError):
            await aclient(Recorder(httpx.ConnectError("no")), max_retries=0).me()

    run(go())


def test_async_pages_and_usage_parse():
    async def go():
        c = aclient(Recorder(J({"pages": [PAGE], "total": 1, "limit": 50, "offset": 0})))
        pl = await c.get_all()
        c2 = aclient(Recorder(J({"group_by": "day", "since": "a", "until": "b",
                                 "total_calls": 0, "total_errors": 0, "buckets": []})))
        return pl, await c2.usage()

    pl, usage = run(go())
    assert pl.pages[0].id == "p1" and usage.buckets == []
