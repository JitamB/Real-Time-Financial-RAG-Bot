"""market_clients must be total: upstream failures return [] and never raise
(they run inside the connector thread where an exception is fatal)."""

from __future__ import annotations

from realtime_rag.connectors import market_clients as mc


def test_quote_and_article_upsert_keys_are_stable():
    q1 = mc.Quote("AAPL", "Apple", 100.0, 1.0, "t1")
    q2 = mc.Quote("AAPL", "Apple", 250.0, -2.0, "t2")
    assert q1.key == q2.key == "quote::AAPL"  # same PK -> upsert replaces
    a1 = mc.Article("T", "b", "src", "http://x/1", "t1")
    a2 = mc.Article("T2", "b2", "src", "http://x/1", "t2")
    assert a1.key == a2.key  # same URL -> same PK (corrections replace)
    assert mc.Article("T", "b", "s", "http://x/2", "t").key != a1.key
    assert "Apple (AAPL)" in q1.as_document()
    assert "http://x/1" in a1.as_document()


def test_fetch_quotes_contains_exceptions(monkeypatch):
    class Boom:
        def __init__(self, *_a, **_k):
            raise RuntimeError("yfinance exploded")

    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.Ticker = Boom
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    # Must not raise; bad symbols are skipped -> empty list.
    assert mc.fetch_quotes({"AAPL": "Apple"}, timeout=1) == []


def test_fetch_news_no_key_is_empty():
    assert mc.fetch_news("", timeout=1) == []
    assert mc.fetch_news("your-newsapi-key-here", timeout=1) == []


def test_fetch_news_http_failure_contained(monkeypatch):
    def boom(*_a, **_k):
        raise ConnectionError("network down")

    monkeypatch.setattr(mc.requests, "get", boom)
    assert mc.fetch_news("realkey", timeout=1) == []
