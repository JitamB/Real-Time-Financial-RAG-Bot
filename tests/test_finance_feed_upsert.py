"""Channel B guarantee: the finance connector is an UPSERT source — a new
value for an existing primary key REPLACES the old row (it does not append /
accumulate). This makes a new AAPL price a true *modification* that flows
incrementally into the DocumentStore, and is the highest-risk piece of the
design (it relies on SessionType.UPSERT). Pinned by this test.

The execution case runs in a SUBPROCESS: pytest shares one process across test
files, and other tests add *streaming* connectors to Pathway's global graph, so
calling pw.run() in-process here would execute those and hang. Isolation keeps
this deterministic and fast.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import Mock

from pathway.internals.api import SessionType

import realtime_rag.config as cfg
from realtime_rag.connectors.finance_feed import FinanceFeedSubject
from realtime_rag.connectors.market_clients import Article, Quote


def test_subject_declares_upsert_and_stop(settings):
    subj = FinanceFeedSubject(settings)
    assert subj._session_type == SessionType.UPSERT
    assert subj._stop is False
    subj.on_stop()
    assert subj._stop is True


def test_content_signature_excludes_timestamp():
    # ts must NOT affect the signature, or dedup never triggers (every poll
    # stamps a fresh ts). A material change MUST change the signature.
    q1 = Quote("AAPL", "Apple", 100.0, 1.5, ts="2026-01-01T00:00:00Z")
    q2 = Quote("AAPL", "Apple", 100.0, 1.5, ts="2026-01-01T00:05:00Z")
    q3 = Quote("AAPL", "Apple", 101.0, 1.5, ts="2026-01-01T00:00:00Z")
    assert q1.content_signature == q2.content_signature  # ts-only delta
    assert q1.content_signature != q3.content_signature  # price moved

    a1 = Article("T", "body", "Src", "http://x/1", ts="2026-01-01T00:00:00Z")
    a2 = Article("T", "body", "Src", "http://x/1", ts="2026-01-01T01:00:00Z")
    a3 = Article("T", "body CORRECTED", "Src", "http://x/1", ts="2026-01-01T00:00:00Z")
    assert a1.content_signature == a2.content_signature
    assert a1.content_signature != a3.content_signature


def test_emit_dedup_skips_unchanged(settings):
    # news_dedup=True (default): identical content signature -> no re-emit
    # (no redundant re-embed); a changed signature DOES re-emit.
    subj = FinanceFeedSubject(settings)
    subj.next = Mock()  # shadow ConnectorSubject.next; capture emissions

    assert subj._emit("quote::AAPL", "AAPL @ $100", "quote", "sigA") is True
    assert subj._emit("quote::AAPL", "AAPL @ $100", "quote", "sigA") is False
    assert subj._emit("quote::AAPL", "AAPL @ $101", "quote", "sigB") is True
    assert subj.next.call_count == 2  # the dedup'd middle call did not emit


def test_emit_dedup_disabled(monkeypatch):
    # NEWS_DEDUP=false -> every poll re-emits even if nothing changed.
    monkeypatch.setenv("NEWS_DEDUP", "false")
    cfg._settings = None
    s = cfg.Settings()
    assert s.news_dedup is False
    subj = FinanceFeedSubject(s)
    subj.next = Mock()
    assert subj._emit("quote::AAPL", "AAPL @ $100", "quote", "sigA") is True
    assert subj._emit("quote::AAPL", "AAPL @ $100", "quote", "sigA") is True
    assert subj.next.call_count == 2


_PROG = textwrap.dedent(
    """
    import csv, sys, tempfile, os
    import pathway as pw
    from pathway.internals.api import SessionType
    from realtime_rag.connectors.finance_feed import FinanceDocSchema

    class FiniteFeed(pw.io.python.ConnectorSubject):
        @property
        def _session_type(self): return SessionType.UPSERT
        def run(self):
            self.next(key="quote::AAPL", data=b"AAPL @ $100",      _metadata={"path":"quote::AAPL"})
            self.next(key="quote::MSFT", data=b"MSFT @ $200",      _metadata={"path":"quote::MSFT"})
            self.next(key="quote::AAPL", data=b"AAPL @ $999 LATEST",_metadata={"path":"quote::AAPL"})
            self.commit()

    out = os.path.join(tempfile.mkdtemp(), "o.csv")
    t = pw.io.python.read(FiniteFeed(), schema=FinanceDocSchema)
    pw.io.csv.write(t, out)
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)

    import base64
    bal = {}
    for r in csv.DictReader(open(out)):
        # Pathway serialises `bytes` columns as base64 in CSV.
        data = base64.b64decode(r["data"]).decode()
        bal[(r["key"], data)] = bal.get((r["key"], data), 0) + int(r["diff"])
    aapl = {d for (k, d), b in bal.items() if k == "quote::AAPL" and b > 0}
    msft = {d for (k, d), b in bal.items() if k == "quote::MSFT" and b > 0}
    assert aapl == {"AAPL @ $999 LATEST"}, f"UPSERT BROKEN: {aapl}"
    assert msft == {"MSFT @ $200"}, msft
    print("UPSERT_OK")
    """
)


def test_same_primary_key_replaces_not_appends():
    r = subprocess.run(
        [sys.executable, "-c", _PROG], capture_output=True, text=True, timeout=180
    )
    assert "UPSERT_OK" in r.stdout, f"stdout={r.stdout}\nstderr={r.stderr[-1500:]}"
