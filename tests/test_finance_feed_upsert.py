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

from pathway.internals.api import SessionType

from realtime_rag.connectors.finance_feed import FinanceFeedSubject


def test_subject_declares_upsert_and_stop(settings):
    subj = FinanceFeedSubject(settings)
    assert subj._session_type == SessionType.UPSERT
    assert subj._stop is False
    subj.on_stop()
    assert subj._stop is True


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
