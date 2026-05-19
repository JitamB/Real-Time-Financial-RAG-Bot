"""External market/news clients for Channel B.

Pure functions with hard timeouts and total exception containment: a failing
upstream (yfinance / NewsAPI down, rate-limited, malformed payload) returns an
empty list and logs a warning — it must NEVER raise, because these run inside the
Pathway connector thread where an exception would kill the connector.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass

import requests

from ..observability.logging import get_logger, log_event

log = get_logger(__name__)

# Symbol -> human company name (improves semantic retrieval over a bare ticker).
DEFAULT_SYMBOLS: dict[str, str] = {
    "AAPL": "Apple",
    "GOOGL": "Google Alphabet",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "META": "Meta Facebook",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "INTC": "Intel",
}


@dataclass(frozen=True)
class Quote:
    symbol: str
    company: str
    price: float
    change_pct: float
    ts: str

    @property
    def key(self) -> str:
        """Primary key for upsert: one live row per symbol."""
        return f"quote::{self.symbol}"

    def as_document(self) -> str:
        return (
            f"[Stock quote — {self.company} ({self.symbol})] "
            f"{self.company} ({self.symbol}) is trading at ${self.price:.2f}, "
            f"{self.change_pct:+.2f}% versus the previous close. "
            f"Quote captured at {self.ts}."
        )


@dataclass(frozen=True)
class Article:
    title: str
    content: str
    source: str
    url: str
    ts: str

    @property
    def key(self) -> str:
        """Primary key for upsert: stable per article URL (corrections replace)."""
        digest = hashlib.sha1(self.url.encode("utf-8", "ignore")).hexdigest()[:16]
        return f"news::{digest}"

    def as_document(self) -> str:
        return (
            f"[Financial news — {self.source}] {self.title}. "
            f"{self.content} (published {self.ts}, source: {self.url})"
        )


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def fetch_quotes(
    symbols: dict[str, str] | None = None, timeout: float = 10.0
) -> list[Quote]:
    """Latest price per symbol via yfinance ``fast_info``. Never raises."""
    symbols = symbols or DEFAULT_SYMBOLS
    out: list[Quote] = []
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - import guard
        log_event(log, "yfinance_import_failed", level=40, error=str(exc))
        return out

    for symbol, company in symbols.items():
        try:
            info = yf.Ticker(symbol).fast_info
            price = info.last_price
            prev = info.previous_close
            if not price or not prev:
                continue
            out.append(
                Quote(
                    symbol=symbol,
                    company=company,
                    price=float(price),
                    change_pct=((float(price) - float(prev)) / float(prev)) * 100.0,
                    ts=_now_iso(),
                )
            )
        except Exception as exc:
            log_event(log, "quote_fetch_failed", level=30, symbol=symbol, error=str(exc))
            continue
    log_event(log, "quotes_fetched", count=len(out))
    return out


def fetch_news(
    api_key: str,
    symbols: dict[str, str] | None = None,
    timeout: float = 10.0,
    page_size: int = 5,
) -> list[Article]:
    """Recent financial headlines via NewsAPI. Never raises; [] if no key."""
    if not api_key or api_key.startswith("your-"):
        return []
    symbols = symbols or DEFAULT_SYMBOLS
    query = "stock market OR " + " OR ".join(symbols.keys())
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": api_key,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            log_event(log, "news_http_error", level=30, status=resp.status_code)
            return []
        articles = resp.json().get("articles", []) or []
    except Exception as exc:
        log_event(log, "news_fetch_failed", level=30, error=str(exc))
        return []

    out: list[Article] = []
    for a in articles:
        url = a.get("url") or "#"
        title = (a.get("title") or "").strip()
        body = (a.get("description") or a.get("content") or "").strip()
        if not title or url == "#":
            continue
        out.append(
            Article(
                title=title,
                content=body,
                source=(a.get("source") or {}).get("name", "NewsAPI"),
                url=url,
                ts=_now_iso(),
            )
        )
    log_event(log, "news_fetched", count=len(out))
    return out
