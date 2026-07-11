"""SEC Form 4 corporate-insider transactions from EDGAR (free, no API key).

Insider open-market *buys* are the single best-replicated alternative-data
signal (Cohen-Malloy-Pomorski 2012; Lakonishok-Lee 2001: buys informative,
sells not). We take them straight from the primary source, EDGAR, for two
reasons the sibling repo's yfinance-derived alt-data cannot match:

* **Point-in-time honest.** Each transaction is keyed on the SEC filing
  *acceptance* timestamp (``known_at``), never the transaction date. Insiders
  have two business days to file, so keying on the trade date would leak days of
  look-ahead.
* **Survivorship-free.** EDGAR is filing-based, so delisted names keep their
  history instead of vanishing the way a ticker-coverage feed does.

Network fetch (:func:`fetch_insider`) is separated from parsing
(:func:`_parse_form4`, pure) so the parser is unit-testable offline on a saved
XML fixture. EDGAR's fair-access policy caps requests at 10/s and requires a
``User-Agent`` that identifies the caller — both are honoured here; ingestion
runs in a single paced process (never fanned out), like the IBKR rule.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime

from trading_simple.db import InsiderTrade

log = logging.getLogger(__name__)

_SOURCE = "sec_edgar"
# EDGAR's fair-access policy REQUIRES a descriptive User-Agent that includes a
# contact email, and returns 403 without one. This default satisfies the format;
# set TRADING_SEC_USER_AGENT to your own contact so SEC can reach you if your
# requests ever cause a problem (good practice for anything beyond light use).
_USER_AGENT = "stock-trading-simple research (contact@example.com)"
_MIN_INTERVAL_S = 0.12  # ~8 req/s, comfortably under EDGAR's 10/s ceiling
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

_cik_cache: dict[str, int] | None = None
_last_request_ts = 0.0


def _headers() -> dict[str, str]:
    """Return the EDGAR request headers (User-Agent from env or the default)."""
    import os

    return {"User-Agent": os.environ.get("TRADING_SEC_USER_AGENT", _USER_AGENT),
            "Accept-Encoding": "gzip, deflate"}


def _get(url: str, *, as_json: bool):
    """GET a URL with EDGAR pacing and headers; return parsed JSON or bytes.

    Enforces a minimum inter-request interval so a whole backfill stays under
    EDGAR's 10 requests/second fair-access limit without any external rate
    limiter.

    Args:
        url: Absolute URL to fetch.
        as_json: Parse the body as JSON when True, else return raw bytes.

    Returns:
        The decoded JSON object, or the raw response bytes.
    """
    import requests

    global _last_request_ts
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=_headers(), timeout=30)
    _last_request_ts = time.monotonic()
    resp.raise_for_status()
    return resp.json() if as_json else resp.content


def _cik_map() -> dict[str, int]:
    """Return a cached ``{ticker: CIK}`` map from EDGAR's ticker directory.

    Fetched once per process. Tickers are upper-cased; EDGAR's dotted classes
    (``BRK.B``) are stored both dotted and dash-formed so either resolves.

    Returns:
        Mapping of ticker to integer CIK.
    """
    global _cik_cache
    if _cik_cache is None:
        data = _get(_TICKERS_URL, as_json=True)
        out: dict[str, int] = {}
        for row in data.values():
            tkr = str(row["ticker"]).upper()
            out[tkr] = int(row["cik_str"])
            out[tkr.replace("-", ".")] = int(row["cik_str"])
        _cik_cache = out
    return _cik_cache


def fetch_insider(symbol: str, since: date) -> list[InsiderTrade]:
    """Fetch open-market (P/S) Form 4 insider trades for one symbol since ``since``.

    Resolves the symbol's CIK, lists its Form 4 filings knowable from ``since``
    onward (walking EDGAR's older submission shards when the recent block does
    not reach back far enough), fetches and parses each filing's primary XML, and
    keeps only open-market purchase (``P`` → ``buy``) and sale (``S`` → ``sell``)
    transactions.

    Args:
        symbol: Bare canonical ticker, e.g. ``"AAPL"``.
        since: Earliest filing date to include (by ``known_at``/acceptance).

    Returns:
        Parsed :class:`~trading_simple.db.InsiderTrade` rows (possibly empty when
        the symbol has no CIK or no qualifying filings).
    """
    cik = _cik_map().get(symbol.upper())
    if cik is None:
        log.warning("SEC: no CIK for %s; skipping insider fetch", symbol)
        return []
    trades: list[InsiderTrade] = []
    for meta in _form4_filings(cik, since):
        acc_nodash = meta["accession"].replace("-", "")
        url = _DOC_URL.format(cik=cik, acc_nodash=acc_nodash, doc=_raw_doc(meta["doc"]))
        try:
            xml = _get(url, as_json=False)
            trades.extend(_parse_form4(xml, symbol, meta["accession"],
                                       known_at=meta["known_at"], filed_at=meta["filed_at"]))
        except Exception as exc:  # noqa: BLE001 — one bad filing must not abort the symbol
            log.warning("SEC: failed to parse %s for %s: %s", meta["accession"], symbol, exc)
    return trades


def _form4_filings(cik: int, since: date) -> list[dict]:
    """List a company's Form 4 filings knowable from ``since`` onward.

    Reads ``filings.recent`` and, only when it does not reach back to ``since``,
    the referenced older submission shards. Returns lightweight dicts carrying
    the accession, primary document name, and the acceptance/filed timestamps.

    Args:
        cik: Integer central index key of the filer.
        since: Earliest filing date to include.

    Returns:
        One dict per Form 4 filing: ``accession``, ``doc``, ``known_at``,
        ``filed_at``.
    """
    subs = _get(_SUBMISSIONS_URL.format(cik=cik), as_json=True)
    blocks = [subs["filings"]["recent"]]
    recent_dates = subs["filings"]["recent"].get("filingDate", [])
    oldest_recent = min(recent_dates) if recent_dates else "9999-12-31"
    if oldest_recent > since.isoformat():  # recent block does not reach back far enough
        for extra in subs["filings"].get("files", []):
            if extra.get("filingTo", "0") >= since.isoformat():
                url = "https://data.sec.gov/submissions/" + extra["name"]
                blocks.append(_get(url, as_json=True))
    out: list[dict] = []
    for block in blocks:
        forms = block.get("form", [])
        for i, form in enumerate(forms):
            if form != "4" or block["filingDate"][i] < since.isoformat():
                continue
            accepted = block["acceptanceDateTime"][i]  # e.g. '2024-01-15T18:03:01.000Z'
            known = datetime.fromisoformat(accepted.replace("Z", "+00:00"))
            out.append({
                "accession": block["accessionNumber"][i],
                "doc": block["primaryDocument"][i],
                "known_at": known,
                "filed_at": known,
            })
    return out


def _raw_doc(doc: str) -> str:
    """Return the raw ownership-XML path from a filing's ``primaryDocument``.

    EDGAR lists the human-readable, XSL-styled rendering (HTML) under an
    ``xslF345.../`` prefix; the machine-readable Form 4 XML lives at the filing
    root under the same basename. Strip the styling prefix so we fetch parseable
    XML, not HTML.

    Args:
        doc: The ``primaryDocument`` value from the submissions index.

    Returns:
        The path to the raw ownership XML relative to the filing directory.
    """
    if "/" in doc and doc.split("/", 1)[0].lower().startswith("xsl"):
        return doc.split("/", 1)[1]
    return doc


def _strip_ns(root: ET.Element) -> None:
    """Strip XML namespaces in place so tag lookups are namespace-agnostic."""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _val(node: ET.Element | None, tag: str) -> str | None:
    """Return the text of ``<tag><value>…</value></tag>`` (or bare ``<tag>``), or None."""
    if node is None:
        return None
    child = node.find(tag)
    if child is None:
        return None
    value = child.find("value")
    text = (value.text if value is not None else child.text)
    return text.strip() if text else None


def _num(text: str | None) -> float | None:
    """Parse a numeric string to float, mapping blanks/garbage to None."""
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_form4(xml: bytes, symbol: str, accession: str, *,
                 known_at: datetime, filed_at: datetime) -> list[InsiderTrade]:
    """Parse a Form 4 XML document into open-market insider trades (pure).

    Extracts the reporting owner's name/role and every non-derivative
    transaction, keeping only open-market purchases (code ``P`` → ``"buy"``) and
    sales (code ``S`` → ``"sell"``); grants, awards, and other codes are dropped.
    The acceptance/filing timestamps come from the caller (the filing metadata),
    not the XML, so this function is a pure transform.

    Args:
        xml: Raw Form 4 XML bytes.
        symbol: Bare symbol the rows are stored under.
        accession: SEC accession number (part of the dedup key).
        known_at: Filing acceptance timestamp (public-availability time).
        filed_at: Filing timestamp (same source as ``known_at``).

    Returns:
        One :class:`~trading_simple.db.InsiderTrade` per kept transaction.
    """
    root = ET.fromstring(xml)
    _strip_ns(root)

    owner = root.find("reportingOwner")
    name = _val(owner.find("reportingOwnerId"), "rptOwnerName") if owner is not None else None
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    role = _role(rel)

    trades: list[InsiderTrade] = []
    table = root.find("nonDerivativeTable")
    if table is None:
        return trades
    for txn in table.findall("nonDerivativeTransaction"):
        code = _val(txn.find("transactionCoding"), "transactionCode")
        if code not in ("P", "S"):  # open-market purchase / sale only
            continue
        traded = _val(txn, "transactionDate")
        if not traded:
            continue
        amounts = txn.find("transactionAmounts")
        trades.append(InsiderTrade(
            symbol=symbol,
            insider=name,
            role=role,
            txn_type="buy" if code == "P" else "sell",
            shares=_num(_val(amounts, "transactionShares")),
            price=_num(_val(amounts, "transactionPricePerShare")),
            traded_at=date.fromisoformat(traded),
            filed_at=filed_at,
            known_at=known_at,
            accession=accession,
            source=_SOURCE,
        ))
    return trades


def _role(rel: ET.Element | None) -> str | None:
    """Build a human role string from a reportingOwnerRelationship node."""
    if rel is None:
        return None
    parts: list[str] = []
    if (_val(rel, "isDirector") or "").lower() in ("1", "true"):
        parts.append("Director")
    if (_val(rel, "isOfficer") or "").lower() in ("1", "true"):
        parts.append(_val(rel, "officerTitle") or "Officer")
    if (_val(rel, "isTenPercentOwner") or "").lower() in ("1", "true"):
        parts.append("10% owner")
    return "/".join(parts) or None
