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

The same EDGAR machinery also serves company **fundamentals**
(:func:`fetch_fundamentals`): the XBRL ``companyfacts`` endpoint returns a
company's entire as-reported financial history in one request, from which
:func:`_parse_companyfacts` (pure) derives point-in-time balance-sheet levels
(equity, assets, shares) and trailing-twelve-month income flows (net income,
gross profit) — the raw material for cross-sectional value/quality factors. Filers'
own mistakes (thousand-fold unit errors, zero share counts) are stored as filed, as a
point-in-time archive must; the parser owns only the choice of XBRL tag, which it makes
per filing or per period so a concept never freezes when a filer changes tags, and
:mod:`trading_simple.features` screens implausible values at read time. Both
network fetches are separated from their pure parsers so the parsers are
unit-testable offline on saved fixtures. EDGAR's fair-access policy caps requests
at 10/s and requires a ``User-Agent`` that identifies the caller — both are
honoured here; ingestion runs in a single paced process (never fanned out), like
the IBKR rule.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from trading_simple.db import Fundamental, InsiderTrade

log = logging.getLogger(__name__)

_SOURCE = "sec_edgar"
# Bulk quarterly Form 345 rows are tagged distinctly from the per-filing path for
# honest provenance (and to match the sibling repo's convention in the shared DB).
_SOURCE_BULK = "sec_edgar_bulk"
# EDGAR's fair-access policy REQUIRES a descriptive User-Agent that includes a
# contact email, and returns 403 without one. This default satisfies the format;
# set TRADING_SEC_USER_AGENT to your own contact so SEC can reach you if your
# requests ever cause a problem (good practice for anything beyond light use).
_USER_AGENT = "stock-trading-simple research (contact@example.com)"
_MIN_INTERVAL_S = 0.12  # ~8 req/s, comfortably under EDGAR's 10/s ceiling
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
# `companyfacts` returns a company's ENTIRE as-reported XBRL history in one
# request, each fact carrying its `filed` date — the point-in-time key. (We never
# use the `frames` API: it returns the latest/restated value, which leaks.)
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_FUND_KNOWN_HOUR = 21  # a filing acts from 21:00 UTC (after the US close), like short interest

# Logical fundamental concept -> ordered (taxonomy, tag) fallbacks, highest priority
# first. EVERY tag is read, and the order is resolved per filing (instants) or per period
# (flows) — see _parse_companyfacts — never once for the whole history. INSTANT concepts
# (balance-sheet levels) are matched here; FLOW concepts (income statement, rolled to
# trailing-twelve-months) are below.
_FUND_INSTANT_TAGS = {
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ("ifrs-full", "Equity"),
    ],
    "assets": [("us-gaap", "Assets"), ("ifrs-full", "Assets")],
    # A point-in-time count first; a filing's weighted-average basic count only when it
    # reports none (multi-class issuers stopped filing a non-dimensional cover-page count).
    "shares": [("dei", "EntityCommonStockSharesOutstanding"),
               ("us-gaap", "CommonStockSharesOutstanding"),
               ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
               ("ifrs-full", "WeightedAverageShares")],
}
#: Share-count tags reported as DURATION facts; :func:`_as_instants` converts them.
_FUND_DURATION_SHARES = frozenset({"us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
                                   "ifrs-full:WeightedAverageShares"})
_FUND_INSTANT_UNIT = {"equity": "USD", "assets": "USD", "shares": "shares"}
_FUND_NI_TAGS = [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"),
                 ("ifrs-full", "ProfitLoss")]
_FUND_GP_TAGS = [("us-gaap", "GrossProfit")]  # merged with revenue - cogs, period by period
_FUND_REV_TAGS = [("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                  ("us-gaap", "Revenues"), ("us-gaap", "SalesRevenueNet"),
                  ("ifrs-full", "Revenue")]
_FUND_COGS_TAGS = [("us-gaap", "CostOfGoodsAndServicesSold"),
                   ("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsSold")]

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

    When the directory lists one ticker under several CIKs, the LOWEST wins:
    operating companies registered decades ago carry low CIKs, while the
    colliding entries are recently-registered shells (EDGAR listed ``XOM``
    under both Exxon Mobil Corp, CIK 34088, and an "ExxonMobil Holdings Corp"
    shell, CIK 2115436 — last-write-wins resolved to the shell, so every
    SEC-sourced sleeve silently returned zero rows for XOM).

    Returns:
        Mapping of ticker to integer CIK.
    """
    global _cik_cache
    if _cik_cache is None:
        data = _get(_TICKERS_URL, as_json=True)
        out: dict[str, int] = {}
        for row in data.values():
            tkr, cik = str(row["ticker"]).upper(), int(row["cik_str"])
            for key in (tkr, tkr.replace("-", ".")):
                if key not in out or cik < out[key]:
                    out[key] = cik
        _cik_cache = out
    return _cik_cache


def fetch_insider(symbol: str, since: date, *, until: date | None = None) -> list[InsiderTrade]:
    """Fetch open-market (P/S) Form 4 insider trades for one symbol since ``since``.

    Resolves the symbol's CIK, lists its Form 4 filings knowable from ``since``
    onward (walking EDGAR's older submission shards when the recent block does
    not reach back far enough), fetches and parses each filing's primary XML, and
    keeps only open-market purchase (``P`` → ``buy``) and sale (``S`` → ``sell``)
    transactions.

    Args:
        symbol: Bare canonical ticker, e.g. ``"AAPL"``.
        since: Earliest filing date to include (by ``known_at``/acceptance).
        until: Latest filing date to include (inclusive); ``None`` means no upper
            bound. Used to fetch only the missing early gap of an already-covered
            symbol without re-pulling the covered tail.

    Returns:
        Parsed :class:`~trading_simple.db.InsiderTrade` rows (possibly empty when
        the symbol has no CIK or no qualifying filings).
    """
    cik = _cik_map().get(symbol.upper())
    if cik is None:
        log.warning("SEC: no CIK for %s; skipping insider fetch", symbol)
        return []
    trades: list[InsiderTrade] = []
    for meta in _form4_filings(cik, since, until):
        acc_nodash = meta["accession"].replace("-", "")
        url = _DOC_URL.format(cik=cik, acc_nodash=acc_nodash, doc=_raw_doc(meta["doc"]))
        try:
            xml = _get(url, as_json=False)
            trades.extend(_parse_form4(xml, symbol, meta["accession"],
                                       known_at=meta["known_at"], filed_at=meta["filed_at"]))
        except Exception as exc:  # noqa: BLE001 — one bad filing must not abort the symbol
            log.warning("SEC: failed to parse %s for %s: %s", meta["accession"], symbol, exc)
    return trades


def _form4_filings(cik: int, since: date, until: date | None = None) -> list[dict]:
    """List a company's Form 4 filings knowable from ``since`` onward.

    Reads ``filings.recent`` and, only when it does not reach back to ``since``,
    the referenced older submission shards. Returns lightweight dicts carrying
    the accession, primary document name, and the acceptance/filed timestamps.

    Args:
        cik: Integer central index key of the filer.
        since: Earliest filing date to include.
        until: Latest filing date to include (inclusive); ``None`` means no bound.

    Returns:
        One dict per Form 4 filing: ``accession``, ``doc``, ``known_at``,
        ``filed_at``.
    """
    until_iso = until.isoformat() if until is not None else None
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
            filed = block["filingDate"][i]
            if form != "4" or filed < since.isoformat() or (until_iso and filed > until_iso):
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


# ---------------------------------------------------------------------------
# Insider — BULK quarterly Form 3/4/5 data sets (whole-market historic backfill)
# ---------------------------------------------------------------------------
# :func:`fetch_insider` above costs one HTTP request PER Form 4 filing, so an
# active name (thousands of filings) makes a full-universe backfill run for days.
# SEC also publishes every quarter's Section-16 filings as a single ~14 MB
# structured-TSV archive covering ALL issuers (2006Q1-present), which turns that
# backfill into a handful of downloads parsed locally. :func:`fetch_insider_bulk`
# uses it for CLOSED quarters; :func:`fetch_insider` remains the incremental path
# for the current, not-yet-published quarter's tail.
_FORM345_LANDING = ("https://www.sec.gov/data-research/sec-markets-data/"
                    "insider-transactions-data-sets")
_FORM345_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
_form345_url_cache: dict[str, str] | None = None


def _form345_urls() -> dict[str, str]:
    """Return ``{quarter: zip_url}`` scraped from the SEC landing page (cached once).

    SEC is migrating the archive path (older quarters live under ``structureddata``,
    the newest under ``datastandardsinnovation``), so a fixed URL guess 404s. The
    published hrefs are the reliable source of truth and also reveal which quarters
    are available yet (the current, incomplete quarter is not).

    Returns:
        Mapping of ``YYYYqN`` quarter label to the absolute zip URL.
    """
    global _form345_url_cache
    if _form345_url_cache is None:
        import re

        html = _get(_FORM345_LANDING, as_json=False).decode("utf-8", "replace")
        urls: dict[str, str] = {}
        for path, quarter in re.findall(r'href="([^"]*?(\d{4}q\d)_form345\.zip)"', html):
            urls[quarter] = path if path.startswith("http") else "https://www.sec.gov" + path
        _form345_url_cache = urls
    return _form345_url_cache


def bulk_published_until() -> date | None:
    """Return the last day SEC's newest published Form 345 data set covers (or None).

    The data sets are published per calendar quarter of filing date, some weeks after
    it ends; :func:`iter_insider_bulk` serves every day up to this one, the per-filing
    :func:`fetch_insider` the days after it.

    Returns:
        The newest published quarter's last day, or ``None`` when none is listed.
    """
    quarters = _form345_urls()
    if not quarters:
        return None
    year, q = map(int, max(quarters).split("q"))
    return date(year + q // 4, q * 3 % 12 + 1, 1) - timedelta(days=1)


def _quarters_since(since: date, until: date) -> list[str]:
    """List ``YYYYqN`` quarter labels spanning ``since``..``until`` inclusive."""
    out, y, q = [], since.year, (since.month - 1) // 3 + 1
    while (y, q) <= (until.year, (until.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def _download_form345(quarter: str, cache_dir: Path) -> Path | None:
    """Download one quarter's ``form345`` zip into ``cache_dir`` (cached; None if unpublished).

    Honours the same EDGAR pacing as :func:`_get` and reuses the cached file on a
    re-run, so a repeated backfill re-parses locally without re-downloading.

    Args:
        quarter: ``YYYYqN`` label.
        cache_dir: Directory to store the zip in.

    Returns:
        Path to the local zip, or ``None`` when SEC has not published this quarter yet.
    """
    import requests

    global _last_request_ts
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{quarter}_form345.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = _form345_urls().get(quarter)
    if url is None:
        return None
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=_headers(), timeout=180)
    _last_request_ts = time.monotonic()
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def iter_insider_bulk(symbols: list[str], since: date, *, until: date | None = None,
                      cache_dir: str | None = None):
    """Yield ``(quarter, [InsiderTrade])`` per published quarter from SEC's data sets.

    Downloads each published quarterly Form 3/4/5 archive spanning ``since``..``until``
    (default today) into ``cache_dir`` and parses it, keeping only ``symbols``. One
    ~14 MB download serves every symbol, so this replaces the per-filing
    :func:`fetch_insider` for a historic backfill (days -> minutes) and is
    survivorship-free (all issuers, including delisted names). Yielding one quarter at
    a time keeps peak memory bounded to a single quarter's trades — a full-universe
    accumulate-then-write would otherwise hold millions of rows at once. The current,
    not-yet-published quarter is not covered here — use :func:`fetch_insider` for that
    incremental tail.

    Args:
        symbols: Bare canonical tickers to keep.
        since: Earliest filing date to include.
        until: Latest filing date to consider (default: today).
        cache_dir: Directory for the downloaded zips (default: a temp subdir).

    Yields:
        ``(quarter_label, rows)`` per published quarter (unpublished quarters skipped).
    """
    import tempfile
    import zipfile

    until = until or date.today()
    symset = {s.upper() for s in symbols}
    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "tsim_form345"
    for quarter in _quarters_since(since, until):
        path = _download_form345(quarter, cache)
        if path is None:
            continue  # quarter not published yet
        try:
            with zipfile.ZipFile(path) as zf:
                sub = zf.read("SUBMISSION.tsv")
                own = zf.read("REPORTINGOWNER.tsv")
                trans = zf.read("NONDERIV_TRANS.tsv")
        except (zipfile.BadZipFile, KeyError) as exc:  # a truncated/renamed archive
            log.warning("SEC: unreadable form345 archive for %s: %s", quarter, exc)
            continue
        yield quarter, _parse_form345_tables(sub, own, trans, symset, since, until)


def fetch_insider_bulk(symbols: list[str], since: date, *, until: date | None = None,
                       cache_dir: str | None = None) -> list[InsiderTrade]:
    """Fetch open-market Form 4 trades for many symbols from SEC's quarterly data sets.

    Convenience wrapper that materialises every quarter from :func:`iter_insider_bulk`
    into one list — fine for a handful of symbols; a full-universe backfill should
    consume :func:`iter_insider_bulk` and UPSERT per quarter to bound memory.

    Args:
        symbols: Bare canonical tickers to keep.
        since: Earliest filing date to include.
        until: Latest filing date to consider (default: today).
        cache_dir: Directory for the downloaded zips (default: a temp subdir).

    Returns:
        Parsed :class:`~trading_simple.db.InsiderTrade` rows across all ``symbols``
        (deduped downstream by the ``insider_trades`` UPSERT).
    """
    trades: list[InsiderTrade] = []
    for _quarter, rows in iter_insider_bulk(symbols, since, until=until, cache_dir=cache_dir):
        trades.extend(rows)
    return trades


def _tsv_rows(data: bytes):
    """Yield each row of a tab-separated SEC dataset table as a dict (pure)."""
    import csv
    import io

    yield from csv.DictReader(io.StringIO(data.decode("latin-1")),
                              delimiter="\t", quoting=csv.QUOTE_NONE)


def _parse_ddmonyyyy(text: str | None) -> date | None:
    """Parse SEC's ``DD-MON-YYYY`` date (e.g. ``05-FEB-2024``), else None."""
    if not text:
        return None
    try:
        day, mon, year = text.split("-")
        return date(int(year), _FORM345_MONTHS[mon.upper()], int(day))
    except (ValueError, KeyError):
        return None


def _role_from_relationship(relationship: str | None, title: str | None) -> str | None:
    """Build a role string from a REPORTINGOWNER relationship + title (pure).

    The relationship field is usually comma-joined (``"Director,Officer"``) but is
    sometimes concatenated without a delimiter (``"TenPercentOwnerOther"``) and
    occasionally empty, so it is decoded by case-insensitive substring — matching the
    output shape of :func:`_role` on the per-filing path.

    Args:
        relationship: The ``RPTOWNER_RELATIONSHIP`` field.
        title: The ``RPTOWNER_TITLE`` field (used as the officer title).

    Returns:
        A ``"/"``-joined role string, or None when no role token is present.
    """
    rel = (relationship or "").lower()
    parts: list[str] = []
    if "director" in rel:
        parts.append("Director")
    if "officer" in rel:
        parts.append((title or "").strip() or "Officer")
    if "tenpercentowner" in rel:
        parts.append("10% owner")
    return "/".join(parts) or None


def _parse_form345_tables(submission: bytes, reporting_owner: bytes, nonderiv_trans: bytes,
                          symbols: set[str], since: date,
                          until: date | None = None) -> list[InsiderTrade]:
    """Derive open-market (P/S) Form 4 trades from three quarterly-dataset TSVs (pure).

    Mirrors :func:`_parse_form4`'s filter: keeps ``DOCUMENT_TYPE`` ``"4"`` (originals
    only — ``4/A`` amendments carry a separate accession that cannot be linked to
    supersede the original, so they are dropped to avoid double counting) and
    non-derivative transaction codes ``P``/``S``. The three tables join on
    ``ACCESSION_NUMBER``; the issuer symbol comes straight from SUBMISSION (no CIK
    lookup). ``known_at``/``filed_at`` are stamped at 21:00 UTC of the filing date —
    the same conservative point-in-time convention as fundamentals (Form 4
    ``FILING_DATE`` is the Eastern-Time date of EDGAR acceptance), coarser than the
    per-filing path's intraday acceptance timestamp but equivalent at the daily-bar
    resolution the feature layer aligns and lags to.

    Args:
        submission: Raw ``SUBMISSION.tsv`` bytes.
        reporting_owner: Raw ``REPORTINGOWNER.tsv`` bytes.
        nonderiv_trans: Raw ``NONDERIV_TRANS.tsv`` bytes.
        symbols: Upper-cased bare tickers to keep.
        since: Earliest filing date to include.
        until: Latest filing date to include (inclusive); ``None`` means no upper
            bound. Lets a gap-fill of an already-covered symbol keep only the
            missing early portion of the boundary quarter.

    Returns:
        Parsed :class:`~trading_simple.db.InsiderTrade` rows for the kept filings.
    """
    keep: dict[str, tuple[str, date]] = {}   # accession -> (symbol, filing_date)
    for row in _tsv_rows(submission):
        if row.get("DOCUMENT_TYPE") != "4":
            continue
        sym = (row.get("ISSUERTRADINGSYMBOL") or "").upper()
        if sym not in symbols:
            continue
        filed = _parse_ddmonyyyy(row.get("FILING_DATE"))
        if filed is None or filed < since or (until is not None and filed > until):
            continue
        keep[row["ACCESSION_NUMBER"]] = (sym, filed)

    owners: dict[str, tuple[str | None, str | None]] = {}
    for row in _tsv_rows(reporting_owner):
        acc = row.get("ACCESSION_NUMBER")
        if acc in keep and acc not in owners:   # first reporting owner wins (like _parse_form4)
            owners[acc] = (row.get("RPTOWNERNAME") or None,
                           _role_from_relationship(row.get("RPTOWNER_RELATIONSHIP"),
                                                   row.get("RPTOWNER_TITLE")))

    trades: list[InsiderTrade] = []
    for row in _tsv_rows(nonderiv_trans):
        acc = row.get("ACCESSION_NUMBER")
        code = row.get("TRANS_CODE")
        if acc not in keep or code not in ("P", "S"):   # open-market purchase / sale only
            continue
        traded = _parse_ddmonyyyy(row.get("TRANS_DATE"))
        if traded is None:
            continue
        sym, filed = keep[acc]
        name, role = owners.get(acc, (None, None))
        known = _known_at(filed)
        trades.append(InsiderTrade(
            symbol=sym, insider=name, role=role,
            txn_type="buy" if code == "P" else "sell",
            shares=_num(row.get("TRANS_SHARES")), price=_num(row.get("TRANS_PRICEPERSHARE")),
            traded_at=traded, filed_at=known, known_at=known,
            accession=acc, source=_SOURCE_BULK,
        ))
    return trades


# ---------------------------------------------------------------------------
# Fundamentals — SEC XBRL companyfacts -> point-in-time value/quality inputs
# ---------------------------------------------------------------------------
def fetch_fundamentals(symbol: str, since: date) -> list[Fundamental]:
    """Fetch as-reported fundamentals for one symbol from SEC XBRL ``companyfacts``.

    Resolves the symbol's CIK and pulls its whole ``companyfacts`` document (one
    request — it carries the entire history), then derives point-in-time rows via
    :func:`_parse_companyfacts`.

    ``since`` is intentionally **not** used to trim: a balance-sheet level or an
    annual figure filed before the backtest window must still seed the
    forward-filled feature at the window's start, so all available history is
    ingested. UPSERT keeps re-runs idempotent, so this is safe to repeat.

    Args:
        symbol: Bare canonical ticker, e.g. ``"AAPL"``.
        since: Present for a uniform provider signature; not used to trim (see above).

    Returns:
        Parsed :class:`~trading_simple.db.Fundamental` rows (empty when the symbol
        has no CIK or no usable facts).
    """
    cik = _cik_map().get(symbol.upper())
    if cik is None:
        log.warning("SEC: no CIK for %s; skipping fundamentals fetch", symbol)
        return []
    try:
        facts = _get(_COMPANYFACTS_URL.format(cik=cik), as_json=True)
    except Exception as exc:  # noqa: BLE001 — a missing/!200 companyfacts must not abort the batch
        log.warning("SEC: companyfacts fetch failed for %s: %s", symbol, exc)
        return []
    return _parse_companyfacts(facts, symbol)


def _parse_companyfacts(facts: dict, symbol: str) -> list[Fundamental]:
    """Derive point-in-time fundamental rows from a ``companyfacts`` document (pure).

    Every tag in a concept's ``_FUND_*_TAGS`` fallback list is read, and the fallback
    order is resolved per FILING (instants) or per PERIOD (flows) — never once for the
    whole history. Locking a concept onto the first tag that ever had data froze it the
    day a filer abandoned that tag: multi-class issuers stopped filing a non-dimensional
    cover-page share count (V, MA and CMCSA sat on 2010 counts for fourteen years),
    AVGO and QCOM moved to the incl.-noncontrolling equity tag, and AMZN stopped tagging
    ``GrossProfit`` after its 2009 annual report.

    * Balance-sheet **instants** (equity, assets, shares): each filing contributes the
      facts of the highest-priority tag it reports (:func:`_by_filing`), then
      :func:`_instant_rows` keeps one row per successively-newer period at first
      disclosure. Shares fall back to the filing's weighted-average basic count
      (:func:`_as_instants`) when it reports no point-in-time count.
    * Income-statement **flows** (net income, gross profit) are rolled to
      trailing-twelve-months per tag by :func:`_flow_rows` and merged earliest filing
      first (:func:`_earliest`). Gross profit merges the tagged ``GrossProfit`` with
      ``revenue - cogs`` (Novy-Marx's definition), so synthesis fills the periods the tag
      does not cover.

    Args:
        facts: The parsed ``companyfacts`` JSON (``{"facts": {taxonomy: {tag: ...}}}``).
        symbol: Bare symbol the rows are stored under.

    Returns:
        All derived :class:`~trading_simple.db.Fundamental` rows (unordered).
    """
    data = facts.get("facts", {})
    rows: list[Fundamental] = []
    for concept, tags in _FUND_INSTANT_TAGS.items():
        found = _as_instants(_all_tagged(data, tags, _FUND_INSTANT_UNIT[concept]))
        rows += _instant_rows(symbol, concept, _by_filing(found, tags))
    rows += _earliest([_flow_rows(symbol, "ni_ttm", _all_tagged(data, [tag], "USD"))
                       for tag in _FUND_NI_TAGS])
    # Flows only: a stray instant fact of a higher tag must not hide a filing's usable flow.
    rev = _by_filing([f for f in _all_tagged(data, _FUND_REV_TAGS, "USD") if f["start"]],
                     _FUND_REV_TAGS)
    cogs = _by_filing([f for f in _all_tagged(data, _FUND_COGS_TAGS, "USD") if f["start"]],
                      _FUND_COGS_TAGS)
    rows += _earliest([_flow_rows(symbol, "gp_ttm", _all_tagged(data, _FUND_GP_TAGS, "USD")),
                       _flow_rows(symbol, "gp_ttm", _synth_gross_profit(rev, cogs))])
    return rows


def _all_tagged(data: dict, tags: list[tuple[str, str]], unit: str) -> list[dict]:
    """Return normalized facts for EVERY ``(taxonomy, tag)`` in ``tags`` under ``unit`` (pure).

    Each fact keeps its ``taxonomy:tag`` label, so the caller resolves the fallback order
    per filing or per period rather than once for the whole history.

    Args:
        data: The ``facts`` sub-object of a companyfacts document.
        tags: Ordered ``(taxonomy, tag)`` fallbacks, highest priority first.
        unit: The unit key to read (``"USD"`` or ``"shares"``).

    Returns:
        A list of normalized fact dicts (see :func:`_norm_fact`), possibly empty.
    """
    return [_norm_fact(f, taxonomy, tag, unit)
            for taxonomy, tag in tags
            for f in data.get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, [])
            if f.get("val") is not None and f.get("end") and f.get("filed")]


def _as_instants(facts: list[dict]) -> list[dict]:
    """Turn weighted-average share counts into one instant per filing; keep instants (pure).

    A weighted-average count is a DURATION fact, and one filing reports several of them
    (the quarter, the year to date, prior-year comparatives). The fallback wants the
    freshest figure the filing knows, so per ``(tag, accession)`` it keeps the latest
    ``end`` and, at that end, the shortest period — the quarter, not the year to date —
    stamped as an instant at ``end``. Duration facts of every other tag are dropped: the
    instant concepts read point-in-time levels only.

    Args:
        facts: Normalized facts from :func:`_all_tagged`.

    Returns:
        The instant facts plus one converted weighted-average fact per filing and tag.
    """
    best: dict[tuple[str, str | None], dict] = {}
    for f in facts:
        if f["tag"] in _FUND_DURATION_SHARES and f["start"] is not None:
            key = (f["tag"], f["accn"])
            if key not in best or (f["end"], f["start"]) > (best[key]["end"], best[key]["start"]):
                best[key] = f
    return [f for f in facts if f["start"] is None] + [dict(f, start=None) for f in best.values()]


def _by_filing(facts: list[dict], tags: list[tuple[str, str]]) -> list[dict]:
    """Keep, per filing, only the facts of the highest-priority tag that filing reports (pure).

    One definition per filing, and the tag order decides which: a 10-K reporting both
    ``StockholdersEquity`` and the incl.-noncontrolling tag contributes the former, while
    a later filing that reports only the latter still contributes a balance sheet.

    Args:
        facts: Normalized facts carrying ``taxonomy:tag`` labels and accessions.
        tags: Ordered ``(taxonomy, tag)`` fallbacks, highest priority first.

    Returns:
        The facts whose tag is the best one reported in their own accession.
    """
    rank = {f"{taxonomy}:{tag}": i for i, (taxonomy, tag) in enumerate(tags)}
    best: dict[str | None, int] = {}
    for f in facts:
        best[f["accn"]] = min(best.get(f["accn"], len(tags)), rank[f["tag"]])
    return [f for f in facts if rank[f["tag"]] == best[f["accn"]]]


def _earliest(series: list[list[Fundamental]]) -> list[Fundamental]:
    """Merge per-tag row series so the earliest filing wins each period (pure).

    Rows are taken in filing order and kept only when their period advances past
    everything already kept — the monotone rule of :func:`_instant_rows`. Tag priority
    (the order of ``series``) only breaks a tie between rows filed the same day. Merging
    by priority instead was measured to lose periods and to adopt a proxy statement's
    figure: DEF 14A pay-versus-performance tables tag ``NetIncomeLoss``, and MA's FY2025
    net income came from the proxy filed ~75 days after the 10-K that first disclosed it.

    Args:
        series: One row list per tag, highest-priority tag first.

    Returns:
        One row per successively-newer period.
    """
    ranked = sorted(((i, r) for i, rows in enumerate(series) for r in rows),
                    key=lambda item: (item[1].filed, item[0], item[1].period_end))
    out: list[Fundamental] = []
    max_end: date | None = None
    for _, r in ranked:
        if max_end is None or r.period_end > max_end:
            max_end = r.period_end
            out.append(r)
    return out


def _norm_fact(f: dict, taxonomy: str, tag: str, unit: str) -> dict:
    """Normalize one raw XBRL fact into parsed dates + a ``taxonomy:tag`` label (pure)."""
    return {
        "start": date.fromisoformat(f["start"]) if f.get("start") else None,
        "end": date.fromisoformat(f["end"]),
        "val": float(f["val"]),
        "fy": f.get("fy"),
        "fp": f.get("fp"),
        "form": f.get("form"),
        "filed": date.fromisoformat(f["filed"]),
        "accn": f.get("accn"),
        "tag": f"{taxonomy}:{tag}",
        "unit": unit,
    }


def _known_at(filed: date) -> datetime:
    """Public-availability timestamp: the filing date at 21:00 UTC (after the US close)."""
    return datetime(filed.year, filed.month, filed.day, _FUND_KNOWN_HOUR, tzinfo=UTC)


def _row(symbol: str, concept: str, f: dict, *, val: float) -> Fundamental:
    """Build a :class:`~trading_simple.db.Fundamental` from a source fact + a value (pure)."""
    return Fundamental(
        symbol=symbol, concept=concept, tag=f["tag"], unit=f["unit"],
        period_start=None, period_end=f["end"], val=val,
        fy=f["fy"], fp=f["fp"], form=f["form"], filed=f["filed"],
        known_at=_known_at(f["filed"]), accn=f["accn"], source=_SOURCE,
    )


def _instant_rows(symbol: str, concept: str, raw: list[dict]) -> list[Fundamental]:
    """One row per successively-newer balance-sheet period, at first disclosure (pure).

    Processes instant facts (no ``start``) in filing order and keeps a fact only
    when its ``period_end`` advances beyond every period already emitted — so
    comparatives and re-reports of an already-known period are dropped, and the
    stored series rises monotonically in ``period_end``. That is exactly what the
    ``ffill`` feature builder needs: on any date it carries the most recent
    balance sheet knowable by then, with no regression to a restated prior period.

    Args:
        symbol: Bare symbol the rows are stored under.
        concept: Logical concept name (``equity`` | ``assets`` | ``shares``).
        raw: Normalized facts for the resolved tag.

    Returns:
        One :class:`~trading_simple.db.Fundamental` per newly-disclosed period.
    """
    facts = sorted((f for f in raw if f["start"] is None), key=lambda f: (f["filed"], f["end"]))
    rows: list[Fundamental] = []
    max_end: date | None = None
    for f in facts:
        if max_end is not None and f["end"] <= max_end:
            continue
        max_end = f["end"]
        rows.append(_row(symbol, concept, f, val=f["val"]))
    return rows


def _flow_rows(symbol: str, concept: str, raw: list[dict]) -> list[Fundamental]:
    """Roll income-statement flows to trailing-twelve-months, point-in-time (pure).

    XBRL reports income flows as fiscal-year-to-date cumulatives (Q1≈¼y, H1≈½y,
    9M≈¾y) plus a full-year (FY) figure. The TTM ending at period *P* in fiscal
    year *Y* is ``FY(Y-1) + YTD(Y, P) - YTD(Y-1, P)`` — which collapses to
    ``FY(Y)`` at year-end, removing the 10-K-vs-10-Q double count while using only
    directly-reported facts. When the prior-year cumulative needed for the delta
    is unavailable, it degrades gracefully to the latest full fiscal-year figure.
    Prior-year terms always come from earlier filings, so the emitted value uses
    no information filed after the period's own filing.

    Like :func:`_instant_rows`, emits one row per successively-newer period.

    Args:
        symbol: Bare symbol the rows are stored under.
        concept: Logical concept name (``ni_ttm`` | ``gp_ttm``).
        raw: Normalized flow facts (each carrying ``start`` and ``end``).

    Returns:
        One :class:`~trading_simple.db.Fundamental` (a TTM value) per period.
    """
    flows = [f for f in raw if f["start"] is not None]
    annuals = [f for f in flows if 340 <= (f["end"] - f["start"]).days <= 380]
    if not annuals:
        return []  # no full year to anchor the fiscal calendar / roll a TTM against

    # Fiscal-year-start anchors, keyed by DATE not by XBRL fy/fp: companyfacts stamps
    # each fact with the *filing's* fy/fp, so a prior-year comparative re-reported in a
    # later filing would collide with the current period under an (fy, qi) key and the
    # in-progress year (no annual filed yet) would be dropped. Each annual's start is an
    # anchor; so is the day after each annual's end (the next fiscal year's start, which
    # covers the in-progress year). Near-duplicates (start vs prior end+1) are collapsed.
    raw_anchors = sorted({f["start"] for f in annuals}
                         | {f["end"] + timedelta(days=1) for f in annuals})
    anchors: list[date] = []
    for a in raw_anchors:
        if not anchors or (a - anchors[-1]).days > 15:
            anchors.append(a)

    def fy_start(f: dict) -> date | None:
        """Snap a fact's start to its nearest fiscal-year-start anchor (else not a YTD)."""
        nearest = min(anchors, key=lambda a: abs((f["start"] - a).days))
        return nearest if abs((f["start"] - nearest).days) <= 7 else None

    def quarter(f: dict) -> int:  # 1..4 (4 == full year), from the duration
        return min(4, max(1, round((f["end"] - f["start"]).days / 91.3)))

    def prior(a: date) -> date | None:  # the fiscal-year anchor ~one year earlier
        cands = [x for x in anchors if 350 <= (a - x).days <= 380]
        return max(cands) if cands else None

    # As-reported cumulative value per (fiscal-year-start anchor, quarter); earliest
    # filing wins, so a later comparative never overwrites the original disclosure.
    cum: dict[tuple[date, int], float] = {}
    cum_fact: dict[tuple[date, int], dict] = {}
    annual: dict[date, float] = {}
    for f in sorted(flows, key=lambda f: f["filed"]):
        a = fy_start(f)
        if a is None:
            continue
        key = (a, quarter(f))
        if key not in cum:
            cum[key], cum_fact[key] = f["val"], f
        if 340 <= (f["end"] - f["start"]).days <= 380:
            annual.setdefault(a, f["val"])

    rows: list[Fundamental] = []
    max_end: date | None = None
    for f in sorted(cum_fact.values(), key=lambda f: (f["filed"], f["end"])):
        if max_end is not None and f["end"] <= max_end:
            continue
        a = fy_start(f)
        ttm = _ttm_value(a, quarter(f), cum, annual, prior(a))
        if ttm is None:
            continue
        max_end = f["end"]
        rows.append(_row(symbol, concept, f, val=ttm))
    return rows


def _ttm_value(a: date, qi: int, cum: dict[tuple[date, int], float],
               annual: dict[date, float], a_prev: date | None) -> float | None:
    """Trailing-twelve-month value for a fiscal period (pure).

    ``FY(prev) + YTD(a, qi) - YTD(prev, qi)``; the full year itself at ``qi == 4``.
    Falls back to the prior full fiscal year when the delta cannot be formed.

    Args:
        a: The period's fiscal-year-start anchor.
        qi: Quarter index 1..4 (4 == full fiscal year).
        cum: ``{(anchor, quarter): cumulative value}``.
        annual: ``{anchor: full-year value}``.
        a_prev: The prior fiscal year's anchor (or ``None`` if unknown).

    Returns:
        The TTM value, or ``None`` when nothing usable is available.
    """
    if qi >= 4:
        return annual.get(a, cum.get((a, 4)))
    if a_prev is None:
        return None
    prior_annual = annual.get(a_prev)
    ytd_cur = cum.get((a, qi))
    ytd_prior = cum.get((a_prev, qi))
    if prior_annual is not None and ytd_cur is not None and ytd_prior is not None:
        return prior_annual + ytd_cur - ytd_prior
    return prior_annual


def _synth_gross_profit(rev: list[dict], cogs: list[dict]) -> list[dict]:
    """Synthesize gross-profit facts as ``revenue - cogs`` per filing and period (pure).

    :func:`_parse_companyfacts` merges these with any tagged ``GrossProfit``, so they fill
    the periods the tag does not cover. Revenue and cost of revenue are matched on an
    identical ``(accession, start, end)``: a like-for-like figure from ONE filing, then fed
    through the same TTM roll-up. Matching on the period alone paired an older filing's
    revenue with the COGS a later 10-K re-presented for that period (discontinued
    operations, a reclassification) and stamped the result with the older filing date —
    a look-ahead that, measured on PH, changed 35 of 69 synthesized TTM values by up to 33%.

    Args:
        rev: Normalized revenue facts.
        cogs: Normalized cost-of-revenue facts.

    Returns:
        Synthetic gross-profit facts, each stamped with its revenue fact's filing.
    """
    if not rev or not cogs:
        return []
    cogs_by = {(c["accn"], c["start"], c["end"]): c for c in cogs if c["start"] is not None}
    out: list[dict] = []
    for r in rev:
        c = cogs_by.get((r["accn"], r["start"], r["end"]))
        if r["start"] is None or c is None:
            continue
        gp = dict(r)
        gp["val"] = r["val"] - c["val"]
        gp["tag"] = "computed:GrossProfit"
        out.append(gp)
    return out
