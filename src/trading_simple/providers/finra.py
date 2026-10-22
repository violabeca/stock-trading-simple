"""FINRA consolidated bi-monthly equity short interest (free).

Short interest / days-to-cover is a well-replicated cross-sectional signal
(Hong, Li, Wang & Zhao: days-to-cover long-short ~1.19%/mo), but the tradable
part is *avoidance*, not shorting: much of the short-leg alpha accrues to the
lender as a borrow fee (Drechsler-Drechsler), and the value-weighted effect is
weak (Asquith-Pathak-Ritter). So this strategy uses low days-to-cover as a
long-only favourable tilt and down-weights high-DTC names — never a fast trigger
(the data is bi-monthly and ~8 business days stale at dissemination).

Two ingestion paths, both parsed by :func:`_records_to_rows` (pure):

* :func:`iter_short_interest_bulk` — the DEFAULT for a backfill. FINRA publishes one
  pipe-delimited file per settlement date at ``cdn.finra.org`` carrying EVERY symbol
  with no auth. This scales: the Query API's symbol ``domainFilter`` times out (504)
  on a full universe, whereas a bulk file is one clean ~2 MB download per settlement.
* :func:`fetch_short_interest` — the Query API, for small/targeted symbol lists.

Point-in-time: a settlement's figures are only knowable at FINRA's dissemination
date (~8 business days after settlement, after 16:00 ET), so ``known_at`` is set
there, never at the settlement date.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time

import pandas as pd

from trading_simple.db import ShortInterest

log = logging.getLogger(__name__)

_SOURCE = "finra"
_API_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
_BULK_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{ymd}.csv"
# A few always-shorted mega-caps whose settlement coverage is complete — used only to
# enumerate FINRA's dissemination schedule via one SMALL Query-API request (a large
# symbol filter is what 504s; a 5-symbol probe does not).
_SCHEDULE_PROBE = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"]
_PAGE = 5000  # FINRA's documented synchronous max per request (was 1000): ~5x fewer requests
# FINRA disseminates consolidated short interest ~8 business days after the
# settlement date, after the close; act only from the next session.
_DISSEM_BDAYS = 8
_DISSEM_UTC = time(21, 0)  # ~16:00-17:00 ET, safely after the US close


def _si_request_body(symbols: list[str], since: date, until: date | None, offset: int) -> dict:
    """Build the FINRA otcMarket query body (pure; unit-testable).

    Filters by symbol and settlement date; a non-null ``until`` adds an inclusive
    upper bound so only the missing early gap of an already-covered range is
    queried.

    Args:
        symbols: Bare canonical tickers to fetch.
        since: Earliest settlement date (``gte``).
        until: Latest settlement date (``lte``), or ``None`` for no upper bound.
        offset: Paging offset.

    Returns:
        The JSON request body.
    """
    compare = [{"fieldName": "settlementDate", "compareType": "gte",
                "fieldValue": since.isoformat()}]
    if until is not None:
        compare.append({"fieldName": "settlementDate", "compareType": "lte",
                        "fieldValue": until.isoformat()})
    return {
        "limit": _PAGE,
        "offset": offset,
        "compareFilters": compare,
        "domainFilters": [
            {"fieldName": "symbolCode", "values": [s.upper() for s in symbols]},
        ],
    }


def fetch_short_interest(symbols: list[str], since: date,
                         *, until: date | None = None) -> list[ShortInterest]:
    """Fetch consolidated short-interest observations for ``symbols`` since ``since``.

    Queries FINRA's free otcMarket Query API, filtering by symbol and settlement
    date, paging until exhausted, and parses each record into a
    :class:`~trading_simple.db.ShortInterest` row with a point-in-time
    ``known_at`` (dissemination) timestamp.

    Args:
        symbols: Bare canonical tickers to fetch.
        since: Earliest settlement date to include.
        until: Latest settlement date to include (inclusive); ``None`` means no
            upper bound. Used to query only the missing early gap of an
            already-covered range.

    Returns:
        Parsed short-interest rows (empty if FINRA returns nothing).
    """
    import requests

    rows: list[ShortInterest] = []
    offset = 0
    while True:
        body = _si_request_body(symbols, since, until, offset)
        resp = requests.post(_API_URL, json=body,
                             headers={"Accept": "application/json"}, timeout=60)
        resp.raise_for_status()
        records = resp.json()
        if not records:
            break
        rows.extend(_records_to_rows(records))
        if len(records) < _PAGE:
            break
        offset += _PAGE
    return rows


#: A bulk file matching none of this many requested symbols is a format change, not a quiet
#: settlement: every listed stock with a FINRA short position appears in every file.
_MIN_MATCHED = 50


def settlement_dates(since: date, until: date | None = None) -> list[date]:
    """List FINRA short-interest settlement dates in ``[since, until]`` (inclusive).

    Enumerated from ONE small Query-API request over a handful of always-covered
    mega-caps (:data:`_SCHEDULE_PROBE`) — reliable and fast, unlike a full-universe
    ``domainFilter`` (which 504s). These are exactly the dates whose bulk files exist,
    so callers never have to guess the bi-monthly calendar or handle 403s on bad dates.

    Args:
        since: Earliest settlement date to include.
        until: Latest settlement date to include (inclusive), or ``None`` for no bound.

    Returns:
        Sorted settlement dates (empty if FINRA returns nothing).
    """
    import requests

    body = _si_request_body(_SCHEDULE_PROBE, since, until, 0)
    resp = requests.post(_API_URL, json=body, headers={"Accept": "application/json"}, timeout=60)
    resp.raise_for_status()
    return sorted({date.fromisoformat(r["settlementDate"][:10])
                   for r in resp.json() if r.get("settlementDate")})


def _parse_bulk_file(text: str) -> list[dict]:
    """Parse a FINRA bulk short-interest file into records (pure).

    The file is **pipe-delimited** despite its ``.csv`` name, and its columns use the
    SAME field names as the Query API (``symbolCode`` / ``settlementDate`` /
    ``currentShortPositionQuantity`` / ``averageDailyVolumeQuantity`` /
    ``daysToCoverQuantity``), so :func:`_records_to_rows` consumes the output directly.

    ``QUOTE_NONE`` is load-bearing: the format has **no quoting convention**, so a ``"``
    in an issuer name is data, not a delimiter. Without it one row like
    ``DOD|"ELEMENTS ""Dogs of the Dow""|...`` opens a quoted field that never closes, and
    :mod:`csv` swallows the rest of the file into a single value — raising
    ``field larger than field limit`` on a file whose longest line is 278 characters.
    That is how the 2018-2019 short-interest history came to be missing: the download
    aborted mid-backfill and the gap sat there looking like "FINRA has no data".
    Do **not** "fix" a recurrence by raising :func:`csv.field_size_limit` — that silently
    turns 15,398 rows into 3,881, which is the same quiet-failure mode this repo has
    already shipped once.

    From the 2026-09-15 settlement on, FINRA QUOTES every field (``"AAPL"|"Apple Inc."|…``).
    Read under ``QUOTE_NONE`` the quotes stay in every key and value, ``symbolCode`` is never
    found, and every file parses into rows that match no symbol — zero rows, no error (short
    interest sat on the 2026-08-31 settlement until 2026-10-20). So the convention is read
    from the header: a quoted file is standard CSV, an unquoted one keeps ``QUOTE_NONE``.

    Args:
        text: The downloaded file body.

    Returns:
        One dict per data row.
    """
    import csv
    import io

    quoted = text.lstrip("\ufeff").startswith('"')
    return list(csv.DictReader(io.StringIO(text), delimiter="|",
                               quoting=csv.QUOTE_MINIMAL if quoted else csv.QUOTE_NONE))


def iter_short_interest_bulk(symbols: list[str], since: date, *, until: date | None = None):
    """Yield ``(settlement_date, [ShortInterest])`` per settlement from the bulk CDN files.

    One pipe-delimited file per settlement date carries EVERY symbol with no auth and no
    symbol filter, so this replaces :func:`fetch_short_interest` for a full-universe
    backfill (whose large ``domainFilter`` times out). Enumerates dates via
    :func:`settlement_dates`, downloads each settlement's file, filters to ``symbols``,
    and reuses :func:`_records_to_rows`. Yielding per settlement keeps memory bounded.

    Args:
        symbols: Bare canonical tickers to keep.
        since: Earliest settlement date to include.
        until: Latest settlement date to include (inclusive); ``None`` means no bound.

    Yields:
        ``(settlement_date, rows)`` per settlement (a missing file is warned and skipped).

    Raises:
        ValueError: A file without a ``symbolCode`` column, or one that matches none of
            :data:`_MIN_MATCHED` or more requested symbols (a format change, not a quiet
            settlement).
    """
    import requests

    symset = {s.upper() for s in symbols}
    for settle in settlement_dates(since, until):
        resp = requests.get(_BULK_URL.format(ymd=settle.strftime("%Y%m%d")), timeout=60)
        if resp.status_code != 200:
            log.warning("FINRA: no bulk file for %s (HTTP %s)", settle, resp.status_code)
            continue
        recs = _parse_bulk_file(resp.text)
        # A changed format reads as "no short interest", never as an error: fail loudly, so the
        # caller leaves the settlement unrecorded and the next run fetches it again.
        if recs and "symbolCode" not in recs[0]:
            raise ValueError(f"FINRA {settle}: unreadable bulk file (columns {list(recs[0])[:3]})")
        recs = [r for r in recs if (r.get("symbolCode") or "").upper() in symset]
        if not recs and len(symset) >= _MIN_MATCHED:
            raise ValueError(f"FINRA {settle}: none of {len(symset)} symbols in the bulk file")
        yield settle, _records_to_rows(recs)


def _dissemination(settlement: date) -> datetime:
    """Return the point-in-time ``known_at`` for a settlement date.

    FINRA publishes ~8 business days after settlement, after the close, so the
    figure is first actionable at that evening (execution then happens next-bar
    open, as everywhere in this repo).

    Args:
        settlement: The short-interest settlement date.

    Returns:
        A UTC datetime ~8 business days after ``settlement``.
    """
    disseminated = (pd.Timestamp(settlement) + pd.offsets.BDay(_DISSEM_BDAYS)).date()
    return datetime.combine(disseminated, _DISSEM_UTC, tzinfo=UTC)


def _records_to_rows(records: list[dict]) -> list[ShortInterest]:
    """Convert FINRA short-interest JSON records into :class:`ShortInterest` rows (pure).

    Reads the consolidated short-interest fields (``symbolCode``,
    ``settlementDate``, ``currentShortPositionQuantity``,
    ``averageDailyVolumeQuantity``, ``daysToCoverQuantity``) and derives
    days-to-cover from the reported short interest and ADV when FINRA omits it.

    Args:
        records: JSON records from the FINRA Query API.

    Returns:
        One :class:`ShortInterest` per record with a parseable settlement date.
    """
    rows: list[ShortInterest] = []
    for rec in records:
        settle_raw = rec.get("settlementDate")
        symbol = rec.get("symbolCode")
        if not settle_raw or not symbol:
            continue
        settlement = date.fromisoformat(settle_raw[:10])
        si = _num(rec.get("currentShortPositionQuantity"))
        adv = _num(rec.get("averageDailyVolumeQuantity"))
        dtc = _num(rec.get("daysToCoverQuantity"))
        if dtc is None and si is not None and adv:
            dtc = si / adv
        disseminated = _dissemination(settlement)
        rows.append(ShortInterest(
            symbol=str(symbol).upper(),
            settlement_date=settlement,
            short_interest=si,
            avg_daily_volume=adv,
            days_to_cover=dtc,
            disseminated_at=disseminated.date(),
            known_at=disseminated,
            source=_SOURCE,
        ))
    return rows


def _num(value) -> float | None:
    """Parse a FINRA numeric field (string or number) to float, else None."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
