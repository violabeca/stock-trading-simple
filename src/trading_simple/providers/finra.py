"""FINRA consolidated bi-monthly equity short interest (free Query API).

Short interest / days-to-cover is a well-replicated cross-sectional signal
(Hong, Li, Wang & Zhao: days-to-cover long-short ~1.19%/mo), but the tradable
part is *avoidance*, not shorting: much of the short-leg alpha accrues to the
lender as a borrow fee (Drechsler-Drechsler), and the value-weighted effect is
weak (Asquith-Pathak-Ritter). So this strategy uses low days-to-cover as a
long-only favourable tilt and down-weights high-DTC names — never a fast trigger
(the data is bi-monthly and ~8 business days stale at dissemination).

Point-in-time: a settlement's figures are only knowable at FINRA's dissemination
date (~8 business days after settlement, after 16:00 ET), so ``known_at`` is set
there, never at the settlement date. Network fetch (:func:`fetch_short_interest`)
is separated from parsing (:func:`_records_to_rows`, pure).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time

import pandas as pd

from trading_simple.db import ShortInterest

log = logging.getLogger(__name__)

_SOURCE = "finra"
_API_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
_PAGE = 1000
# FINRA disseminates consolidated short interest ~8 business days after the
# settlement date, after the close; act only from the next session.
_DISSEM_BDAYS = 8
_DISSEM_UTC = time(21, 0)  # ~16:00-17:00 ET, safely after the US close


def fetch_short_interest(symbols: list[str], since: date) -> list[ShortInterest]:
    """Fetch consolidated short-interest observations for ``symbols`` since ``since``.

    Queries FINRA's free otcMarket Query API, filtering by symbol and settlement
    date, paging until exhausted, and parses each record into a
    :class:`~trading_simple.db.ShortInterest` row with a point-in-time
    ``known_at`` (dissemination) timestamp.

    Args:
        symbols: Bare canonical tickers to fetch.
        since: Earliest settlement date to include.

    Returns:
        Parsed short-interest rows (empty if FINRA returns nothing).
    """
    import requests

    rows: list[ShortInterest] = []
    offset = 0
    while True:
        body = {
            "limit": _PAGE,
            "offset": offset,
            "compareFilters": [
                {"fieldName": "settlementDate", "compareType": "gte",
                 "fieldValue": since.isoformat()},
            ],
            "domainFilters": [
                {"fieldName": "symbolCode", "values": [s.upper() for s in symbols]},
            ],
        }
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
