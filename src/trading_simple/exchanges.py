"""Exchange registry: how each trading venue names, prices, and times its bars.

One symbol can trade on different venues with different currencies, ticker
suffixes, and session hours. This module is the single source of truth for that:
it tells the yfinance provider which suffix to use (``NESN`` -> ``NESN.SW``),
the IBKR provider which contract to request (``EBS``/``CHF``), and everyone else
whether a bar falls inside the venue's regular trading hours (``rth``).

Everything here is pure stdlib (``zoneinfo``), so it is unit-testable with no
network or database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

#: Bars finer than a day belong to a session; daily bars are the regular candle.
INTRADAY_BARS = frozenset({"1m", "5m", "1h"})


@dataclass(frozen=True, slots=True)
class ExchangeSpec:
    """Everything the framework needs to know about one trading venue.

    Attributes:
        code: Canonical key, e.g. ``"US"`` or ``"SIX"``.
        name: Human-readable venue name.
        ib_exchange: IBKR contract exchange, e.g. ``"SMART"`` or ``"EBS"``.
        currency: ISO currency of the venue, e.g. ``"USD"``.
        tz: IANA timezone name, e.g. ``"America/New_York"``.
        rth_open: Regular session open, local wall-clock time.
        rth_close: Regular session close, local wall-clock time.
        yf_suffix: yfinance ticker suffix (``".SW"`` for SIX, ``""`` for US).
    """

    code: str
    name: str
    ib_exchange: str
    currency: str
    tz: str
    rth_open: time
    rth_close: time
    yf_suffix: str = ""

    @property
    def zoneinfo(self) -> ZoneInfo:
        """The venue's timezone as a :class:`zoneinfo.ZoneInfo`."""
        return ZoneInfo(self.tz)


_US = ExchangeSpec(
    code="US", name="US equities (NYSE/Nasdaq)", ib_exchange="SMART", currency="USD",
    tz="America/New_York", rth_open=time(9, 30), rth_close=time(16, 0),
)
_SIX = ExchangeSpec(
    # SIX has no retail extended-hours feed; every SIX intraday bar is regular-session.
    code="SIX", name="SIX Swiss Exchange", ib_exchange="EBS", currency="CHF",
    tz="Europe/Zurich", rth_open=time(9, 0), rth_close=time(17, 30), yf_suffix=".SW",
)

#: All known venues. Add a new one by appending a spec — nothing else changes.
REGISTRY: dict[str, ExchangeSpec] = {_US.code: _US, _SIX.code: _SIX}

DEFAULT_EXCHANGE = "US"

#: Symbols whose primary listing is not the US default (Swiss blue chips), so
#: they resolve with zero configuration. The ``exchanges:`` map in
#: ``config/settings.yaml`` wins over this built-in list.
BUILTIN_SYMBOL_EXCHANGE: dict[str, str] = {
    sym: "SIX"
    for sym in (
        "NESN", "NOVN", "ROG", "UHR", "UHRN", "ZURN", "ABBN", "CFR",
        "LONN", "SIKA", "GIVN", "SLHN", "SREN", "GEBN", "PGHN", "SCMN",
        "BAER", "ALC", "SOON", "LOGN", "STMN", "TEMN", "VACN", "HOLN",
        "KNIN", "SGSN",
    )
}


def resolve_exchange(symbol: str, overrides: dict[str, str] | None = None) -> ExchangeSpec:
    """Resolve a bare symbol to its trading venue.

    Priority: caller ``overrides`` (from ``settings.exchanges``) -> built-in
    Swiss list -> US default. Matching is case-insensitive.

    Args:
        symbol: Bare canonical symbol, e.g. ``"AAPL"`` or ``"NESN"``.
        overrides: Optional per-symbol venue map, e.g. ``{"NESN": "SIX"}``.

    Returns:
        The venue's :class:`ExchangeSpec`.

    Raises:
        KeyError: If a mapped venue code is not in :data:`REGISTRY`.
    """
    sym = symbol.upper()
    code = None
    if overrides:
        code = next((v for k, v in overrides.items() if k.upper() == sym), None)
    if code is None:
        code = BUILTIN_SYMBOL_EXCHANGE.get(sym, DEFAULT_EXCHANGE)
    try:
        return REGISTRY[code]
    except KeyError as exc:
        raise KeyError(f"Unknown exchange code {code!r}. Known: {sorted(REGISTRY)}") from exc


def is_rth(ts_utc, spec: ExchangeSpec, bar: str) -> bool:
    """Whether a bar belongs to the venue's regular trading hours.

    Daily bars are the regular-session candle, so they are always ``True``.
    Intraday bars compare their *local* wall-clock time to
    ``[rth_open, rth_close)`` — localizing first makes this DST-safe.

    Args:
        ts_utc: Timezone-aware bar-open timestamp (pandas or stdlib).
        spec: The venue the symbol trades on.
        bar: Bar size, e.g. ``"1d"`` or ``"1m"``.

    Returns:
        True for regular-session (or daily) bars, False for pre/post-market.
    """
    if bar not in INTRADAY_BARS:
        return True
    local = (ts_utc.tz_convert(spec.tz) if hasattr(ts_utc, "tz_convert")
             else ts_utc.astimezone(spec.zoneinfo))
    return spec.rth_open <= local.time() < spec.rth_close
