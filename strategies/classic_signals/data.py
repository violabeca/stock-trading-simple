"""Data for the classic_signals strategy: 1-minute candles of the stocks, their sector funds and the
S&P 500 fund, daily candles, and quarterly report (earnings) dates.

Candles come from the shared database (the 1-minute candles were downloaded from Interactive
Brokers, regular trading hours only). Earnings dates come from Yahoo Finance and are kept in a
small file under ``var/classic_signals/``.

Times: the database stores every candle by its OPENING time in UTC. Here every minute candle is
indexed by its opening time in New York, so 09:30 is the first candle of a regular session and
15:59 the last. Daily candles are indexed by their session date (no time, no time zone).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VAR = HERE.parents[1] / "var" / "classic_signals"   # git-ignored: cached earnings dates
CHAINS = HERE.parents[1] / "var" / "marketdata" / "chains"   # end-of-day option chains
NY = "America/New_York"
SESSION_MINUTES = 390        # 09:30 to 16:00
HALF_DAY_MINUTES = 210       # 09:30 to 13:00 (the day before Independence Day, Black Friday ...)
#: Sessions that close at 13:00. Their stored candles after 13:00 carry a "regular hours" label
#: (it is set from the clock alone): the loaders treat them as after-hours.
HALF_DAYS = frozenset(["2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"])
#: The parts of a New York day, in minutes after midnight: the pre-market opens at 04:00, the
#: regular session runs 09:30-16:00 (13:00 on a half-day), the after-hours end at 20:00, and the
#: overnight session (special venues only, Sunday to Thursday nights) runs 20:00-04:00.
PRE_OPEN, REGULAR_OPEN, REGULAR_CLOSE, HALF_DAY_CLOSE, POST_CLOSE = 240, 570, 960, 780, 1200
PARTS = ("pre-market", "regular", "after-hours", "overnight")

#: The stocks this strategy explores, each with the fund that stands for its sector.
#: ServiceNow: software (IGV); GE Vernova: industrials (XLI); Rigetti Computing: quantum
#: computing (QTUM, the Defiance Quantum ETF, which holds it at about 2%).
SECTOR = {"NOW": "IGV", "GEV": "XLI", "RGTI": "QTUM"}
#: Every stock the tuning notebook may pick, each with the fund that stands for its sector (the
#: sector trend filter reads its daily trend): the three above, and a group of eleven (2026-10-18)
#: with the SPDR sector funds (financials XLF, energy XLE, consumer discretionary XLY, health care
#: XLV, technology XLK) and, for the uranium miner Cameco, the Global X Uranium ETF (URA).
SECTOR_FUND = {**SECTOR, "LNC": "XLF", "JEF": "XLF", "COF": "XLF", "FTI": "XLE", "KMX": "XLY",
               "EXPE": "XLY", "GEHC": "XLV", "CCJ": "URA", "FLEX": "XLK"}
MARKET = "SPY"                          # the S&P 500 fund: the market as a whole


def symbols(stocks: dict[str, str] = SECTOR, market: str = MARKET) -> list[str]:
    """Every symbol the strategy reads: the stocks, their sector funds and the market fund."""
    return list(dict.fromkeys([*stocks, *stocks.values(), market]))


def _day_after(day) -> pd.Timestamp:
    """Midnight UTC after ``day``: ``load_panel`` reads up to a timestamp, and a bare date would be
    00:00 UTC of that day, which drops the whole session."""
    return pd.Timestamp(day) + pd.Timedelta(days=1)


def last_stamp(end, bar: str) -> pd.Timestamp:
    """The last database timestamp that belongs to the session ``end``. Minute candles of a
    session are stamped during its day (13:30-20:59 UTC), so read up to midnight UTC after it.
    A DAILY candle is stamped 00:00 UTC of its own date, so midnight UTC after ``end`` would
    already be the NEXT session's daily candle: read up to ``end`` itself."""
    return pd.Timestamp(end) if bar == "1d" else _day_after(end)


def drop_after_half_day_close(minutes: pd.DataFrame) -> pd.DataFrame:
    """The candles without those at or after 13:00 on a half-day (:data:`HALF_DAYS`)."""
    t = minutes.index
    late = np.isin(t.strftime("%Y-%m-%d"), list(HALF_DAYS)) & (t.hour >= 13)
    return minutes[~late]


def session_part(index: pd.DatetimeIndex) -> np.ndarray:
    """The part of the day of every New York minute (:data:`PARTS`): 0 pre-market (04:00-09:30),
    1 regular session (09:30-16:00, 09:30-13:00 on a half-day), 2 after-hours (until 20:00),
    3 overnight (20:00-04:00). Set from the clock, so a half-day's afternoon is after-hours
    whatever label the database carries."""
    minute = np.asarray(index.hour * 60 + index.minute)
    half = np.isin(index.strftime("%Y-%m-%d"), list(HALF_DAYS))
    close = np.where(half, HALF_DAY_CLOSE, REGULAR_CLOSE)
    return np.select([(minute < PRE_OPEN) | (minute >= POST_CLOSE), minute < REGULAR_OPEN,
                      minute < close], [3, 0, 1], 2)


def load_minutes(engine, names: list[str], start, end, *,
                 extended: bool = False) -> dict[str, pd.DataFrame]:
    """The 1-minute candles of every symbol from ``start`` to ``end`` (both days included).

    Args:
        engine: Database engine.
        names: Symbols to load.
        start: First day, e.g. ``"2025-01-01"``.
        end: Last day.
        extended: Also every candle outside the regular session: pre-market (04:00-09:30),
            after-hours (until 20:00) and overnight (20:00-04:00, where stored); else the
            regular session only.

    Returns:
        ``{symbol: candles}``; each frame is indexed by the candle's New York opening time and has
        ``open``, ``high``, ``low``, ``close``, ``volume``, ``traded`` (False for a minute
        without a single trade: Interactive Brokers then repeats the last price with no volume)
        and ``regular`` (True inside the regular session, :func:`session_part`).
    """
    from trading_simple.data import load_panel

    panel = load_panel(engine, list(names), "1m", pd.Timestamp(start), last_stamp(end, "1m"),
                       session="both" if extended else "regular")
    out = {}
    for name in names:
        if name not in panel.symbols:
            raise ValueError(f"no 1-minute candles for {name} between {start} and {end}")
        frame = pd.DataFrame({"open": panel.open_[name], "high": panel.high[name],
                              "low": panel.low[name], "close": panel.close[name],
                              "volume": panel.volume[name]}).dropna(subset=["close"])
        frame.index = frame.index.tz_convert(NY)
        frame.index.name = "time"
        frame["traded"] = frame["volume"] > 0
        frame["regular"] = session_part(frame.index) == 1
        out[name] = frame if extended else frame[frame["regular"]]
    return out


def load_days(engine, names: list[str], start, end) -> dict[str, pd.DataFrame]:
    """The stored daily candles of every symbol from ``start`` to ``end`` (both days included).

    Daily candles are stored at 00:00 UTC of their session date, so the UTC date IS the session
    date (in New York time that moment is still the evening before).

    Returns:
        ``{symbol: candles}`` indexed by session date, with ``open``, ``high``, ``low``,
        ``close`` and ``volume``.
    """
    from trading_simple.data import load_panel

    panel = load_panel(engine, list(names), "1d", pd.Timestamp(start), last_stamp(end, "1d"),
                       session="regular")
    out = {}
    for name in names:
        if name not in panel.symbols:
            raise ValueError(f"no daily candles for {name} between {start} and {end}")
        frame = pd.DataFrame({"open": panel.open_[name], "high": panel.high[name],
                              "low": panel.low[name], "close": panel.close[name],
                              "volume": panel.volume[name]}).dropna(subset=["close"])
        frame.index = frame.index.tz_convert("UTC").tz_localize(None).normalize()
        frame.index.name = "day"
        out[name] = frame
    return out


def session_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The session date of every New York minute (midnight, no time zone)."""
    return index.tz_localize(None).normalize()


def daily_from_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    """Daily candles built from 1-minute candles: the first open, the highest high, the lowest
    low, the last close and the summed volume of each session, plus its number of candles."""
    g = minutes.groupby(session_day(minutes.index))
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(), "candles": g.size()})
    out.index.name = "day"
    return out


def resample(minutes: pd.DataFrame, size: int) -> pd.DataFrame:
    """Candles of ``size`` minutes built from 1-minute candles. Each part of the day is cut into
    blocks counted from its own start (04:00 for the pre-market, 09:30 for the regular session,
    16:00 or a half-day's 13:00 for the after-hours, 20:00 and midnight for the overnight), so no
    candle mixes two parts or crosses midnight; the last block of a part can be shorter (the
    15:30 block of 60-minute candles holds 30 minutes).

    Args:
        minutes: 1-minute candles (:func:`load_minutes`).
        size: Minutes per candle (1 returns the input).

    Returns:
        The candles, indexed by their opening time, with the same columns.
    """
    if size == 1:
        return minutes
    t = minutes.index
    minute = np.asarray(t.hour * 60 + t.minute)
    part = session_part(t)
    close = np.where(np.isin(t.strftime("%Y-%m-%d"), list(HALF_DAYS)), HALF_DAY_CLOSE,
                     REGULAR_CLOSE)
    anchor = np.select([part == 0, part == 1, part == 2, minute >= POST_CLOSE],
                       [PRE_OPEN, REGULAR_OPEN, close, POST_CLOSE], 0)
    start = t.normalize() + pd.to_timedelta(anchor + (minute - anchor) // size * size, unit="min")
    g = minutes.groupby(start)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(), "traded": g["traded"].any()})
    if "regular" in minutes:
        out["regular"] = g["regular"].first()
    out.index.name = "time"
    return out


def coverage_counts(engine, names: list[str], start, end) -> pd.DataFrame:
    """How many 1-minute candles each symbol has on each session, read straight from the
    database as counts (no prices are loaded, so this can look at every month, the final-check
    months included, without showing what prices did there).

    The regular session is counted in the frame itself; the pre-market, the after-hours and the
    overnight in ``counts.attrs`` (``pre``, ``post``, ``night`` and how many of those minutes had
    a trade, ``pre_traded``, ``post_traded``, ``night_traded``; overnight minutes count on the
    calendar day they fall on). Candles stored after a half-day's 13:00 close count as after-hours;
    how many per symbol is in ``counts.attrs["relabelled"]``.

    Returns:
        Sessions x symbols (regular-session candles), 0 where a symbol has no candle on a day
        another symbol traded.
    """
    from sqlalchemy import text

    ny = "(ts AT TIME ZONE 'America/New_York')"
    clock = f"{ny}::time"
    pre = f"NOT rth AND {clock} >= '04:00' AND {clock} < '09:30'"
    post = f"NOT rth AND {clock} >= '16:00' AND {clock} < '20:00'"
    night = f"NOT rth AND ({clock} >= '20:00' OR {clock} < '04:00')"
    late = f"rth AND extract(hour FROM {ny}) >= 13"
    count = lambda cond, name: (f"sum(CASE WHEN {cond} THEN 1 ELSE 0 END) AS {name}, "  # noqa: E731
                                f"sum(CASE WHEN {cond} AND volume > 0 THEN 1 ELSE 0 END) "
                                f"AS {name}_traded")
    with engine.connect() as c:
        rows = pd.read_sql(text(
            f"SELECT symbol, {ny}::date AS day, {count('rth', 'n')}, {count(late, 'late')}, "
            f"{count(pre, 'pre')}, {count(post, 'post')}, {count(night, 'night')} "
            "FROM prices_ohlcv WHERE bar = '1m' "
            "AND symbol = ANY(CAST(:s AS text[])) AND ts >= :a AND ts < :b GROUP BY symbol, day"),
            c, params={"s": list(names), "a": pd.Timestamp(start), "b": last_stamp(end, "1m")})
    rows["day"] = pd.to_datetime(rows["day"])
    half = rows["day"].dt.strftime("%Y-%m-%d").isin(HALF_DAYS)
    for kind in ("", "_traded"):
        moved = rows[f"late{kind}"].where(half, 0)
        rows[f"n{kind}"] -= moved
        rows[f"post{kind}"] += moved
    wide = lambda col: rows.pivot(index="day", columns="symbol", values=col).reindex(  # noqa: E731
        columns=list(names)).fillna(0).astype(int)
    counts = wide("n")
    keep = counts.sum(axis=1) > 0                  # a session: something traded 09:30-16:00
    counts = counts[keep]
    counts.attrs["traded"] = wide("n_traded")[keep]
    for col in ("pre", "pre_traded", "post", "post_traded", "night", "night_traded"):
        counts.attrs[col] = wide(col)[keep]
    counts.attrs["relabelled"] = rows["late"].where(half, 0).groupby(rows["symbol"]).sum() \
        .reindex(list(names)).fillna(0).astype(int)
    return counts


def data_check(counts: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol: the first and last session, how many sessions are complete (390
    candles), half-days (210 candles: the market closed at 13:00), short (some minutes missing)
    or missing altogether (other symbols traded that day), the share of regular-session minutes
    without a single trade, and for the pre-market, the after-hours and the overnight (where
    stored) their typical number of minutes per session and the share of those with a trade.

    Args:
        counts: :func:`coverage_counts`.
    """
    a = counts.attrs
    rows = []
    for name in counts.columns:
        n = counts[name]
        have = n[n > 0]
        row = {
            "symbol": name,
            "first session": have.index.min().date() if len(have) else None,
            "last session": have.index.max().date() if len(have) else None,
            "sessions": int(len(have)),
            "complete (390)": int((n == SESSION_MINUTES).sum()),
            "half-days (210)": int((n == HALF_DAY_MINUTES).sum()),
            "short": int(((n > 0) & (n != SESSION_MINUTES) & (n != HALF_DAY_MINUTES)).sum()),
            "missing": int((n == 0).sum()),
            "minutes without a trade": (float(1 - a["traded"][name].sum() / n.sum())
                                        if "traded" in a and n.sum() else np.nan)}
        for part, word in (("pre", "pre-market"), ("post", "after-hours"), ("night", "overnight")):
            if part in a and a[part][name].sum():
                row[f"{word} minutes per session"] = float(a[part][name][n > 0].median())
                row[f"{word} minutes with a trade"] = float(
                    a[f"{part}_traded"][name].sum() / a[part][name].sum())
        row["half-day candles after 13:00 (counted as after-hours)"] = int(
            a["relabelled"].get(name, 0)) if "relabelled" in a else 0
        rows.append(row)
    return pd.DataFrame(rows)


def load_chains(symbol: str, start, end, *, folder: Path = CHAINS) -> pd.DataFrame:
    """A stock's end-of-day option chains between ``start`` and ``end`` (MarketData.app, written
    by ``scripts/marketdata_dump.py`` as ``<folder>/<symbol>/<first>_<last>.csv.gz``): one row per
    contract and day with its expiry, side, strike, days to expiry, bid and ask, the stock's price,
    open interest and volume (these history files carry no implied volatility). Empty when none
    are stored."""
    cols = ["date", "expiration", "side", "strike", "dte", "bid", "ask", "underlyingPrice",
            "openInterest", "volume"]
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    files = [f for f in sorted((Path(folder) / symbol).glob("*_*.csv.gz"))
             if pd.Timestamp(f.name[:10]) <= hi and pd.Timestamp(f.name[11:21]) >= lo]
    if not files:
        return pd.DataFrame(columns=cols)
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d[d["date"].between(lo, hi)].reset_index(drop=True)


def earnings_dates(stocks: list[str], *, refresh: bool = False) -> pd.DataFrame:
    """Quarterly report dates of the stocks (Yahoo Finance), past and upcoming, kept in
    ``var/classic_signals/earnings.csv`` (``refresh`` asks Yahoo again).

    A report before the open moves the stock in that day's session; one after the close moves it
    in the NEXT session. ``reaction day`` is that first session.

    Returns:
        ``symbol``, ``time`` (New York), ``when`` ("before the open", "after the close" or
        "during the day") and ``reaction day``; empty if Yahoo cannot be reached.
    """
    path = VAR / "earnings.csv"
    if path.exists() and not refresh:
        out = pd.read_csv(path)
        if set(stocks) <= set(out["symbol"]):
            out["time"] = pd.to_datetime(out["time"], utc=True).dt.tz_convert(NY)
            out["reaction day"] = pd.to_datetime(out["reaction day"])
            return out[out["symbol"].isin(stocks)].reset_index(drop=True)
    import yfinance as yf

    rows = []
    for name in stocks:
        try:
            frame = yf.Ticker(name).get_earnings_dates(limit=20)
        except Exception as exc:  # noqa: BLE001 (Yahoo fails in many ways; the strategy goes on)
            print(f"earnings dates for {name}: Yahoo did not answer ({exc!r})")
            continue
        if frame is None or frame.empty:
            continue
        for when in frame.index:
            t = pd.Timestamp(when)
            t = t.tz_localize(NY) if t.tzinfo is None else t.tz_convert(NY)
            minute = t.hour * 60 + t.minute
            kind = ("before the open" if minute < 570 else "after the close" if minute >= 960
                    else "during the day")
            day = t.tz_localize(None).normalize()
            react = day + pd.offsets.BDay(1) if kind == "after the close" else day
            rows.append({"symbol": name, "time": t, "when": kind, "reaction day": react})
    out = pd.DataFrame(rows, columns=["symbol", "time", "when", "reaction day"])
    if len(out):
        VAR.mkdir(parents=True, exist_ok=True)
        out.sort_values(["symbol", "time"]).to_csv(path, index=False)
    return out.sort_values(["symbol", "time"]).reset_index(drop=True)
