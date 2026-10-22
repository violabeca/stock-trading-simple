"""overnight_open research helpers: the night before a session, and its first trading hour.

Loaded by path from the two marimo notebooks (like tori_trades_trendline's ``topdown.py``). There
is no ``strategy.py``, so the strategy loader and ``tsim`` skip this folder. No figures here: the
plotly code lives in the notebook cells. Every function is pure or a cached loader.

**The night** of session *d* runs from the previous session's close (16:00 New York) to *d*'s
open (09:30). Its parts, by New York clock, come from the 1-minute bars of the ~530 names that
have extended-hours bars (2025-06-18 → 2026-06-26):

* after-hours ``post`` 16:01-19:59 of the previous session. The 16:00 minute holds the closing
  auction; the daily bar's close IS that official close, and it is the night's reference price;
* the dark hours 20:00-04:00: no prints for these names (only ~5 other names trade there);
* ``late`` 08:00-08:09: overnight trades reported late at stale prices, so volume only;
* pre-market ``early`` 04:00-07:59 and ``pre`` 08:10-09:29;
* ``first``: the first trading hour 09:30-10:29, the outcome.

Minutes with volume 0 are IBKR placeholders (a copy of the previous close) and are dropped.

**News.** SEC 8-K filings carry EDGAR's acceptance time to the second, so they can be placed in
the night (corrected filer by filer: for about half the filers EDGAR's JSON stamp is 4-5 hours
late, see :func:`unshift`). MarketData.app's Benzinga headlines carry only a calendar date (every
item is stamped 00:00), so a headline dated on session day *d* may have come out after the open.
It is counted only as ``news_same``, a HINDSIGHT column, never a signal.

* Loaders (cached under ``var/overnight_open/``): :func:`universe`, :func:`load_parts`,
  :func:`load_daily`, :func:`fetch_filings`, :func:`news_days`, :func:`read_headlines`,
  :func:`minute_bars`.
* The table: :func:`build` (all of it, cached), :func:`night_table` (one row per name and
  session), :func:`flags`.
* Statistics: :func:`day_t`, :func:`cluster_diff`, :func:`rank_ic`, :func:`auc`, :func:`deciles`,
  :func:`lift`, :func:`continuation`, :func:`ic_table`, :func:`flag_table`, :func:`walk_probs`,
  :func:`boot_auc`, :func:`ladder`, :func:`prereg`.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "var" / "overnight_open"
NEWS = ROOT / "var" / "marketdata" / "news"
NY = "America/New_York"

#: The ~530 names with extended-hours 1-minute bars; after 2026-06-26 only 7 names have them.
START, END = "2025-06-18", "2026-06-26"
#: Last discovery session. The validation months after it are read ONCE (``prereg``).
DISCOVERY_END = "2026-02-27"
#: First date of the Benzinga headlines (MarketData.app keeps one year).
NEWS_START = "2025-10-14"
#: First month the model ladder is scored in discovery (the news baselines are warm by then).
SCORE_FROM = "2025-11"
VALIDATION_FROM = "2026-03"
#: Half-day sessions: their after-hours start at 13:00, so they and the next session are left out.
HALF_DAYS = ("2025-07-03", "2025-11-28", "2025-12-24")
BASE_DAYS = 20          # sessions in every "normal" baseline (the day itself never included)
ATR_DAYS = 14           # Wilder ATR on daily bars
STRONG_X = 2.0          # pre-registered: a strong first hour has 2x the stock's usual range
MIN_MOVE_ATR = 0.5      # pre-registered: a night "moved" when the pre-market is 0.5 ATR away
#: Days on which the extended-hours names are listed (their union is the universe).
SAMPLE_DAYS = ("2025-07-17", "2025-11-17", "2026-03-17", "2026-06-17")

#: Night parts by New York clock, ``[start, end)``.
PARTS = {"post": ("16:01", "20:00"), "early": ("04:00", "08:00"), "late": ("08:00", "08:10"),
         "pre": ("08:10", "09:30"), "first": ("09:30", "10:30")}

#: Every signal known before 09:30: column -> (group, what it measures).
FEATURES = {
    "atr_pct": ("previous days", "daily ATR(14) as a share of the price"),
    "fh_norm": ("previous days", "the stock's usual first-hour range (median of 20 sessions)"),
    "prev_ret_atr": ("previous days", "yesterday's close-to-close move, in ATRs"),
    "prev_range_atr": ("previous days", "yesterday's high-low range, in ATRs"),
    "prev_vol_rel": ("previous days", "yesterday's volume / its 20-day mean"),
    "ret_5d_atr": ("previous days", "the last 5 sessions' move, in ATRs"),
    "ah_ret_atr": ("night move", "after-hours: last print by 20:00 vs the close, in ATRs"),
    "ah_range_atr": ("night move", "after-hours high-low range, in ATRs"),
    "on_ret_atr": ("night move", "dark hours: first pre-market print vs last after-hours print"),
    "pm_ret_atr": ("night move", "pre-market: last print by 09:29 vs the close, in ATRs"),
    "pm_range_atr": ("night move", "pre-market high-low range, in ATRs"),
    "ah_vol_rel": ("night activity", "after-hours volume / its own 20-day mean"),
    "on_vol_rel": ("night activity", "08:00-08:09 volume (overnight trades reported late) / mean"),
    "pm_vol_rel": ("night activity", "pre-market volume / its own 20-day mean"),
    "pm_vol_share": ("night activity", "pre-market volume / a normal day's volume"),
    "sec_prev": ("news", "SEC 8-Ks filed during the previous session (09:30-16:00)"),
    "sec_night": ("news", "SEC 8-Ks accepted during the night (16:00-09:30)"),
    "earnings_night": ("news", "8-Ks with Item 2.02 (results) accepted during the night"),
    "news_prev": ("news", "Benzinga headlines dated from the previous session to the day before"),
    "news_prev_rel": ("news", "news_prev / its 20-day mean"),
}
GROUPS = ("previous days", "night move", "night activity", "news")
#: The ladder: each step adds one group to everything before it.
STEPS = {("" if k == 0 else "+ ") + g: [c for c, (grp, _) in FEATURES.items()
                                       if GROUPS.index(grp) <= k]
         for k, g in enumerate(GROUPS)}

#: Signed moves: a size question reads them as |x| (a big move either way).
MOVES = ("prev_ret_atr", "ret_5d_atr", "ah_ret_atr", "on_ret_atr", "pm_ret_atr")
#: Plain names of the common 8-K items.
ITEMS = {"1.01": "material agreement", "1.02": "agreement ended", "1.03": "bankruptcy",
         "2.01": "deal completed", "2.02": "RESULTS", "2.03": "new debt",
         "2.05": "restructuring", "2.06": "impairment", "3.01": "listing notice",
         "3.02": "share sale", "3.03": "holder rights", "4.01": "auditor change",
         "5.01": "control change", "5.02": "officer / director change", "5.03": "bylaws",
         "5.07": "shareholder vote", "7.01": "Reg FD release", "8.01": "other event",
         "9.01": "exhibits"}


def _engine(engine=None):
    """The given engine, else the shared DB's."""
    if engine is not None:
        return engine
    from trading_simple.db import make_engine
    return make_engine()


def _key(symbols) -> str:
    """A short, stable fingerprint of a symbol list (cache file names)."""
    joined = ",".join(sorted(symbols)).encode()
    return f"{len(symbols)}-{hashlib.md5(joined).hexdigest()[:8]}"


def _read(sql: str, engine, **params) -> pd.DataFrame:
    """One SQL query as a frame."""
    from sqlalchemy import text
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# --- loaders -------------------------------------------------------------------------------

def universe(engine=None, path: Path = CACHE / "universe.pkl") -> list[str]:
    """The names with extended-hours 1-minute bars: those listed on any of :data:`SAMPLE_DAYS`.

    One after-hours hour (16:30-17:30 New York) per day is enough: every one of these names has a
    row in every after-hours minute (placeholders included), and a short window keeps it fast.

    Args:
        engine: SQLAlchemy engine (the shared DB when omitted).
        path: Cache file.

    Returns:
        Sorted tickers.
    """
    if path.exists():
        return pd.read_pickle(path)
    eng, names = _engine(engine), set()
    for day in SAMPLE_DAYS:
        a = (pd.Timestamp(day) + pd.Timedelta(hours=16, minutes=30)).tz_localize(NY)
        got = _read("SELECT DISTINCT symbol FROM prices_ohlcv WHERE bar = '1m' AND NOT rth "
                    "AND ts >= :a AND ts < :b", eng, a=a.tz_convert("UTC"),
                    b=(a + pd.Timedelta(hours=1)).tz_convert("UTC"))
        names |= set(got["symbol"])
    out = sorted(names)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


def _case() -> str:
    """The SQL CASE that names each minute's night part (NULL outside every part)."""
    return "CASE " + " ".join(f"WHEN t >= '{a}' AND t < '{b}' THEN '{k}'"
                              for k, (a, b) in PARTS.items()) + " END"


_PARTS_SQL = """
SELECT symbol, day, part, first(open, ts) AS open, last(close, ts) AS close,
       max(high) AS high, min(low) AS low, sum(volume) AS volume, count(*) AS prints
FROM (SELECT symbol, ts, open, high, low, close, volume,
             (ts AT TIME ZONE 'America/New_York')::date AS day, {case} AS part
      FROM (SELECT symbol, ts, open, high, low, close, volume,
                   (ts AT TIME ZONE 'America/New_York')::time AS t
            FROM prices_ohlcv
            WHERE bar = '1m' AND symbol = ANY(:s) AND ts >= :a AND ts < :b
              AND volume > 0) x) y
WHERE part IS NOT NULL
GROUP BY symbol, day, part
"""


def load_parts(symbols, start: str = START, end: str = END, engine=None,
               progress=None) -> pd.DataFrame:
    """Each name's night parts and first hour, aggregated in the database, cached per month.

    One row per name, New York date and part of :data:`PARTS`: the first open, last close, high,
    low, volume and number of printed minutes. Placeholder minutes (volume 0) are left out, so a
    part with no trade has no row. A stopped build resumes at the first month not cached.

    Args:
        symbols: Tickers.
        start: First date.
        end: Last date.
        engine: SQLAlchemy engine (the shared DB when omitted).
        progress: Optional wrapper around the month list (a progress bar).

    Returns:
        Long frame ``symbol, day, part, open, close, high, low, volume, prints``.
    """
    months = list(pd.period_range(start, end, freq="M"))
    eng, key, out = None, _key(symbols), []
    for m in (progress or iter)(months):
        path = CACHE / f"parts_{m}_{key}.pkl"
        if not path.exists():
            eng = eng or _engine(engine)
            a = pd.Timestamp(m.start_time).tz_localize(NY).tz_convert("UTC")
            b = pd.Timestamp((m + 1).start_time).tz_localize(NY).tz_convert("UTC")
            df = _read(_PARTS_SQL.format(case=_case()), eng, s=list(symbols), a=a, b=b)
            df["day"] = pd.to_datetime(df["day"])
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(path)
        out.append(pd.read_pickle(path))
    df = pd.concat(out, ignore_index=True)
    return df[(df["day"] >= start) & (df["day"] <= end)].reset_index(drop=True)


def load_daily(symbols, start: str = START, end: str = END, engine=None,
               warmup_days: int = 100) -> pd.DataFrame:
    """Daily bars from ``warmup_days`` calendar days before ``start``, cached.

    ``open`` is the official opening print and ``close`` the official close. Both are
    split-adjusted as stored and not dividend-adjusted, the same basis as the 1-minute bars.
    Daily bars are stored at 00:00 UTC of the trade date.

    Args:
        symbols: Tickers.
        start: First session that needs features.
        end: Last date.
        engine: SQLAlchemy engine (the shared DB when omitted).
        warmup_days: Calendar days loaded before ``start`` for the ATR and the baselines.

    Returns:
        Long frame ``symbol, day, open, high, low, close, volume``.
    """
    path = CACHE / f"daily_{start}_{end}_{_key(symbols)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    a = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=warmup_days)
    df = _read("SELECT symbol, ts, open, high, low, close, volume FROM prices_ohlcv "
               "WHERE bar = '1d' AND symbol = ANY(:s) AND ts >= :a AND ts <= :b",
               _engine(engine), s=list(symbols), a=a, b=pd.Timestamp(end, tz="UTC"))
    df["day"] = pd.to_datetime(df.pop("ts"), utc=True).dt.tz_localize(None).dt.normalize()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


_ACCEPTED = re.compile(r'Accepted</div>\s*<div class="info">([^<]+)<')


def unshift(stamps: pd.Series) -> pd.Series:
    """Undo EDGAR's doubled time-zone offset: the JSON stamp minus New York's UTC offset.

    For about half the filers, ``acceptanceDateTime`` in EDGAR's submissions JSON is late by New
    York's UTC offset (4 h in summer, 5 h in winter), as if the Eastern-to-UTC conversion ran
    twice: Micron's 8-K accepted at 16:02:22 New York reads ``00:02:22Z`` the next day, 20:02 in
    New York. The filing's index page shows the true time.

    Args:
        stamps: JSON stamps (UTC, tz-aware).

    Returns:
        The true acceptance times (UTC).
    """
    t = stamps - pd.Timedelta(hours=4)
    offset = t.dt.tz_localize(None) - t.dt.tz_convert(NY).dt.tz_localize(None)   # 4 h or 5 h
    return stamps - offset


def _accepted(sec, cik: int, accession: str) -> pd.Timestamp:
    """A filing's true acceptance time (UTC), read from its EDGAR index page."""
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
           f"{accession}-index.htm")
    html = sec._get(url, as_json=False).decode("utf-8", "replace")
    return pd.Timestamp(_ACCEPTED.search(html).group(1)).tz_localize(NY).tz_convert("UTC")


def _true_times(sec, cik: int, rows: pd.DataFrame) -> pd.DataFrame:
    """Set ``ts`` (true acceptance, UTC) and ``checked`` on one filer's 8-Ks.

    The JSON error is a property of the filer (checked on 132 filings of 6 names: every filing of
    a filer is either right or late by the doubled offset). So the oldest and newest filing are
    read from their index pages; when both show the same error, it is applied to all the filer's
    filings, otherwise every filing's index page is read.
    """
    js = pd.to_datetime(rows["json"], utc=True)
    probe = sorted({js.idxmin(), js.idxmax()})
    true = {i: _accepted(sec, cik, rows.at[i, "accession"]) for i in probe}

    def kind(i):
        if abs(js[i] - true[i]) < pd.Timedelta(minutes=1):
            return "right"
        return "doubled" if abs(unshift(js[[i]]).iloc[0] - true[i]) < pd.Timedelta(minutes=1) \
            else "other"

    kinds = {kind(i) for i in probe}
    if kinds in ({"right"}, {"doubled"}):
        rows["ts"] = js if kinds == {"right"} else unshift(js)
        rows["checked"] = "filer"
    else:
        rows["ts"] = [_accepted(sec, cik, a) for a in rows["accession"]]
        rows["checked"] = "index page"
    for i in probe:
        rows.at[i, "ts"], rows.at[i, "checked"] = true[i], "index page"
    return rows


def fetch_filings(symbols, since: str = "2025-05-01", until: str = "2026-07-31",
                  path: Path = CACHE / "sec_8k.pkl", progress=None) -> pd.DataFrame:
    """Every 8-K and 8-K/A filed in ``[since, until]``, with its TRUE acceptance time.

    EDGAR's submissions JSON lists the filings (1-2 requests per name); its acceptance stamp is
    corrected filer by filer from the index pages (:func:`unshift`, :func:`_true_times`, 2 more
    requests per name). The fetch is paced (``providers.sec``, ~8 requests a second, EDGAR's
    fair-access rule) and resumable: the cache is saved every 50 names. A name without a CIK or
    without any 8-K is stored with ``ts`` empty, so it is not asked again. Foreign private
    issuers file 6-Ks, not 8-Ks, so they have none (:func:`night_table` leaves them empty).

    Args:
        symbols: Tickers.
        since: Earliest filing date kept.
        until: Latest filing date kept.
        path: Cache file.
        progress: Optional wrapper around the list of names still to fetch.

    Returns:
        ``symbol, ts`` (New York, tz-aware), ``json`` (the raw stamp), ``checked``
        (``"index page"`` or ``"filer"``), ``filed`` (EDGAR's filing date), ``form, items,
        accession, url``.
    """
    import time

    from trading_simple.providers import sec

    cols = ["symbol", "ts", "json", "checked", "filed", "form", "items", "accession", "url"]
    have = pd.read_pickle(path) if path.exists() else pd.DataFrame(columns=cols)
    todo = sorted(set(symbols) - set(have["symbol"]))
    cik = sec._cik_map() if todo else {}
    got: list[pd.DataFrame] = []

    def save():
        nonlocal have, got
        if got:
            have = pd.concat([d for d in (have, *got) if len(d)], ignore_index=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            have.to_pickle(path)
            got = []

    def filings_of(s):
        c = cik[s]
        subs = sec._get(sec._SUBMISSIONS_URL.format(cik=c), as_json=True)
        blocks = [subs["filings"]["recent"]]
        if min(blocks[0].get("filingDate") or ["0"]) > since:
            blocks += [sec._get("https://data.sec.gov/submissions/" + f["name"], as_json=True)
                       for f in subs["filings"].get("files", []) if f.get("filingTo", "0") >= since]
        rows = [(s, None, b["acceptanceDateTime"][i], None, b["filingDate"][i], form,
                 b["items"][i], b["accessionNumber"][i],
                 f"https://www.sec.gov/Archives/edgar/data/{c}/"
                 f"{b['accessionNumber'][i].replace('-', '')}/{b['primaryDocument'][i]}")
                for b in blocks for i, form in enumerate(b.get("form", []))
                if form in ("8-K", "8-K/A") and since <= b["filingDate"][i] <= until]
        rows = pd.DataFrame(rows, columns=cols).drop_duplicates("accession", ignore_index=True)
        return _true_times(sec, c, rows) if len(rows) else rows

    for k, s in enumerate((progress or iter)(todo), 1):
        rows = pd.DataFrame(columns=cols)
        if s in cik:
            for attempt in range(4):
                try:
                    rows = filings_of(s)
                    break
                except Exception:  # noqa: BLE001 - timeouts, resets: wait and ask again
                    time.sleep(10 * (attempt + 1))
            else:
                continue                                  # not cached: asked again next run
        got.append(rows if len(rows) else pd.DataFrame([[s] + [None] * 8], columns=cols))
        if k % 50 == 0:
            save()
    save()
    out = have[have["symbol"].isin(symbols)].reset_index(drop=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert(NY)
    return out


def _news_files(symbol: str, first: str, last: str, root: Path) -> list[Path]:
    """A name's monthly news files that overlap ``[first, last]``."""
    return [f for f in sorted((root / symbol).glob("*.csv.gz"))
            if f.name[11:21] >= first and f.name[:10] <= last]


def read_headlines(symbol: str, first: str = NEWS_START, last: str = END,
                   root: Path = NEWS) -> pd.DataFrame:
    """One name's Benzinga headlines (MarketData.app), one row per distinct headline and date.

    Args:
        symbol: Ticker.
        first: First date (``YYYY-MM-DD``).
        last: Last date.
        root: The news folder.

    Returns:
        ``date`` (New York calendar date, no time of day: the feed stamps 00:00),
        ``headline, source``, oldest first.
    """
    parts = []
    for f in _news_files(symbol, first, last, root):
        try:
            parts.append(pd.read_csv(f, usecols=["headline", "source", "publicationDate"]))
        except (pd.errors.EmptyDataError, ValueError):
            continue
    if not parts:
        return pd.DataFrame(columns=["date", "headline", "source"])
    n = pd.concat(parts, ignore_index=True)
    stamp = pd.to_datetime(n.pop("publicationDate"), unit="s", utc=True).dt.tz_convert(NY)
    n.insert(0, "date", stamp.dt.tz_localize(None).dt.normalize())
    n = n.drop_duplicates(["headline", "date"])
    n = n[(n["date"] >= first) & (n["date"] <= last)]
    return n.sort_values("date").reset_index(drop=True)


def news_days(symbols, first: str = NEWS_START, last: str = END, root: Path = NEWS,
              progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benzinga headline counts per name and calendar date, and each name's covered span.

    A name's span runs from its first news file's first date to its last file's last date (an
    empty file means "asked, nothing there", so it still counts as covered).

    Args:
        symbols: Tickers.
        first: First date.
        last: Last date.
        root: The news folder.
        progress: Optional wrapper around the symbol list.

    Returns:
        ``(counts, spans)``: ``symbol, date, n`` and ``symbol, a, b``.
    """
    path = CACHE / f"news_{first}_{last}_{_key(symbols)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    counts, spans = [], []
    for s in (progress or iter)(sorted(symbols)):
        files = _news_files(s, first, last, root)
        if not files:
            continue
        spans.append((s, max(pd.Timestamp(files[0].name[:10]), pd.Timestamp(first)),
                      min(pd.Timestamp(files[-1].name[11:21]), pd.Timestamp(last))))
        h = read_headlines(s, first, last, root)
        if len(h):
            c = h.groupby("date").size().rename("n").reset_index()
            counts.append(c.assign(symbol=s))
    out = (pd.concat(counts, ignore_index=True)[["symbol", "date", "n"]] if counts
           else pd.DataFrame(columns=["symbol", "date", "n"]),
           pd.DataFrame(spans, columns=["symbol", "a", "b"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


def minute_bars(symbol: str, first_day, last_day, engine=None) -> pd.DataFrame:
    """One name's printed 1-minute bars from 04:00 of ``first_day`` to 20:00 of ``last_day``.

    Args:
        symbol: Ticker.
        first_day: First New York date.
        last_day: Last New York date.
        engine: SQLAlchemy engine (the shared DB when omitted).

    Returns:
        ``time`` (New York wall clock, naive), ``open, high, low, close, volume`` and ``part``:
        ``post`` / ``dark`` / ``early`` / ``late`` / ``pre`` / ``rth`` / ``close`` (the 16:00
        closing-auction minute). Placeholder minutes (volume 0) are left out.
    """
    a = pd.Timestamp(first_day).tz_localize(NY) + pd.Timedelta(hours=4)
    b = pd.Timestamp(last_day).tz_localize(NY) + pd.Timedelta(hours=20)
    df = _read("SELECT ts, open, high, low, close, volume FROM prices_ohlcv WHERE bar = '1m' "
               "AND symbol = :s AND ts >= :a AND ts < :b AND volume > 0 ORDER BY ts",
               _engine(engine), s=symbol, a=a.tz_convert("UTC"), b=b.tz_convert("UTC"))
    t = pd.to_datetime(df.pop("ts"), utc=True).dt.tz_convert(NY).dt.tz_localize(None)
    clock = t.dt.hour * 60 + t.dt.minute
    df.insert(0, "time", t)
    df["part"] = np.select(
        [clock < 4 * 60, clock < 8 * 60, clock < 8 * 60 + 10, clock < 9 * 60 + 30,
         clock < 16 * 60, clock == 16 * 60, clock < 20 * 60],
        ["dark", "early", "late", "pre", "rth", "close", "post"], "dark")
    return df


def build(engine=None, progress=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Everything both notebooks read: the night table, the filings and the daily bars.

    Every step is cached, so only the first call is slow: about 25 minutes of 1-minute bars
    (month by month) and about 4 minutes of SEC requests (name by name). :func:`night_table`
    itself is recomputed every time (seconds), so a change to it needs no rebuild.

    Args:
        engine: SQLAlchemy engine (the shared DB when omitted).
        progress: Optional ``progress(iterable, title)`` wrapper (a progress bar).

    Returns:
        ``(table, filings, daily)``.
    """
    def wrap(title):
        return (lambda xs: progress(xs, title)) if progress else None

    names = universe(engine)
    parts = load_parts(names, engine=engine, progress=wrap("1-minute bars, month by month"))
    daily = load_daily(names, engine=engine)
    filings = fetch_filings(names, progress=wrap("SEC 8-K filings, name by name"))
    news = news_days(names, progress=wrap("Benzinga headlines, name by name"))
    return night_table(parts, daily, filings, news), filings, daily


# --- the table -----------------------------------------------------------------------------

def _calendar(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading sessions: the dates on which at least half the names have a daily bar."""
    n = daily.groupby("day")["symbol"].nunique()
    return pd.DatetimeIndex(n.index[n >= 0.5 * n.max()]).sort_values()


def _ratio(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """``a / b`` where ``b > 0``, else NaN."""
    return (a / b).where(b > 0)


def _base(x: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """The mean (or median) of the :data:`BASE_DAYS` sessions BEFORE each row."""
    return getattr(x.shift(1).rolling(BASE_DAYS, min_periods=BASE_DAYS // 2), how)()


def _filing_frames(filings: pd.DataFrame, days: pd.DatetimeIndex, syms: list[str]) -> dict:
    """Wide ``sec_night``, ``earnings_night`` and ``sec_prev`` counts.

    A filing accepted during a session's regular hours (09:30-16:00) counts as ``sec_prev`` of
    the NEXT session; any other filing counts in the night of the first session that opens after
    it (weekends and holidays roll forward). A name that filed no 8-K at all is not an 8-K filer
    (a foreign issuer, a fund): its columns stay NaN rather than a misleading 0.
    """
    zero = pd.DataFrame(0.0, index=days, columns=syms)
    filers = set(filings.dropna(subset=["ts"])["symbol"]) & set(syms)
    out = {k: zero.copy() for k in ("sec_night", "earnings_night", "sec_prev")}
    e = filings[filings["symbol"].isin(filers)].dropna(subset=["ts"])
    if len(e):
        ts = pd.DatetimeIndex(e["ts"]).tz_convert(NY)
        opens = (days + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY)
        closes = (days + pd.Timedelta(hours=16)).tz_localize(NY)
        k = opens.searchsorted(ts, side="left")
        prev_close = closes[np.clip(k - 1, 0, len(days) - 1)]
        during = (k >= 1) & (ts < prev_close)
        ok = k < len(days)
        rows = pd.DataFrame({"day": days[np.clip(k, 0, len(days) - 1)],
                             "symbol": e["symbol"].to_numpy(), "during": during,
                             "earn": e["items"].fillna("").str.contains("2.02", regex=False)
                             .to_numpy()})[ok]
        for name, sel in (("sec_night", ~rows["during"]),
                          ("earnings_night", ~rows["during"] & rows["earn"]),
                          ("sec_prev", rows["during"])):
            c = rows[sel].groupby(["day", "symbol"]).size().unstack()
            out[name] = out[name].add(c.reindex(index=days, columns=syms).fillna(0))
    keep = [s in filers for s in syms]
    return {k: v.where(np.broadcast_to(keep, v.shape)) for k, v in out.items()}


def _news_frames(news: tuple[pd.DataFrame, pd.DataFrame], days: pd.DatetimeIndex,
                 syms: list[str]) -> dict:
    """Wide ``news_prev``, ``news_prev_rel`` and ``news_same`` (hindsight) headline counts.

    Session *d* with previous session *p*: ``news_prev`` counts the headlines dated *p* up to the
    day before *d* (all published before *d*'s midnight, so known before its open);
    ``news_same`` counts those dated *d* itself, which may have come out after the open.
    NaN outside a name's covered span.
    """
    counts, spans = news
    prev = pd.DataFrame(0.0, index=days, columns=syms)
    same = prev.copy()
    c = counts[counts["symbol"].isin(syms)]
    if len(c):
        date = pd.DatetimeIndex(c["date"])
        k = days.searchsorted(date, side="right")      # first session after the date
        ok = k < len(days)
        p = c[ok].assign(day=days[k[ok]]).groupby(["day", "symbol"])["n"].sum().unstack()
        prev = prev.add(p.reindex(index=days, columns=syms).fillna(0))
        on = date.isin(days)
        s = c[on].assign(day=date[on]).groupby(["day", "symbol"])["n"].sum().unstack()
        same = same.add(s.reindex(index=days, columns=syms).fillna(0))
    span = spans.set_index("symbol").reindex(syms)
    first_prev = pd.Series(days, index=days).shift(1)
    a = span["a"].to_numpy(dtype="datetime64[ns]")[None, :]
    b = span["b"].to_numpy(dtype="datetime64[ns]")[None, :]
    d = days.to_numpy()[:, None]
    covered_prev = (first_prev.to_numpy()[:, None] >= a) & (d - np.timedelta64(1, "D") <= b)
    covered_same = (d >= a) & (d <= b)
    prev = prev.where(covered_prev)
    return {"news_prev": prev, "news_prev_rel": _ratio(prev, _base(prev)),
            "news_same": same.where(covered_same)}


def night_table(parts: pd.DataFrame, daily: pd.DataFrame, filings: pd.DataFrame | None = None,
                news: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per name and session: every signal known before 09:30, and the first hour.

    Signals (:data:`FEATURES`) come from the daily bars up to the previous session, the night's
    parts and the news; moves are in units of the daily ATR (Wilder, :data:`ATR_DAYS`, known at
    the previous close), volumes against the name's own mean of the :data:`BASE_DAYS` sessions
    before. Outcomes start at the official open ``O`` (what a market-on-open order gets):

    * ``fh_ret`` = last print of 09:30-10:29 / ``O`` - 1, and ``fh_ret_x`` = the same minus the
      day's mean over all names (the market taken out);
    * ``fh_range`` = the first hour's high-low range (the open included) / ``O``, and
      ``fh_range_rel`` = that / ``fh_norm``, the name's median first-hour range over the
      sessions before (2 = twice its usual first hour);
    * ``rest_ret`` = the close / the first hour's last print - 1, what came after;
    * ``gap`` = ``O`` / previous close - 1, known AT the open (the pre-market move is the
      version known before it).

    Rows left out: before a name's baselines are warm, half-days and the sessions after them,
    and the rare day whose 1-minute open is more than 10% from the official open (a scale
    mismatch, e.g. an unadjusted split).

    Args:
        parts: :func:`load_parts`.
        daily: :func:`load_daily`.
        filings: :func:`fetch_filings` (news columns NaN when omitted).
        news: :func:`news_days` (Benzinga columns NaN when omitted).

    Returns:
        Long frame: ``day, month, symbol, split``, the features, ``news_same`` and the outcomes.
    """
    days = _calendar(daily)
    syms = sorted(set(parts["symbol"]) & set(daily["symbol"]))

    def wide(frame, col):
        return frame.pivot(index="day", columns="symbol", values=col).reindex(
            index=days, columns=syms)

    d = {c: wide(daily, c) for c in ("open", "high", "low", "close", "volume")}
    p = {k: {c: wide(parts[parts["part"] == k], c)
             for c in ("open", "high", "low", "close", "volume")} for k in PARTS}
    seen = parts.groupby("symbol")["day"].agg(["min", "max"]).reindex(syms)
    when = days.to_numpy()[:, None]
    in_span = pd.DataFrame((when >= seen["min"].to_numpy()[None, :])
                           & (when <= seen["max"].to_numpy()[None, :]),
                           index=days, columns=syms) & d["volume"].notna()
    vol = {k: p[k]["volume"].fillna(0).where(in_span) for k in PARTS}   # no print = no volume

    opn, C = d["open"], d["close"].shift(1)
    tr = np.fmax(d["high"] - d["low"], np.fmax((d["high"] - C).abs(), (d["low"] - C).abs()))
    atr_pct = tr.ewm(alpha=1 / ATR_DAYS, min_periods=ATR_DAYS).mean().shift(1) / C
    post_close = p["post"]["close"].shift(1)
    pm_close = p["pre"]["close"].fillna(p["early"]["close"])        # last print before 09:30
    pm_first = p["early"]["open"].fillna(p["pre"]["open"])          # first print after 04:00
    pm_hi = np.fmax(p["early"]["high"], p["pre"]["high"])
    pm_lo = np.fmin(p["early"]["low"], p["pre"]["low"])
    pm_vol = vol["early"] + vol["pre"]
    first = p["first"]
    fh_range = ((np.fmax(first["high"], opn) - np.fmin(first["low"], opn)) / opn).where(
        first["close"].notna())
    fh_norm = _base(fh_range, "median")
    frames = {
        "atr_pct": atr_pct,
        "fh_norm": fh_norm,
        "prev_ret_atr": (C / d["close"].shift(2) - 1) / atr_pct,
        "prev_range_atr": (d["high"] - d["low"]).shift(1) / C / atr_pct,
        "prev_vol_rel": _ratio(d["volume"].shift(1), _base(d["volume"].shift(1))),
        "ret_5d_atr": (C / d["close"].shift(6) - 1) / atr_pct,
        "ah_ret_atr": (post_close / C - 1) / atr_pct,
        "ah_range_atr": (p["post"]["high"] - p["post"]["low"]).shift(1) / C / atr_pct,
        "on_ret_atr": (pm_first / post_close - 1) / atr_pct,
        "pm_ret_atr": (pm_close / C - 1) / atr_pct,
        "pm_range_atr": (pm_hi - pm_lo) / C / atr_pct,
        "ah_vol_rel": _ratio(vol["post"].shift(1), _base(vol["post"].shift(1))),
        "on_vol_rel": _ratio(vol["late"], _base(vol["late"])),
        "pm_vol_rel": _ratio(pm_vol, _base(pm_vol)),
        "pm_vol_share": _ratio(pm_vol, _base(d["volume"])),
    }
    nan = pd.DataFrame(np.nan, index=days, columns=syms)
    frames |= (_filing_frames(filings, days, syms) if filings is not None else
               dict.fromkeys(("sec_night", "earnings_night", "sec_prev"), nan))
    frames |= (_news_frames(news, days, syms) if news is not None else
               dict.fromkeys(("news_prev", "news_prev_rel", "news_same"), nan))
    frames |= {"gap": opn / C - 1, "fh_ret": first["close"] / opn - 1, "fh_range": fh_range,
               "fh_range_rel": fh_range / fh_norm, "rest_ret": d["close"] / first["close"] - 1}

    half = days.isin(pd.to_datetime(list(HALF_DAYS)))
    skip = half | np.r_[False, half[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_ok = ~(np.log(first["open"] / opn).abs() > 0.1)
    keep = (in_span & first["close"].notna() & opn.notna() & atr_pct.notna() & fh_norm.notna()
            & scale_ok & np.broadcast_to(~skip[:, None], in_span.shape))
    out = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    out = out[keep.stack(future_stack=True)]
    out = out.rename_axis(["day", "symbol"]).reset_index()
    out.insert(1, "month", out["day"].dt.to_period("M").astype(str))
    out.insert(3, "split", np.where(out["day"] <= DISCOVERY_END, "discovery", "validation"))
    by_day = out.groupby("day")["fh_ret"]
    out["fh_ret_x"] = (out["fh_ret"] - by_day.transform("mean")).where(
        by_day.transform("count") >= 20)
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def flags(t: pd.DataFrame, *, move_atr: float = 1.0, vol_x: float = 3.0,
          news_x: float = 3.0, min_news: int = 3) -> pd.DataFrame:
    """The unusual-night flags, one boolean column each.

    Args:
        t: :func:`night_table`.
        move_atr: A move flag fires at this many daily ATRs (either direction).
        vol_x: A volume flag fires at this multiple of the name's own normal.
        news_x: The headline flag fires at this multiple of the name's normal count ...
        min_news: ... and at least this many headlines.

    Returns:
        Frame aligned with ``t``.
    """
    return pd.DataFrame({
        "after-hours move": t["ah_ret_atr"].abs() >= move_atr,
        "pre-market move": t["pm_ret_atr"].abs() >= move_atr,
        "after-hours volume": t["ah_vol_rel"] >= vol_x,
        "overnight volume (08:00)": t["on_vol_rel"] >= vol_x,
        "pre-market volume": t["pm_vol_rel"] >= vol_x,
        "SEC filing at night": t["sec_night"] > 0,
        "earnings at night": t["earnings_night"] > 0,
        "headlines": (t["news_prev_rel"] >= news_x) & (t["news_prev"] >= min_news),
    }, index=t.index)


# --- statistics ----------------------------------------------------------------------------

def day_t(values, days) -> dict:
    """Mean of ``values`` with a standard error clustered by day.

    Cluster-robust (CR1): names on one day share that day's market and news, so treating them as
    independent overstates t by roughly the square root of the names per day.

    Args:
        values: One value per row (NaN dropped).
        days: Each row's session.

    Returns:
        ``{"mean", "se", "t", "n", "days"}``.
    """
    v = pd.Series(np.asarray(values, dtype=float))
    d = pd.Series(np.asarray(days))
    ok = v.notna().to_numpy()
    v, d = v[ok], d[ok]
    n, g = len(v), d.nunique()
    if n < 3 or g < 3:
        return {"mean": float(v.mean()) if n else np.nan, "se": np.nan, "t": np.nan, "n": n,
                "days": g}
    m = float(v.mean())
    sums = (v - m).groupby(d.to_numpy()).sum().to_numpy()
    se = math.sqrt((sums ** 2).sum() * g / (g - 1)) / n
    return {"mean": m, "se": se, "t": m / se if se > 0 else np.nan, "n": n, "days": g}


def cluster_diff(values, group, days) -> dict:
    """Mean of ``values`` where ``group`` minus where not, with a t clustered by day.

    An OLS of the values on a constant and the 0/1 group, with a CR1 covariance by day.

    Returns:
        ``{"diff", "se", "t", "n"}``.
    """
    v, g = np.asarray(values, float), np.asarray(group, float)
    ok = np.isfinite(v) & np.isfinite(g)
    v, g, dd = v[ok], g[ok], np.asarray(days)[ok]
    if ok.sum() < 3 or g.min() == g.max():
        return {"diff": np.nan, "se": np.nan, "t": np.nan, "n": int(ok.sum())}
    X = np.column_stack([np.ones_like(g), g])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ v
    e = v - X @ beta
    score = pd.DataFrame(X * e[:, None]).groupby(dd).sum().to_numpy()
    G = len(score)
    cov = bread @ (score.T @ score) @ bread * G / (G - 1)
    se = math.sqrt(cov[1, 1])
    return {"diff": float(beta[1]), "se": se, "t": beta[1] / se if se > 0 else np.nan,
            "n": int(ok.sum())}


def rank_ic(x, y, days) -> dict:
    """Spearman rank correlation of ``x`` with ``y``, with a t clustered by day.

    The t is :func:`day_t` of the slope of rank(y) on rank(x), both centred: the rank
    correlation up to scale.

    Returns:
        ``{"ic", "t", "n"}``.
    """
    x, y, days = np.asarray(x, float), np.asarray(y, float), np.asarray(days)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"ic": np.nan, "t": np.nan, "n": int(ok.sum())}
    rx = pd.Series(x[ok]).rank(pct=True).to_numpy() - 0.5
    ry = pd.Series(y[ok]).rank(pct=True).to_numpy() - 0.5
    return {"ic": float(np.corrcoef(rx, ry)[0, 1]),
            "t": day_t(rx * ry / np.mean(rx * rx), days[ok])["t"], "n": int(ok.sum())}


def auc(p, y) -> float:
    """Area under the ROC curve: the chance a random positive scores above a random negative.

    Returns:
        The AUC, NaN when either class is missing.
    """
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, pos = p[ok], y[ok] > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(p)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def deciles(t: pd.DataFrame, x: str, y: str, bins: int = 10) -> pd.DataFrame:
    """Dose-response: the mean of ``y`` in each tenth of ``x``, with a day-clustered error.

    Args:
        t: Rows with ``day``.
        x: The signal.
        y: The outcome (a 0/1 column gives a rate).
        bins: Number of equal-count bins.

    Returns:
        One row per bin: ``bin, x`` (median), ``y, se, n``.
    """
    s = t[[x, y, "day"]].dropna()
    if len(s) < bins * 20:
        return pd.DataFrame(columns=["bin", "x", "y", "se", "n"])
    b = pd.qcut(s[x].rank(method="first"), bins, labels=False)
    rows = []
    for k, g in s.groupby(b):
        r = day_t(g[y].astype(float), g["day"])
        rows.append({"bin": int(k) + 1, "x": float(g[x].median()), "y": r["mean"],
                     "se": r["se"], "n": r["n"]})
    return pd.DataFrame(rows)


def lift(t: pd.DataFrame, flag: pd.Series, y: str, *, by: str = "pm_ret_atr",
         q: int = 5) -> dict:
    """How much higher ``y`` is on flagged nights than on unflagged nights that moved as much.

    Each night falls in a fifth of ``|by|`` (no pre-market print = its own group). Its expected
    ``y`` is the mean of the UNFLAGGED nights in the same group; the lift is the flagged nights'
    mean excess over that expectation, with a t clustered by day. It separates "this flag
    matters" from "flagged nights simply moved more before the open".

    Returns:
        ``{"n", "rate", "matched", "lift", "t"}``.
    """
    size = t[by].abs()
    group = pd.qcut(size.rank(method="first"), q, labels=False).fillna(-1)
    yy = t[y].astype(float)
    expected = group.map(yy[~flag].groupby(group[~flag]).mean())
    r = day_t((yy - expected)[flag], t.loc[flag, "day"])
    return {"n": int(flag.sum()), "rate": float(yy[flag].mean()),
            "matched": float(expected[flag].mean()), "lift": r["mean"], "t": r["t"]}


def continuation(t: pd.DataFrame, mask=None, *, min_move: float = MIN_MOVE_ATR) -> dict:
    """Does the first hour continue the night's move, beyond the market?

    On nights whose pre-market moved at least ``min_move`` ATRs, the market-demeaned first-hour
    return signed by the night's direction: positive = the first hour went on in the night's
    direction, negative = it faded the night. In basis points, t clustered by day.

    Args:
        t: :func:`night_table` rows.
        mask: Optional boolean subset (e.g. a flag).
        min_move: The night's minimum move in ATRs.

    Returns:
        ``{"n", "days", "bp", "se", "t", "share"}`` (share = fraction that continued).
    """
    m = t["pm_ret_atr"].abs().ge(min_move) & t["fh_ret_x"].notna()
    if mask is not None:
        m &= mask
    v = np.sign(t.loc[m, "pm_ret_atr"]) * t.loc[m, "fh_ret_x"] * 1e4
    r = day_t(v, t.loc[m, "day"])
    return {"n": r["n"], "days": r["days"], "bp": r["mean"], "se": r["se"],
            "t": r["t"], "share": float((v > 0).mean()) if len(v) else np.nan}


def ic_table(t: pd.DataFrame, y: str, *, size: bool) -> pd.DataFrame:
    """Every signal's rank correlation with ``y``, t clustered by day.

    Args:
        t: :func:`night_table` rows.
        y: The outcome column.
        size: True for a size outcome: the signed moves (:data:`MOVES`) enter as |x|.

    Returns:
        ``group, signal, what, ic, t, n``.
    """
    rows = []
    for c, (grp, words) in FEATURES.items():
        absolute = size and c in MOVES
        r = rank_ic(t[c].abs() if absolute else t[c], t[y], t["day"])
        rows.append({"group": grp, "signal": ("|" + c + "|") if absolute else c, "what": words,
                     "ic": r["ic"], "t": r["t"], "n": r["n"]})
    return pd.DataFrame(rows)


def flag_table(t: pd.DataFrame, fl: pd.DataFrame, y: str = "strong") -> pd.DataFrame:
    """Per flag: how often it fires, the size outcome against nights that moved as much
    (:func:`lift`), and the direction (:func:`continuation`).

    Args:
        t: :func:`night_table` rows with the 0/1 column ``y``.
        fl: :func:`flags` on the same rows.
        y: The size outcome.

    Returns:
        One row per flag, plus "any flag" and "no flag".
    """
    anyf = fl.any(axis=1)
    rows = []
    for name in [*fl.columns, "any flag", "no flag"]:
        m = anyf if name == "any flag" else ~anyf if name == "no flag" else fl[name]
        lf = lift(t, m, y) if name != "no flag" else {"lift": np.nan, "t": np.nan,
                                                       "matched": np.nan}
        c = continuation(t, m)
        rows.append({"flag": name, "nights": int(m.sum()), "share of nights": m.mean(),
                     "strong": t.loc[m, y].mean(), "strong if moved as much": lf["matched"],
                     "lift": lf["lift"], "lift t": lf["t"], "moved nights": c["n"],
                     "continuation bp": c["bp"], "continuation t": c["t"]})
    return pd.DataFrame(rows)


def walk_probs(t: pd.DataFrame, cols, y: str, *, first_month: str,
               last_month: str | None = None) -> pd.Series:
    """Out-of-sample P(y) per row: each month scored by a model fit on all EARLIER months.

    A small gradient-boosted tree (scikit-learn ``HistGradientBoostingClassifier``, fixed
    settings, missing values allowed). No month ever sees its own or a later outcome; the
    outcome is same-day, so earlier months are complete when a month starts.

    Args:
        t: Rows with ``month``.
        cols: The signals.
        y: The 0/1 outcome column (NaN rows are not fit).
        first_month: First month scored (``YYYY-MM``).
        last_month: Last month scored (all when omitted).

    Returns:
        P(y = 1), NaN outside the scored months.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = pd.Series(np.nan, index=t.index)
    months = sorted(m for m in t["month"].unique()
                    if m >= first_month and (last_month is None or m <= last_month))
    for m in months:
        fit = t[(t["month"] < m) & t[y].notna()]
        if fit[y].nunique() < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        model.fit(fit[list(cols)].to_numpy(float), fit[y].astype(int).to_numpy())
        test = (t["month"] == m).to_numpy()
        p[test] = model.predict_proba(t.loc[test, list(cols)].to_numpy(float))[:, 1]
    return p


def boot_auc(probs: dict, y, days, *, draws: int = 400, seed: int = 0) -> pd.DataFrame:
    """AUC of each model on day-bootstrap resamples (whole sessions drawn with replacement).

    Args:
        probs: Model name -> scores, all on the same rows.
        y: The 0/1 outcome.
        days: Each row's session.
        draws: Resamples.
        seed: Random seed.

    Returns:
        ``draws`` x models frame of AUCs; the first row is the full sample.
    """
    y = np.asarray(y, float)
    P = {k: np.asarray(v, float) for k, v in probs.items()}
    _, inv = np.unique(np.asarray(days), return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    rows = [{k: auc(v, y) for k, v in P.items()}]
    for _ in range(draws):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        rows.append({k: auc(v[idx], y[idx]) for k, v in P.items()})
    return pd.DataFrame(rows)


def ladder(t: pd.DataFrame, y: str, *, first_month: str, last_month: str,
           steps: dict = STEPS, draws: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample AUC as the night's information is added group by group.

    Every step's model is walked forward (:func:`walk_probs`) and scored on the SAME rows.
    The gain of each step over the one before comes with a day-bootstrap 95% interval.

    Args:
        t: :func:`night_table` rows with the 0/1 column ``y``.
        y: The outcome.
        first_month: First scored month.
        last_month: Last scored month.
        steps: Step name -> its (cumulative) signals.
        draws: Bootstrap resamples.

    Returns:
        ``(table, boot)``: per step ``step, signals, auc, lo, hi, gain, gain_lo, gain_hi``; and the
        :func:`boot_auc` draws (for any other comparison).
    """
    probs = {k: walk_probs(t, cols, y, first_month=first_month, last_month=last_month)
             for k, cols in steps.items()}
    ok = t[y].notna() & pd.concat(probs, axis=1).notna().all(axis=1)
    boot = boot_auc({k: v[ok] for k, v in probs.items()}, t.loc[ok, y], t.loc[ok, "day"],
                    draws=draws)
    names, rows = list(steps), []
    for i, k in enumerate(names):
        gain = boot[k] - (boot[names[i - 1]] if i else 0.5)
        rows.append({"step": k, "signals": len(steps[k]), "auc": boot[k].iloc[0],
                     "lo": boot[k].iloc[1:].quantile(0.025),
                     "hi": boot[k].iloc[1:].quantile(0.975),
                     "gain": gain.iloc[0], "gain_lo": gain.iloc[1:].quantile(0.025),
                     "gain_hi": gain.iloc[1:].quantile(0.975)})
    return pd.DataFrame(rows), boot


def news_night(t: pd.DataFrame) -> pd.Series:
    """Nights with news known before the open: an 8-K at night or a headline flag."""
    f = flags(t)
    return f["SEC filing at night"] | f["headlines"]


def prereg(t: pd.DataFrame, *, first_month: str, last_month: str, draws: int = 400) -> dict:
    """The four pre-registered numbers (README) on one window.

    * V1: AUC gain for a strong first hour (``fh_range_rel`` >= :data:`STRONG_X`) of "previous
      days + the night's price move" over "previous days";
    * V2: AUC gain of all signals (activity and news too) over "previous days + night move";
    * V3: continuation of the night's move in the first hour, all moved nights;
    * V4: continuation on news nights minus quiet nights.

    Args:
        t: :func:`night_table` (all months: the models train on everything before the window).
        first_month: First month of the window.
        last_month: Last month.
        draws: Bootstrap resamples.

    Returns:
        ``{"V1": (gain, lo, hi), "V2": (gain, lo, hi), "V3": continuation dict,
        "V4": cluster_diff dict}``.
    """
    t = t.assign(strong=(t["fh_range_rel"] >= STRONG_X).astype(float))
    names = list(STEPS)
    _, boot = ladder(t, "strong", first_month=first_month, last_month=last_month, draws=draws)
    out = {}
    for key, (a, b) in {"V1": (names[1], names[0]), "V2": (names[3], names[1])}.items():
        g = boot[a] - boot[b]
        out[key] = (g.iloc[0], g.iloc[1:].quantile(0.025), g.iloc[1:].quantile(0.975))
    w = t[(t["month"] >= first_month) & (t["month"] <= last_month)]
    out["V3"] = continuation(w)
    m = w["pm_ret_atr"].abs().ge(MIN_MOVE_ATR) & w["fh_ret_x"].notna()
    v = np.sign(w.loc[m, "pm_ret_atr"]) * w.loc[m, "fh_ret_x"] * 1e4
    out["V4"] = cluster_diff(v, news_night(w)[m], w.loc[m, "day"])
    return out
