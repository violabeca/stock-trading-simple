"""options_hedge research helpers: do stocks with "outlier" activity then move more than their
options priced in?

Loaded by path from ``options_hedge.ipynb`` (as ``rs``) and from ``tests/test_options_hedge.py``;
the framework never imports it. The test it serves is pre-registered in this folder's README:
four screens fire on day t, an at-the-money straddle is bought at day t+1's close (at the ask)
and held to expiry, and the screened days are compared with random name-days matched on the
entry date and on how expensive the options were.

Data: MarketData.app end-of-day chains and news in ``var/marketdata/`` (quotes stamped 16:00
New York, no implied volatility), daily bars from the shared DB, SEC 8-K Item 2.02 earnings
dates. Every
derived table is cached in ``var/options_hedge/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "var" / "marketdata" / "chains"
NEWS = ROOT / "var" / "marketdata" / "news"
CACHE = ROOT / "var" / "options_hedge"
NY = "America/New_York"

#: Horizon -> (min DTE, max DTE, target DTE): the expiry closest to the target inside the band,
#: the shorter one on a tie (so "7d" is the shortest expiry 4-14 calendar days out).
HORIZONS = {"30d": (21, 45, 30), "7d": (4, 14, 4)}
#: Exploratory variants (not pre-registered): the same expiry rule, but the strike nearest the price
#: among straddles that already pass the spread filter. The pre-registered rule takes the nearest
#: two-sided strike, often a fresh $1 strike with no open interest whose wide quote then fails the
#: filter while the standard strike beside it is liquid (found by the independent re-scoring).
LIQUID = {"30d_liq": "30d", "7d_liq": "7d"}
MAX_STRIKE_GAP = 0.05   # the ATM strike is at most this far from the price
MAX_SPREAD = 0.10       # straddle bid-ask spread / mid at entry
FEE = 0.65              # $ per contract (entry: both legs; exit: the leg in the money)
LOOKBACK = 20           # sessions of trailing history behind every screen (the day excluded)
RATIO = 3.0             # "unusual" = at least this multiple of the trailing norm
FRESH_DAYS = 5          # a screen counts only on the first day of an episode
MIN_OPT_BASE = 50       # options_volume: the trailing median volume at least this (contracts)
MIN_NEWS = 3            # news: at least this many headlines on the day
BREAK_JUMP = 0.15       # |change of log(chain price / DB adjusted close)| above this = split/break
EARN_PAD = (2, 1)       # earnings reaction within [t - 2, expiry + 1] sessions excludes a straddle
ALPHA = 0.003           # one-sided discovery bar per cell (~0.025 / 8 cells)
SCREENS = ("options_volume", "stock_volume", "news", "outlier_move")
CHAIN_COLS = ["date", "optionSymbol", "expiration", "side", "strike", "bid", "ask",
              "underlyingPrice", "volume", "openInterest"]
LEGS = ("cb", "ca", "pb", "pa")


# --- option chains --------------------------------------------------------------------------

def option_root(symbol: str) -> str:
    """The OCC root a standard contract on ``symbol`` carries (punctuation dropped).

    Args:
        symbol: Stock ticker, e.g. ``"MU"``.

    Returns:
        The expected root; an adjusted contract after a split or merger carries another one
        (``"XYZ1"``) and is left out.
    """
    return symbol.replace(".", "").replace("-", "")


def chain_day_summary(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reduce one name's end-of-day chain rows to one row per date.

    Every date keeps the stock price, the day's total option volume and open interest, and for
    each horizon in :data:`HORIZONS` the at-the-money straddle: the expiry closest to the
    horizon's target inside its band among expiries with a qualifying strike, then the strike
    nearest the price (the lower one on a tie) among strikes at most :data:`MAX_STRIKE_GAP` away
    whose call AND put both have a two-sided quote (bid > 0, ask >= bid). Each :data:`LIQUID`
    variant does the same among straddles whose spread is already within :data:`MAX_SPREAD`.

    Args:
        raw: MarketData chain rows (:data:`CHAIN_COLS`), any number of dates.
        symbol: The ticker; rows of non-standard contracts (another root) are dropped.

    Returns:
        One row per date: ``symbol, date, S, opt_vol, opt_oi`` and per horizon ``h`` (and per
        :data:`LIQUID` variant): ``h_exp, h_K, h_cb, h_ca, h_pb, h_pa`` (NaN where no straddle
        qualifies). Empty when no standard contract is in ``raw``.
    """
    d = raw[raw["optionSymbol"].str[:-15] == option_root(symbol)]
    if d.empty:
        return pd.DataFrame()
    exp = pd.to_datetime(d["expiration"], unit="s", utc=True).dt.tz_convert(NY)
    d = d.assign(date=pd.to_datetime(d["date"]), exp=exp.dt.tz_localize(None).dt.normalize())
    day = d.groupby("date").agg(S=("underlyingPrice", "median"), opt_vol=("volume", "sum"),
                                opt_oi=("openInterest", "sum"))
    q = d[(d["bid"] > 0) & (d["ask"] >= d["bid"])]
    key = ["date", "exp", "strike"]
    side = {s: q[q["side"] == s].drop_duplicates(key)[key + ["bid", "ask"]]
            for s in ("call", "put")}
    legs = side["call"].merge(side["put"], on=key, suffixes=("_c", "_p"))
    legs = legs.rename(columns={"bid_c": "cb", "ask_c": "ca", "bid_p": "pb", "ask_p": "pa"})
    legs = legs.join(day["S"], on="date")
    legs = legs[(legs["strike"] / legs["S"] - 1).abs() <= MAX_STRIKE_GAP]
    legs["dte"] = (legs["exp"] - legs["date"]).dt.days
    mid = (legs["cb"] + legs["ca"] + legs["pb"] + legs["pa"]) / 2
    liquid = (legs["ca"] + legs["pa"] - legs["cb"] - legs["pb"]) / mid <= MAX_SPREAD
    out = day.copy()
    variants = {**{h: (HORIZONS[h], legs) for h in HORIZONS},
                **{v: (HORIZONS[h], legs[liquid]) for v, h in LIQUID.items()}}
    for h, ((lo, hi, target), pick) in variants.items():
        x = pick[pick["dte"].between(lo, hi)].copy()
        x["ekey"] = (x["dte"] - target).abs() * 1000 + x["dte"]
        best = x.loc[x.groupby("date")["ekey"].idxmin(), ["date", "exp"]]
        x = x.merge(best, on=["date", "exp"])
        x["kkey"] = (x["strike"] - x["S"]).abs()
        x = x.sort_values(["date", "kkey", "strike"]).drop_duplicates("date").set_index("date")
        out[f"{h}_exp"] = x["exp"]
        out[f"{h}_K"] = x["strike"]
        for leg in LEGS:
            out[f"{h}_{leg}"] = x[leg]
    return out.reset_index().assign(symbol=symbol)


def read_chains(symbol: str, first: str, last: str, root: Path = CHAINS) -> pd.DataFrame:
    """Summarize one name's monthly chain files from month ``first`` to ``last`` (``YYYY-MM``).

    Args:
        symbol: Ticker (a folder under ``root``).
        first: First month, inclusive.
        last: Last month, inclusive.
        root: The chains folder.

    Returns:
        :func:`chain_day_summary` rows of every readable file in range (empty files skipped).
    """
    parts = []
    for f in sorted((root / symbol).glob("*.csv.gz")):
        if not first <= f.name[:7] <= last:
            continue
        try:
            raw = pd.read_csv(f, usecols=CHAIN_COLS)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        parts.append(chain_day_summary(raw, symbol))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def chain_summaries(first: str, last: str, symbols: list[str] | None = None,
                    root: Path = CHAINS, cache: Path = CACHE) -> pd.DataFrame:
    """Every name's daily chain summary for months ``first``..``last``, cached (pickle).

    Args:
        first: First month (``YYYY-MM``).
        last: Last month (``YYYY-MM``).
        symbols: Names to read; every folder under ``root`` when omitted.
        root: The chains folder.
        cache: Cache folder (the file name carries the month range and the name count).

    Returns:
        The concatenated :func:`chain_day_summary` rows.
    """
    symbols = sorted(symbols or [p.name for p in root.iterdir() if p.is_dir()])
    path = cache / f"chains_{first}_{last}_{len(symbols)}_v2.pkl"
    if path.exists():
        return pd.read_pickle(path)
    parts = []
    for k, s in enumerate(symbols, 1):
        parts.append(read_chains(s, first, last, root))
        if k % 100 == 0:
            print(f"\r{k}/{len(symbols)} chains", end="")
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    cache.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


# --- stock bars, news, earnings ------------------------------------------------------------

def load_daily(symbols: list[str], start: str, end: str, engine=None,
               cache: Path = CACHE) -> pd.DataFrame:
    """Daily bars from the shared DB (long: ``symbol, date, close, adj_close, volume``), cached.

    Daily bars are stored at 00:00 UTC of the trade date, so ``date`` is that calendar day.

    Args:
        symbols: Tickers.
        start: First date (inclusive).
        end: Last date (inclusive).
        engine: SQLAlchemy engine; ``trading_simple.db.make_engine()`` when omitted.
        cache: Cache folder.

    Returns:
        Long frame of the stored bars.
    """
    path = cache / f"daily_{start}_{end}_{len(symbols)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from sqlalchemy import text

    from trading_simple.db import make_engine
    sql = text("SELECT symbol, ts, close, adj_close, volume FROM prices_ohlcv WHERE bar = '1d' "
               "AND symbol = ANY(:s) AND ts >= :a AND ts <= :b")
    with (engine or make_engine()).connect() as conn:
        long = pd.read_sql(sql, conn, params={"s": list(symbols),
                                              "a": pd.Timestamp(start, tz="UTC"),
                                              "b": pd.Timestamp(end, tz="UTC")})
    long["date"] = pd.to_datetime(long.pop("ts"), utc=True).dt.tz_localize(None).dt.normalize()
    cache.mkdir(parents=True, exist_ok=True)
    long.to_pickle(path)
    return long


def news_counts(symbols: list[str], sessions: pd.DatetimeIndex, first: str, last: str,
                root: Path = NEWS) -> pd.DataFrame:
    """Headlines per name and session (sessions x symbols); NaN where no news file covers it.

    A headline's ``publicationDate`` is a New York calendar date (the feed stamps midnight); it
    counts on the first session on or after that date, so weekend items roll to Monday. Duplicate
    headlines on one date count once.

    Args:
        symbols: Tickers.
        sessions: The trading calendar.
        first: First month (``YYYY-MM``) of news files to read.
        last: Last month.
        root: The news folder.

    Returns:
        Wide count frame; a name's sessions before its first covered day or after its last are NaN.
    """
    out = pd.DataFrame(np.nan, index=sessions, columns=sorted(symbols))
    for s in out.columns:
        files = [f for f in sorted((root / s).glob("*.csv.gz")) if first <= f.name[:7] <= last]
        if not files:
            continue
        a, b = pd.Timestamp(files[0].name[:10]), pd.Timestamp(files[-1].name[11:21])
        parts = []
        for f in files:
            try:
                parts.append(pd.read_csv(f, usecols=["headline", "publicationDate"]))
            except (pd.errors.EmptyDataError, ValueError):
                continue
        span = (sessions >= a) & (sessions <= b)
        out.loc[span, s] = 0.0
        n = pd.concat(parts) if parts else pd.DataFrame(columns=["headline", "publicationDate"])
        if n.empty:
            continue
        day = pd.to_datetime(n["publicationDate"], unit="s", utc=True).dt.tz_convert(NY)
        n = n.assign(day=day.dt.tz_localize(None).dt.normalize())
        n = n.drop_duplicates(["headline", "day"])
        pos = sessions.searchsorted(n["day"].to_numpy(), side="left")
        pos = pos[pos < len(sessions)]
        cnt = pd.Series(1.0, index=sessions[pos]).groupby(level=0).sum()
        cnt = cnt[(cnt.index >= a) & (cnt.index <= b)]
        out.loc[cnt.index, s] += cnt
    return out


def fetch_earnings(symbols: list[str], since: str = "2025-06-01",
                   path: Path = CACHE / "earnings_sec.pkl") -> pd.DataFrame:
    """Earnings releases from SEC EDGAR, fetched once per name and cached (resumable).

    A US filer furnishes its earnings release on an 8-K carrying Item 2.02 ("Results of
    Operations"); EDGAR stamps the acceptance to the second, in UTC. Foreign private issuers file
    6-Ks instead and get no dates (so they drop out of the test: their calendar is unknown). A name
    without a CIK or without Item 2.02 filings is stored with ``ts`` NaT so it is not asked again.

    Args:
        symbols: Tickers.
        since: Earliest filing date kept.
        path: Cache file (pickle), saved every 200 names.

    Returns:
        Every cached row for ``symbols``: ``symbol, ts`` (New York time, tz-aware).
    """
    from trading_simple.providers import sec

    have = pd.read_pickle(path) if path.exists() else pd.DataFrame({"symbol": [], "ts": []})
    todo = sorted(set(symbols) - set(have["symbol"]))
    cik = sec._cik_map() if todo else {}
    got = []

    def save():
        nonlocal have, got
        if got:
            new = pd.concat(got, ignore_index=True)
            new["ts"] = pd.to_datetime(new["ts"], utc=True)
            have = pd.concat([d for d in (have, new) if len(d)], ignore_index=True)
            have["ts"] = pd.to_datetime(have["ts"], utc=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            have.to_pickle(path)
            got = []

    def blocks_of(c):
        subs = sec._get(sec._SUBMISSIONS_URL.format(cik=c), as_json=True)
        blocks = [subs["filings"]["recent"]]
        if min(blocks[0].get("filingDate") or ["0"]) > since:
            shards = [f["name"] for f in subs["filings"].get("files", [])
                      if f.get("filingTo", "0") >= since]
            blocks += [sec._get("https://data.sec.gov/submissions/" + name, as_json=True)
                       for name in shards]
        return blocks

    import time
    for k, s in enumerate(todo, 1):
        ts, blocks = [], []
        if s in cik:
            for attempt in range(4):
                try:
                    blocks = blocks_of(cik[s])
                    break
                except Exception:  # noqa: BLE001 — timeouts, resets: wait and ask again
                    time.sleep(10 * (attempt + 1))
            else:
                continue                                   # not cached: asked again next run
            for b in blocks:
                for i, form in enumerate(b.get("form", [])):
                    if (form in ("8-K", "8-K/A") and "2.02" in b["items"][i].split(",")
                            and b["filingDate"][i] >= since):
                        ts.append(b["acceptanceDateTime"][i])
        got.append(pd.DataFrame({"symbol": s, "ts": ts or [None]}))
        if k % 200 == 0:
            save()
            print(f"\r{k}/{len(todo)} names", end="")
    save()
    out = have[have["symbol"].isin(symbols)].reset_index(drop=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert(NY)
    return out


def reaction_positions(earnings: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The session whose close first reflects each earnings release, as a calendar position.

    A release at or after 16:00 New York reacts on the next session, any earlier one on its own
    day (the first session on or after a weekend date).

    Args:
        earnings: :func:`fetch_earnings` rows (NaT rows are dropped).
        sessions: The trading calendar.

    Returns:
        ``symbol, pos`` (index into ``sessions``; ``len(sessions)`` when past its end).
    """
    e = earnings.dropna(subset=["ts"])
    local = e["ts"].dt.tz_convert(NY)
    day = local.dt.tz_localize(None).dt.normalize().to_numpy()
    pos = sessions.searchsorted(day, side="left")
    same = (pos < len(sessions)) & (sessions[np.minimum(pos, len(sessions) - 1)] == day)
    pos = np.where(same & (local.dt.hour >= 16).to_numpy(), pos + 1, pos)
    return pd.DataFrame({"symbol": e["symbol"].to_numpy(), "pos": pos})


# --- screens -------------------------------------------------------------------------------

def _fresh(fire: pd.DataFrame) -> pd.DataFrame:
    """Keep a firing only when the same screen did not fire in the previous FRESH_DAYS sessions."""
    recent = fire.astype(float).shift(1).rolling(FRESH_DAYS, min_periods=1).max().fillna(0) > 0
    return fire & ~recent


def screens(opt_vol: pd.DataFrame, vol_adj: pd.DataFrame, adj: pd.DataFrame,
            news: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """The four pre-registered screens on day t, each from data through t's close only.

    Every trailing norm is built from the LOOKBACK sessions before t (``shift(1)``) and needs all
    of them present. ``pool`` marks the name-days on which a screen could fire (its inputs exist);
    ``fire`` the first day of each episode (:func:`_fresh`).

    Args:
        opt_vol: Total option volume (sessions x symbols).
        vol_adj: Split-adjusted share volume.
        adj: Adjusted close.
        news: Headline counts (:func:`news_counts`).

    Returns:
        ``{screen: (pool, fire)}`` boolean frames, keyed by :data:`SCREENS`.
    """
    def med(x):
        return x.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).median()

    out = {}
    b = med(opt_vol)
    pool = b.notna() & opt_vol.notna() & (b >= MIN_OPT_BASE)
    out["options_volume"] = (pool, pool & (opt_vol >= RATIO * b))
    b = med(vol_adj)
    pool = b.notna() & vol_adj.notna() & (b > 0)
    out["stock_volume"] = (pool, pool & (vol_adj >= RATIO * b))
    m = news.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    pool = m.notna() & news.notna()
    out["news"] = (pool, pool & (news >= MIN_NEWS) & (news >= RATIO * m))
    r = np.log(adj).diff()
    sd = r.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).std()
    pool = sd.notna() & r.notna() & (sd > 0)
    out["outlier_move"] = (pool, pool & (r.abs() >= RATIO * sd))
    return {k: (p, _fresh(f)) for k, (p, f) in out.items()}


# --- straddle outcomes ---------------------------------------------------------------------

def _at(wide: pd.DataFrame, pos: np.ndarray, symbols: np.ndarray, fill=np.nan) -> np.ndarray:
    """``wide.iloc[pos, column of symbol]`` for many rows at once (``fill`` where out of range)."""
    a = wide.to_numpy()
    ci = wide.columns.get_indexer(symbols)
    ok = (ci >= 0) & (pos >= 0) & (pos < len(wide))
    out = np.full(len(pos), fill, dtype=a.dtype if a.dtype == bool else float)
    out[ok] = a[pos[ok], ci[ok]]
    return out


def straddles(summ: pd.DataFrame, sessions: pd.DatetimeIndex, h: str) -> pd.DataFrame:
    """One row per name and entry session: the horizon's ATM straddle bought at that close.

    The screen day ``t`` is the session before the entry. The payoff is the straddle's intrinsic
    value at the chain's stock price on the expiry date (NaN when that date has no chain row).

    Args:
        summ: :func:`chain_summaries` rows.
        sessions: The trading calendar.
        h: Horizon key of :data:`HORIZONS`.

    Returns:
        ``symbol, t, entry, exp, t_pos, exp_pos, K, S0, ST, payoff, cost_ask, cost_mid, spread,
        implied, outlay, r_ask, r_mid`` plus the four legs.
    """
    s = summ.dropna(subset=[f"{h}_K"])
    pos = sessions.get_indexer(s["date"])
    s = s[pos > 0]
    pos = pos[pos > 0]
    legs = {leg: s[f"{h}_{leg}"].to_numpy() for leg in LEGS}
    df = pd.DataFrame({"symbol": s["symbol"].to_numpy(), "t": sessions[pos - 1],
                       "entry": s["date"].to_numpy(), "exp": s[f"{h}_exp"].to_numpy(),
                       "t_pos": pos - 1, "K": s[f"{h}_K"].to_numpy(), "S0": s["S"].to_numpy(),
                       **legs})
    df["exp_pos"] = sessions.searchsorted(df["exp"].to_numpy(), side="left")
    price = summ.set_index(["symbol", "date"])["S"]
    df["ST"] = price.reindex(pd.MultiIndex.from_arrays([df["symbol"], df["exp"]])).to_numpy()
    df["payoff"] = (df["ST"] - df["K"]).abs()
    df["cost_ask"] = df["ca"] + df["pa"]
    df["cost_mid"] = (df["cb"] + df["ca"] + df["pb"] + df["pa"]) / 2
    df["spread"] = (df["ca"] + df["pa"] - df["cb"] - df["pb"]) / df["cost_mid"]
    df["implied"] = df["cost_mid"] / df["S0"]
    df["outlay"] = 100 * df["cost_ask"] + 2 * FEE
    df["r_ask"] = r_ask(df["payoff"], df["outlay"])
    df["r_mid"] = df["payoff"] / df["cost_mid"] - 1
    return df


def r_ask(payoff, outlay, edge: float = 1.0):
    """Straddle return bought at the ask: payoff (× ``edge``, a planted effect) less the exit fee.

    Args:
        payoff: Intrinsic value at expiry per share.
        outlay: Dollars paid for one straddle (100 shares per contract, fees included).
        edge: Multiplier on the payoff (1.0 = the real one).

    Returns:
        ``(100 × payoff × edge − FEE·[payoff > 0]) / outlay − 1``.
    """
    pay = 100 * payoff * edge
    return (pay - FEE * (pay > 0)) / outlay - 1


def breaks(chain_price: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Sessions where the chain's raw price and the DB's adjusted close disagree by a jump.

    A split moves the raw price and not the adjusted one; so does a wrong bar. Where either price
    is missing the ratio is carried forward, so a jump across a gap still shows.

    Args:
        chain_price: The chain's stock price (sessions x symbols).
        adj: The DB's adjusted close, same shape.

    Returns:
        Boolean frame: True on a session whose log ratio moved more than :data:`BREAK_JUMP`.
    """
    q = np.log(chain_price / adj.reindex_like(chain_price)).ffill()
    return q.diff().abs() > BREAK_JUMP


def flag_rows(df: pd.DataFrame, brk: pd.DataFrame, defined: pd.DataFrame,
              reactions: pd.DataFrame) -> pd.DataFrame:
    """Add ``broken`` and ``earn`` to :func:`straddles` rows.

    ``broken``: a break (:func:`breaks`) in [t − LOOKBACK, expiry], or the chain/DB price ratio
    undefined at t or at expiry. ``earn``: an earnings reaction in [t − EARN_PAD[0],
    expiry + EARN_PAD[1]] sessions.

    Args:
        df: :func:`straddles` rows.
        brk: :func:`breaks` frame.
        defined: Where the chain/DB price ratio exists (sessions x symbols).
        reactions: :func:`reaction_positions` rows.

    Returns:
        ``df`` with the two flags.
    """
    sym = df["symbol"].to_numpy()
    t, e = df["t_pos"].to_numpy(), df["exp_pos"].to_numpy()
    c = brk.astype(int).cumsum()
    c_e = _at(c, np.minimum(e, len(c) - 1), sym, 0)
    lo = t - LOOKBACK - 1
    c_lo = np.where(lo >= 0, _at(c, np.maximum(lo, 0), sym, 0), 0)
    ok = _at(defined, t, sym, False).astype(bool) & _at(defined, e, sym, False).astype(bool)
    df = df.assign(broken=(c_e - c_lo > 0) | ~ok)
    earn = np.zeros(len(df), bool)
    by = reactions.groupby("symbol")["pos"].apply(lambda p: np.sort(p.to_numpy()))
    for s, idx in df.groupby("symbol").indices.items():
        if s not in by.index:
            continue
        r = by[s]
        a = np.searchsorted(r, t[idx] - EARN_PAD[0], side="left")
        b = np.searchsorted(r, e[idx] + EARN_PAD[1], side="right")
        earn[idx] = b > a
    return df.assign(earn=earn)


def terciles(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Implied-move tercile (1 = cheapest third, 3 = dearest) within each name over ``mask`` rows.

    Args:
        df: Rows with ``symbol`` and ``implied``.
        mask: The rows to rank among (others get 0).

    Returns:
        Integer series aligned to ``df``.
    """
    pct = df.loc[mask].groupby("symbol")["implied"].rank(pct=True, method="first")
    out = pd.Series(0, index=df.index)
    out.loc[mask] = np.ceil(pct * 3).clip(1, 3).astype(int)
    return out


# --- the test ------------------------------------------------------------------------------

def _cells(pool: pd.DataFrame, events: pd.DataFrame):
    """(pool rows of a cell, event count) for every (entry, tercile) cell holding an event."""
    groups = pool.groupby(["entry", "tercile"]).indices
    for cell, k in events.groupby(["entry", "tercile"]).size().items():
        yield groups[cell], k


def draw_means(pool: pd.DataFrame, events: pd.DataFrame, arrays: dict[str, np.ndarray],
               n: int, seed: int) -> dict[str, np.ndarray]:
    """Means of ``n`` random event sets matched to ``events`` cell by cell (with replacement).

    Every random set draws, in each (entry date, tercile) cell, as many pool rows as the events
    have there. The same draws are scored on every array in ``arrays`` (pool-aligned values).

    Args:
        pool: Eligible rows (``entry``, ``tercile``); ``events`` must be a subset.
        events: The screen's rows.
        arrays: Name -> values aligned to ``pool`` rows.
        n: Number of random sets.
        seed: RNG seed.

    Returns:
        Name -> ``n`` means.
    """
    rng = np.random.default_rng(seed)
    tot = {k: np.zeros(n) for k in arrays}
    for rows, k in _cells(pool, events):
        idx = rows[rng.integers(0, len(rows), size=(n, k))]
        for name, a in arrays.items():
            tot[name] += a[idx].sum(axis=1)
    return {k: v / len(events) for k, v in tot.items()}


def p_value(null: np.ndarray, obs: float) -> float:
    """One-sided p: the share of random sets scoring at least ``obs`` (with the +1 correction)."""
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def without_best(events: pd.DataFrame, value: str = "r_ask") -> tuple[float, float]:
    """Mean ``value`` without the best entry week, and without the 5 top-contributing names.

    Args:
        events: The screen's rows (``entry``, ``symbol``, ``value``).
        value: Column to average.

    Returns:
        ``(mean without the best week, mean without the top-5 names)``.
    """
    week = events["entry"].dt.to_period("W-FRI")
    best = events.groupby(week)[value].sum().idxmax()
    top = events.groupby("symbol")[value].sum().nlargest(5).index
    return (float(events.loc[week != best, value].mean()),
            float(events.loc[~events["symbol"].isin(top), value].mean()))


def week_ci(events: pd.DataFrame, value: str = "r_ask", n: int = 5000, seed: int = 0):
    """95% interval of the mean from resampling whole entry weeks (overlapping holds share weeks).

    Args:
        events: Rows with ``entry`` and ``value``.
        value: Column to average.
        n: Bootstrap draws.
        seed: RNG seed.

    Returns:
        ``(low, high)``.
    """
    g = events.groupby(events["entry"].dt.to_period("W-FRI"))[value].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), size=(n, len(g)))
    means = g["sum"].to_numpy()[pick].sum(1) / g["count"].to_numpy()[pick].sum(1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def name_cluster_p(pool: pd.DataFrame, events: pd.DataFrame, n: int = 5000,
                   seed: int = 0, value: str = "r_ask") -> float:
    """One-sided p of the events' excess over their cells, resampling whole NAMES.

    The matched control treats events as independent picks, but one name can fire again a few
    sessions later and its straddles then overlap (one outcome counted twice). Here each event's
    excess is its ``r_ask`` minus the mean of its (entry, tercile) cell in the pool, and the names
    are bootstrapped with all their events, so a name's repeated events move together. Not part of
    the pre-registered bar: a warning when it disagrees with the control.

    Args:
        pool: Eligible rows (``entry``, ``tercile``, ``r_ask``).
        events: The screen's rows (a subset of ``pool``).
        n: Bootstrap draws.
        seed: RNG seed.
        value: The return column.

    Returns:
        Share of name-resampled means of the excess at or below zero; NaN under 10 names.
    """
    if events["symbol"].nunique() < 10:
        return float("nan")
    cell = pool.groupby(["entry", "tercile"])[value].mean()
    ex = events[value].to_numpy() - cell.reindex(
        pd.MultiIndex.from_arrays([events["entry"], events["tercile"]])).to_numpy()
    g = pd.DataFrame({"s": events["symbol"].to_numpy(), "x": ex})
    g = g.groupby("s")["x"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), size=(n, len(g)))
    means = g["sum"].to_numpy()[pick].sum(1) / g["count"].to_numpy()[pick].sum(1)
    return float((1 + np.sum(means <= 0)) / (1 + n))


def score_cell(pool: pd.DataFrame, events: pd.DataFrame, n_draws: int = 20000,
               edges=(1.15, 1.30), seed: int = 0, value: str = "r_ask") -> dict:
    """Every pre-registered number of one screen × horizon cell.

    Args:
        pool: Eligible rows on which the screen could fire (with ``tercile``).
        events: The rows on which it fired.
        n_draws: Random sets for the control (and for each planted edge).
        edges: Planted payoff multipliers for the power check (a straddle BUYER's payoff; with
            another ``value`` the power is NaN).
        seed: RNG seed.
        value: The return column scored (``r_ask``: bought at the ask; e.g. a seller's return).

    Returns:
        Dict: counts, observed means, control mean / quantile, p, robustness, week CI, the excess
        over the control with a ±1.96 sd band of the random sets, power per edge, the four pass
        flags and ``null`` (the random-set means, for plots).
    """
    pool = pool.reset_index(drop=True)
    pay, out = pool["payoff"].to_numpy(), pool["outlay"].to_numpy()
    arrays = {"r_ask": pool[value].to_numpy(), "r_mid": pool["r_mid"].to_numpy()}
    null = draw_means(pool, events, arrays, n_draws, seed)
    planted = draw_means(pool, events, {e: r_ask(pay, out, e) for e in edges}, n_draws, seed + 1)
    obs = float(events[value].mean())
    no_week, no_names = without_best(events, value)
    lo, hi = week_ci(events, value)
    p = p_value(null["r_ask"], obs)
    cut = np.quantile(null["r_ask"], 1 - ALPHA)
    res = dict(n=len(events), names=events["symbol"].nunique(),
               weeks=events["entry"].dt.to_period("W-FRI").nunique(),
               r_ask=obs, r_mid=float(events["r_mid"].mean()),
               ratio=float((events["payoff"] / events["cost_mid"]).mean()),
               ctrl_r_ask=float(null["r_ask"].mean()), ctrl_r_mid=float(null["r_mid"].mean()),
               ctrl_q=float(cut), p=p, no_best_week=no_week, no_top5=no_names, ci_lo=lo, ci_hi=hi,
               excess=obs - float(null["r_ask"].mean()),
               excess_lo=obs - float(null["r_ask"].mean()) - 1.96 * float(null["r_ask"].std()),
               excess_hi=obs - float(null["r_ask"].mean()) + 1.96 * float(null["r_ask"].std()))
    for e in edges:
        res[f"power_{e:.2f}"] = float(np.mean(planted[e] >= cut)) if value == "r_ask" else np.nan
    res["p_names"] = name_cluster_p(pool, events, value=value)
    res["tradable"] = obs > 0
    res["skill"] = p <= ALPHA
    res["robust"] = no_week > 0 and no_names > 0
    res["passed"] = res["tradable"] and res["skill"] and res["robust"]
    res["null"] = null["r_ask"]
    return res


# --- assembly ------------------------------------------------------------------------------

def calendar(summ: pd.DataFrame, daily: pd.DataFrame, share: float = 0.3) -> pd.DatetimeIndex:
    """Trading sessions: dates on which at least ``share`` of the names have a chain row or a bar.

    Args:
        summ: :func:`chain_summaries` rows.
        daily: :func:`load_daily` rows.
        share: Minimum share of a source's names present on the date.

    Returns:
        Sorted session dates.
    """
    days = []
    for d in (summ, daily):
        n = d.groupby("date")["symbol"].nunique()
        days.append(n.index[n >= share * d["symbol"].nunique()])
    return pd.DatetimeIndex(sorted(set(days[0]) | set(days[1])))


def assemble(summ: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
             earnings: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    """Wide frames, the four screens and every horizon's straddle rows with their flags.

    Each straddle row carries, per screen ``k``, ``k_pool`` and ``k_fire`` read on its screen day
    ``t`` (the session before the entry).

    Args:
        summ: :func:`chain_summaries` rows.
        daily: :func:`load_daily` rows.
        news: :func:`news_counts` frame on ``sessions``.
        earnings: :func:`fetch_earnings` rows.
        sessions: The trading calendar.

    Returns:
        ``{"frames": {...}, "screens": {...}, "react": reactions, "rows": {horizon: frame}}``,
        with a rows frame for every horizon and :data:`LIQUID` variant present in ``summ``.
    """
    symbols = sorted(summ["symbol"].unique())

    def wide(d, col):
        return d.pivot(index="date", columns="symbol", values=col).reindex(
            index=sessions, columns=symbols)

    f = {"S": wide(summ, "S"), "opt_vol": wide(summ, "opt_vol"),
         "adj": wide(daily, "adj_close"), "close": wide(daily, "close"),
         "volume": wide(daily, "volume")}
    f["vol_adj"] = f["volume"] * f["close"] / f["adj"]
    f["news"] = news.reindex(index=sessions, columns=symbols)
    scr = screens(f["opt_vol"], f["vol_adj"], f["adj"], f["news"])
    brk = breaks(f["S"], f["adj"])
    defined = (f["S"] / f["adj"]).notna()
    react = reaction_positions(earnings, sessions)
    rows = {}
    for h in [h for h in (*HORIZONS, *LIQUID) if f"{h}_K" in summ]:
        df = flag_rows(straddles(summ, sessions, h), brk, defined, react)
        t, sym = df["t_pos"].to_numpy(), df["symbol"].to_numpy()
        for k, (pool, fire) in scr.items():
            df[f"{k}_pool"] = _at(pool, t, sym, False).astype(bool)
            df[f"{k}_fire"] = _at(fire, t, sym, False).astype(bool)
        rows[h] = df
    return {"frames": f, "screens": scr, "breaks": brk, "react": react, "rows": rows}


def window_rows(df: pd.DataFrame, react: pd.DataFrame, sessions: pd.DatetimeIndex,
                first_entry: str, last_exp: str, max_spread: float = MAX_SPREAD) -> pd.DataFrame:
    """The rows of one test window, with ``elig`` (scored), ``earn_set`` (information) and terciles.

    A row is in the window when bought on or after ``first_entry`` and expiring on or before
    ``last_exp``. ``elig``: spread within ``max_spread``, an outcome price, no break, the name
    has an earnings reaction inside the window (so its calendar is known), and no earnings in the
    hold. ``earn_set``: the same, except the hold DOES touch an earnings release.

    Args:
        df: One horizon's :func:`assemble` rows.
        react: :func:`reaction_positions` rows.
        sessions: The trading calendar.
        first_entry: First entry date.
        last_exp: Last expiry date.
        max_spread: Liquidity filter (the pre-registered :data:`MAX_SPREAD` by default).

    Returns:
        The window's rows with ``elig``, ``earn_set``, ``tercile`` (within ``elig``) and
        ``tercile_e`` (within ``earn_set``).
    """
    a, b = pd.Timestamp(first_entry), pd.Timestamp(last_exp)
    w = df[(df["entry"] >= a) & (df["exp"] <= b)].copy()
    lo, hi = sessions.searchsorted(a), sessions.searchsorted(b, side="right")
    known = set(react.loc[(react["pos"] >= lo) & (react["pos"] < hi), "symbol"])
    base = (w["spread"] <= max_spread) & w["ST"].notna() & ~w["broken"] & w["symbol"].isin(known)
    w["elig"] = base & ~w["earn"]
    w["earn_set"] = base & w["earn"]
    w["tercile"] = terciles(w, w["elig"])
    w["tercile_e"] = terciles(w, w["earn_set"])
    return w


def score_window(w: pd.DataFrame, n_draws: int = 20000, earn: bool = False,
                 cells: list[str] | None = None) -> dict:
    """:func:`score_cell` for every screen of one horizon's window.

    Args:
        w: :func:`window_rows` output.
        n_draws: Random sets per control.
        earn: Score the earnings-window set instead of the eligible one (information only).
        cells: Screens to score (all of :data:`SCREENS` when omitted); any name with
            ``<name>_pool`` / ``<name>_fire`` columns in ``w`` works.

    Returns:
        ``{screen: score_cell result}`` (screens without events are left out).
    """
    out = {}
    for k in cells or SCREENS:
        if earn:
            pool = w[w["earn_set"] & w[f"{k}_pool"]].assign(tercile=lambda x: x["tercile_e"])
        else:
            pool = w[w["elig"] & w[f"{k}_pool"]]
        ev = pool[pool[f"{k}_fire"]]
        if len(ev):
            seed = SCREENS.index(k) if k in SCREENS else len(SCREENS)
            out[k] = score_cell(pool, ev, n_draws=n_draws, seed=seed)
    return out


# --- second study: option premia ----------------------------------------------------------

def bootstrap_p(x: pd.Series, groups: pd.Series, n: int = 10000, seed: int = 0) -> float:
    """One-sided p that the mean of ``x`` is ≤ 0, resampling whole ``groups`` (names or weeks).

    Args:
        x: Returns.
        groups: The cluster of each return, same index.
        n: Bootstrap draws.
        seed: RNG seed.

    Returns:
        Share of resampled means at or below zero (with the +1 correction); NaN under 10 groups.
    """
    g = pd.DataFrame({"g": groups.to_numpy(), "x": x.to_numpy()}).groupby("g")["x"].agg(
        ["sum", "count"])
    if len(g) < 10:
        return float("nan")
    pick = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    means = g["sum"].to_numpy()[pick].sum(1) / g["count"].to_numpy()[pick].sum(1)
    return float((1 + np.sum(means <= 0)) / (1 + n))


def event_straddles(symbol: str, plans: pd.DataFrame, root: Path = CHAINS) -> pd.DataFrame:
    """Open a liquid ATM straddle at one close and mark the same contracts at a later close.

    For each plan row, at the ``entry`` close: among expiries on or after ``min_exp``, the first
    one with a straddle within :data:`MAX_STRIKE_GAP` of the price whose bid-ask spread is within
    :data:`MAX_SPREAD` of its mid (both legs bid > 0), at the strike nearest the price; then the
    same call and put at the ``exit`` close.

    Args:
        symbol: Ticker.
        plans: ``entry, exit, min_exp`` (dates) plus any columns to carry along.
        root: The chains folder.

    Returns:
        The plans that found a straddle at entry, with ``exp, K, S0, cb0, ca0, pb0, pa0`` and, at
        the exit, ``S1, cb1, ca1, pb1, pa1`` (NaN where a leg is not quoted then).
    """
    days = set(pd.to_datetime(plans["entry"])) | set(pd.to_datetime(plans["exit"]))
    months = {d.strftime("%Y-%m") for d in days}
    parts = []
    for f in sorted((root / symbol).glob("*.csv.gz")):
        if f.name[:7] not in months:
            continue
        try:
            raw = pd.read_csv(f, usecols=CHAIN_COLS)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        raw = raw[raw["optionSymbol"].str[:-15] == option_root(symbol)]
        raw = raw.assign(date=pd.to_datetime(raw["date"]))
        parts.append(raw[raw["date"].isin(days)])
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    exp = pd.to_datetime(d["expiration"], unit="s", utc=True).dt.tz_convert(NY)
    d["exp"] = exp.dt.tz_localize(None).dt.normalize()
    key = ["date", "exp", "strike"]
    side = {s: d[d["side"] == s].drop_duplicates(key)[key + ["bid", "ask", "underlyingPrice"]]
            for s in ("call", "put")}
    legs = side["call"].merge(side["put"].drop(columns="underlyingPrice"), on=key,
                              suffixes=("_c", "_p"))
    legs = legs.rename(columns={"bid_c": "cb", "ask_c": "ca", "bid_p": "pb", "ask_p": "pa",
                                "underlyingPrice": "S"})
    good = legs[(legs["cb"] > 0) & (legs["pb"] > 0) & (legs["ca"] >= legs["cb"])
                & (legs["pa"] >= legs["pb"])]
    mid = (good["cb"] + good["ca"] + good["pb"] + good["pa"]) / 2
    good = good[((good["ca"] + good["pa"] - good["cb"] - good["pb"]) / mid <= MAX_SPREAD)
                & ((good["strike"] / good["S"] - 1).abs() <= MAX_STRIKE_GAP)]
    rows = []
    for plan in plans.itertuples(index=False):
        g = good[(good["date"] == pd.Timestamp(plan.entry))
                 & (good["exp"] >= pd.Timestamp(plan.min_exp))]
        if g.empty:
            continue
        g = g[g["exp"] == g["exp"].min()]
        g = g.iloc[[int(np.argmin((g["strike"] - g["S"]).abs().to_numpy()))]].iloc[0]
        x = legs[(legs["date"] == pd.Timestamp(plan.exit)) & (legs["exp"] == g["exp"])
                 & (legs["strike"] == g["strike"])]
        x = x.iloc[0] if len(x) else None
        rows.append({**plan._asdict(), "symbol": symbol, "exp": g["exp"], "K": g["strike"],
                     "S0": g["S"], "cb0": g["cb"], "ca0": g["ca"], "pb0": g["pb"], "pa0": g["pa"],
                     **{f"{c}1": (x[c] if x is not None else np.nan)
                        for c in ("S", "cb", "ca", "pb", "pa")}})
    return pd.DataFrame(rows)
