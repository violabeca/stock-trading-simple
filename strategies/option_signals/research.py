"""option_signals research helpers: implied volatilities from end-of-day chains, four published
option signals, and a rank-IC gate against market-neutral forward stock returns.

Loaded by path from ``option_signals.ipynb`` (as ``os_rs``), from the options_hedge premia
notebook and from ``tests/test_option_signals.py``; the framework never imports it. The test is
pre-registered in this folder's README.

Data: MarketData.app chains in ``var/marketdata/chains/`` (16:00 New York quotes, no implied
volatility), daily bars from the shared DB. Derived tables are cached in ``var/option_signals/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "var" / "marketdata" / "chains"
CACHE = ROOT / "var" / "option_signals"
NY = "America/New_York"

R = 0.0375                  # risk-free rate, constant over the year studied
DTE_BAND = (21, 60, 30)     # expiry A: days out, closest to 30 (shorter on a tie)
ATM_BAND = (0.97, 1.03)     # K/S of the at-the-money call and put
OTM_PUT = (0.90, 0.975, 0.95)   # K/S band and target of the out-of-the-money put
PAIR_BAND = (0.90, 1.10)    # K/S band of the call-put pairs behind cpiv
MONEYNESS = 0.20            # contracts priced at most this far from the price
UNIVERSE = 500              # names per day, by trailing median dollar volume
MIN_PRICE = 5.0
HORIZONS = (1, 5, 21)       # forward sessions
BETA_DAYS = 120
SIGNALS = {"skew": -1, "cpiv": 1, "vrp": -1, "os": -1}   # signal -> published sign
T_BAR, T_PARTIAL, MONTH_SHARE = 2.65, 2.0, 0.6
CHAIN_COLS = ["date", "optionSymbol", "expiration", "side", "strike", "bid", "ask",
              "underlyingPrice", "volume", "openInterest"]


# --- Black-Scholes ---------------------------------------------------------------------------

def bs_price(S, K, T, r, q, sigma, call):
    """European Black-Scholes price with a continuous dividend yield (vectorized).

    Args:
        S: Stock price.
        K: Strike.
        T: Years to expiry.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        call: True for a call, False for a put (array or scalar).

    Returns:
        The option price.
    """
    S, K, T, q, sigma = (np.asarray(x, float) for x in (S, K, T, q, sigma))
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    c = fs * ndtr(d1) - fk * ndtr(d2)
    p = fk * ndtr(-d2) - fs * ndtr(-d1)
    return np.where(call, c, p)


def implied_vol(price, S, K, T, r, q, call, lo: float = 1e-3, hi: float = 5.0,
                iters: int = 60) -> np.ndarray:
    """Black-Scholes implied volatility by bisection (vectorized).

    Args:
        price: Option price (the mid).
        S: Stock price.
        K: Strike.
        T: Years to expiry.
        r: Risk-free rate.
        q: Dividend yield.
        call: True for calls.
        lo: Lowest volatility searched.
        hi: Highest volatility searched.
        iters: Bisection steps.

    Returns:
        Implied volatility; NaN where the price is outside the no-arbitrage bounds (at or below the
        discounted intrinsic value, or above the bound reached at ``hi``).
    """
    price, S, K, T, q = (np.asarray(x, float) for x in (price, S, K, T, q))
    call = np.broadcast_to(np.asarray(call, bool), price.shape)
    low = np.full(price.shape, lo)
    high = np.full(price.shape, hi)
    floor = bs_price(S, K, T, r, q, low, call)
    cap = bs_price(S, K, T, r, q, high, call)
    ok = (price > floor) & (price < cap) & (T > 0)
    for _ in range(iters):
        mid = 0.5 * (low + high)
        above = bs_price(S, K, T, r, q, mid, call) > price
        high = np.where(above, mid, high)
        low = np.where(above, low, mid)
    return np.where(ok, 0.5 * (low + high), np.nan)


def dividend_events(close: pd.DataFrame, adj: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every name's ex-dividend dates and sizes, read from the DB's adjusted ÷ raw close ratio.

    The ratio steps up on each ex-date by 1 ÷ (1 − dividend ÷ previous close), so the size comes out
    as a fraction of the price (scale-free: it applies to the chain's own price even for a name
    whose DB prices were later split-adjusted).

    Args:
        close: Raw close (sessions × symbols).
        adj: Adjusted close, same shape.

    Returns:
        ``{symbol: DataFrame(date, y)}`` with ``y`` = dividend ÷ price (only payers appear).
    """
    ratio = (adj / close).ffill()
    step = ratio / ratio.shift(1)
    out = {}
    for s in step.columns:
        e = step[s][step[s] > 1 + 1e-4]
        if len(e):
            out[s] = pd.DataFrame({"date": e.index, "y": (1 - 1 / e).to_numpy()})
    return out


def dividend_fraction(events: pd.DataFrame | None, dates, exps, r: float = R) -> np.ndarray:
    """Present value of the dividends projected inside (date, expiry], as a fraction of the price.

    Only ex-dates known by ``date`` are used. The next ex-date is the last one plus the median gap
    of the last four (365 days with a single event), repeated up to the expiry; each dividend is the
    median size of the last four. A projected date already passed by less than 30 days is taken as
    imminent (the next session); one passed by more is taken as stopped.

    Args:
        events: The name's :func:`dividend_events` frame (``None`` = no dividends).
        dates: Valuation dates.
        exps: Expiry dates, same length.
        r: Discount rate.

    Returns:
        PV fraction per (date, expiry) pair; 0 without dividends.
    """
    out = np.zeros(len(dates))
    if events is None or events.empty:
        return out
    ev = events["date"].to_numpy(dtype="datetime64[D]")
    ys = events["y"].to_numpy()
    for i, (d, e) in enumerate(zip(np.asarray(dates, dtype="datetime64[D]"),
                                   np.asarray(exps, dtype="datetime64[D]"), strict=True)):
        k = int(np.searchsorted(ev, d, side="right"))
        if k == 0:
            continue
        gaps = np.diff(ev[max(0, k - 5):k]).astype(int)
        gap = int(np.median(gaps)) if len(gaps) else 365
        y = float(np.median(ys[max(0, k - 4):k]))
        nxt = ev[k - 1] + np.timedelta64(gap, "D")
        if nxt <= d:
            if (d - nxt).astype(int) >= 30:
                continue
            nxt = d + np.timedelta64(1, "D")
        while nxt <= e:
            out[i] += y * np.exp(-r * (nxt - d).astype(int) / 365)
            nxt = nxt + np.timedelta64(gap, "D")
    return out


# --- chains -> one row per name-day ---------------------------------------------------------

def _nearest(x: pd.DataFrame, key: pd.Series, by: str = "date") -> pd.DataFrame:
    """One row per ``by`` value: the smallest ``key`` (ties to the lower strike)."""
    x = x.assign(_k=key)
    return x.sort_values([by, "_k", "strike"]).drop_duplicates(by).set_index(by)


def surface_day_summary(raw: pd.DataFrame, symbol: str, divs: pd.DataFrame | None = None,
                        r: float = R) -> pd.DataFrame:
    """Reduce one name's chain rows to one row per date: volumes and the IVs behind the signals.

    Expiry A is the expiry 21–60 days out closest to 30 among those quoting an ATM call and put
    (K/S in :data:`ATM_BAND`, bid > 0). On it, every standard contract with bid > 0, ask ≥ bid and
    K/S within ±:data:`MONEYNESS` gets a Black-Scholes IV on its mid, with the price net of the
    present value of the dividends projected before the expiry (:func:`dividend_fraction`).

    Args:
        raw: MarketData chain rows (:data:`CHAIN_COLS`).
        symbol: The ticker; non-standard roots are dropped.
        divs: The name's :func:`dividend_events` (``None`` = no dividends).
        r: Risk-free rate.

    Returns:
        One row per date: ``symbol, date, S, call_vol, put_vol, call_oi, put_oi, exp, T, div,
        iv_call, iv_put, K_atm_call, K_atm_put, iv_otm_put, K_otm_put, otm_put_bid, otm_put_ask,
        cpiv, n_pairs`` (NaN where not available).
    """
    d = raw[raw["optionSymbol"].str[:-15] == symbol.replace(".", "").replace("-", "")]
    if d.empty:
        return pd.DataFrame()
    exp = pd.to_datetime(d["expiration"], unit="s", utc=True).dt.tz_convert(NY)
    d = d.assign(date=pd.to_datetime(d["date"]), exp=exp.dt.tz_localize(None).dt.normalize(),
                 volume=d["volume"].fillna(0), openInterest=d["openInterest"].fillna(0))
    is_call = d["side"] == "call"
    day = pd.DataFrame({
        "S": d.groupby("date")["underlyingPrice"].median(),
        "call_vol": d[is_call].groupby("date")["volume"].sum(),
        "put_vol": d[~is_call].groupby("date")["volume"].sum(),
        "call_oi": d[is_call].groupby("date")["openInterest"].sum(),
        "put_oi": d[~is_call].groupby("date")["openInterest"].sum()})
    day[["call_vol", "put_vol", "call_oi", "put_oi"]] = day[
        ["call_vol", "put_vol", "call_oi", "put_oi"]].fillna(0)
    x = d.join(day["S"].rename("Sd"), on="date")
    x = x.assign(dte=(x["exp"] - x["date"]).dt.days, m=x["strike"] / x["Sd"])
    lo, hi, target = DTE_BAND
    x = x[x["dte"].between(lo, hi) & (x["bid"] > 0) & (x["ask"] >= x["bid"])
          & ((x["m"] - 1).abs() <= MONEYNESS)]
    atm = x[x["m"].between(*ATM_BAND)]
    both = atm.groupby(["date", "exp"])["side"].nunique()
    both = both[both == 2].reset_index()
    out = day.copy()
    if both.empty:
        return out.reset_index().assign(symbol=symbol)
    both["ekey"] = (both["exp"] - both["date"]).dt.days.sub(target).abs() * 1000 + \
        (both["exp"] - both["date"]).dt.days
    pick = both.sort_values(["date", "ekey"]).drop_duplicates("date")[["date", "exp"]]
    x = x.merge(pick, on=["date", "exp"])
    x["T"] = x["dte"] / 365.0
    frac = dividend_fraction(divs, pick["date"], pick["exp"], r)
    x = x.merge(pick.assign(div=frac), on=["date", "exp"])
    x["iv"] = implied_vol((x["bid"] + x["ask"]) / 2, x["Sd"] * (1 - x["div"]), x["strike"],
                          x["T"], r, 0.0, x["side"].eq("call").to_numpy())
    x = x[x["iv"].notna()]
    call, put = x[x["side"] == "call"], x[x["side"] == "put"]
    first = x.drop_duplicates("date").set_index("date")
    out["exp"], out["T"], out["div"] = first["exp"], first["T"], first["div"]
    a = call[call["m"].between(*ATM_BAND)]
    a = _nearest(a, (a["m"] - 1).abs())
    out["iv_call"], out["K_atm_call"] = a["iv"], a["strike"]
    a = put[put["m"].between(*ATM_BAND)]
    a = _nearest(a, (a["m"] - 1).abs())
    out["iv_put"], out["K_atm_put"] = a["iv"], a["strike"]
    a = put[put["m"].between(OTM_PUT[0], OTM_PUT[1])]
    a = _nearest(a, (a["m"] - OTM_PUT[2]).abs())
    out["iv_otm_put"], out["K_otm_put"] = a["iv"], a["strike"]
    out["otm_put_bid"], out["otm_put_ask"] = a["bid"], a["ask"]
    pairs = call.merge(put, on=["date", "strike"], suffixes=("_c", "_p"))
    pairs = pairs[pairs["m_c"].between(*PAIR_BAND)]
    w = pairs["openInterest_c"] + pairs["openInterest_p"]
    wsum = w.groupby(pairs["date"]).transform("sum")
    w = np.where(wsum > 0, w, 1.0)
    diff = pairs["iv_c"] - pairs["iv_p"]
    g = pd.DataFrame({"date": pairs["date"], "wd": w * diff, "w": w}).groupby("date").sum()
    out["cpiv"] = g["wd"] / g["w"]
    out["n_pairs"] = pairs.groupby("date").size()
    return out.reset_index().assign(symbol=symbol)


def read_surfaces(symbol: str, first: str, last: str, divs: pd.DataFrame | None = None,
                  root: Path = CHAINS) -> pd.DataFrame:
    """:func:`surface_day_summary` over one name's monthly files ``first``..``last`` (``YYYY-MM``).

    Args:
        symbol: Ticker.
        first: First month.
        last: Last month.
        divs: The name's :func:`dividend_events`.
        root: The chains folder.

    Returns:
        Concatenated rows (empty files skipped).
    """
    parts = []
    for f in sorted((root / symbol).glob("*.csv.gz")):
        if not first <= f.name[:7] <= last:
            continue
        try:
            raw = pd.read_csv(f, usecols=CHAIN_COLS)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        parts.append(surface_day_summary(raw, symbol, divs))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def surfaces(first: str, last: str, divs: dict[str, pd.DataFrame], symbols: list[str],
             root: Path = CHAINS, cache: Path = CACHE) -> pd.DataFrame:
    """Every name's daily surface summary for months ``first``..``last``, cached (pickle).

    Args:
        first: First month (``YYYY-MM``).
        last: Last month.
        divs: :func:`dividend_events` of every name.
        symbols: Names to read.
        root: The chains folder.
        cache: Cache folder.

    Returns:
        The concatenated :func:`surface_day_summary` rows.
    """
    path = cache / f"surf_{first}_{last}_{len(symbols)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    parts = []
    for k, s in enumerate(sorted(symbols), 1):
        parts.append(read_surfaces(s, first, last, divs.get(s), root))
        if k % 100 == 0:
            print(f"\r{k}/{len(symbols)} surfaces", end="")
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    cache.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def load_daily(symbols: list[str], start: str, end: str, engine=None,
               cache: Path = CACHE) -> pd.DataFrame:
    """Daily bars from the shared DB (long: ``symbol, date, open, close, adj_close, volume``).

    Args:
        symbols: Tickers.
        start: First date.
        end: Last date (inclusive).
        engine: SQLAlchemy engine (``make_engine()`` when omitted).
        cache: Cache folder.

    Returns:
        Long frame; ``date`` is the trade date (bars are stored at 00:00 UTC of it).
    """
    path = cache / f"daily_{start}_{end}_{len(symbols)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from sqlalchemy import text

    from trading_simple.db import make_engine
    sql = text("SELECT symbol, ts, open, close, adj_close, volume FROM prices_ohlcv "
               "WHERE bar = '1d' AND symbol = ANY(:s) AND ts >= :a AND ts <= :b")
    with (engine or make_engine()).connect() as conn:
        long = pd.read_sql(sql, conn, params={"s": list(symbols),
                                              "a": pd.Timestamp(start, tz="UTC"),
                                              "b": pd.Timestamp(end, tz="UTC")})
    long["date"] = pd.to_datetime(long.pop("ts"), utc=True).dt.tz_localize(None).dt.normalize()
    cache.mkdir(parents=True, exist_ok=True)
    long.to_pickle(path)
    return long


# --- signals, universe, outcomes -------------------------------------------------------------

def wide(long: pd.DataFrame, col: str, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Pivot a long ``symbol, date`` frame to sessions × symbols."""
    return long.pivot(index="date", columns="symbol", values=col).reindex(index=index,
                                                                          columns=columns)


def signals(surf: dict[str, pd.DataFrame], adj: pd.DataFrame, volume: pd.DataFrame) -> dict:
    """The four pre-registered signals on day t (sessions × symbols), each from data through t.

    Args:
        surf: Wide surface columns: ``iv_call, iv_put, iv_otm_put, cpiv, call_vol, put_vol``.
        adj: Adjusted close.
        volume: Share volume.

    Returns:
        ``{"skew", "cpiv", "vrp", "os"}`` frames plus ``iv_atm`` and ``rv21``.
    """
    iv_atm = (surf["iv_call"] + surf["iv_put"]) / 2
    r = np.log(adj).diff()
    rv21 = r.rolling(21, min_periods=21).std() * np.sqrt(252)
    opt = (surf["call_vol"] + surf["put_vol"]) * 100
    os_ = opt.rolling(5, min_periods=5).sum() / volume.rolling(5, min_periods=5).sum()
    return {"skew": surf["iv_otm_put"] - surf["iv_call"], "cpiv": surf["cpiv"],
            "vrp": rv21 - iv_atm, "os": os_.where(np.isfinite(os_)), "iv_atm": iv_atm,
            "rv21": rv21}


def universe(iv_atm: pd.DataFrame, price: pd.DataFrame, dollar_vol: pd.DataFrame,
             n: int = UNIVERSE, min_price: float = MIN_PRICE) -> pd.DataFrame:
    """Names on day t: ATM IV known, price ≥ ``min_price``, top ``n`` by 20-session median $ volume.

    Args:
        iv_atm: ATM IV (NaN = no ATM quotes).
        price: The chain's stock price (raw).
        dollar_vol: Daily dollar volume.
        n: Names kept per day.
        min_price: Minimum price.

    Returns:
        Boolean frame.
    """
    med = dollar_vol.rolling(20, min_periods=10).median()
    ok = iv_atm.notna() & (price >= min_price) & med.notna()
    rank = med.where(ok).rank(axis=1, ascending=False, method="first")
    return ok & (rank <= n)


def market_beta(adj: pd.DataFrame, days: int = BETA_DAYS) -> tuple[pd.DataFrame, pd.Series]:
    """Rolling OLS beta of each name's daily return on the equal-weight return of all names.

    Args:
        adj: Adjusted close (sessions × symbols).
        days: Window (sessions, through t).

    Returns:
        ``(beta frame, market return series)``.
    """
    r = adj.pct_change(fill_method=None)
    m = r.mean(axis=1)
    cov = r.rolling(days, min_periods=days // 2).cov(m)
    var = m.rolling(days, min_periods=days // 2).var()
    return cov.div(var, axis=0), m


def residual_forward(open_adj: pd.DataFrame, beta: pd.DataFrame, mask: pd.DataFrame,
                     h: int) -> pd.DataFrame:
    """Forward return from t+1's open to t+1+h's open, minus beta × the universe's mean of it.

    Args:
        open_adj: Adjusted open.
        beta: Beta known at t.
        mask: The day's universe.
        h: Sessions held.

    Returns:
        Residual forward return indexed by the signal day t (NaN off the universe).
    """
    fwd = open_adj.shift(-(1 + h)) / open_adj.shift(-1) - 1
    fwd = fwd.where(mask)
    mkt = fwd.mean(axis=1)
    return fwd - beta.where(mask).mul(mkt, axis=0)


def daily_ic(signal: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
             min_names: int = 50) -> pd.Series:
    """Each day's Spearman correlation between the signal and the forward return over ``mask``.

    Args:
        signal: Signal (sessions × symbols).
        fwd: Forward (residual) return.
        mask: Names scored per day.
        min_names: Days with fewer usable names are NaN.

    Returns:
        IC per day.
    """
    ok = mask & signal.notna() & fwd.notna()
    a = signal.where(ok).rank(axis=1)
    b = fwd.where(ok).rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    ic = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    return ic.where(ok.sum(axis=1) >= min_names)


def residualize(signal: pd.DataFrame, controls: list[pd.DataFrame], mask: pd.DataFrame,
                min_names: int = 50) -> pd.DataFrame:
    """The signal minus its cross-sectional OLS fit on ``controls``, day by day.

    Args:
        signal: Signal.
        controls: Control frames, same shape.
        mask: Names used per day.
        min_names: Days with fewer complete names are left NaN.

    Returns:
        Residual signal (NaN where any input is missing).
    """
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    S = signal.to_numpy()
    C = np.stack([c.to_numpy() for c in controls], axis=-1)
    M = mask.to_numpy()
    for i in range(len(signal)):
        ok = M[i] & np.isfinite(S[i]) & np.isfinite(C[i]).all(axis=1)
        if ok.sum() < min_names:
            continue
        X = np.column_stack([np.ones(ok.sum()), C[i][ok]])
        coef, *_ = np.linalg.lstsq(X, S[i][ok], rcond=None)
        out.iloc[i, np.flatnonzero(ok)] = S[i][ok] - X @ coef
    return out


def nw_t(x: pd.Series, lag: int) -> tuple[float, float]:
    """Mean and Newey-West t-statistic of a series (Bartlett weights up to ``lag``).

    Args:
        x: Values (NaN dropped).
        lag: Autocorrelation lags.

    Returns:
        ``(mean, t)``; t is NaN with fewer than 10 values.
    """
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 10:
        return float(np.mean(v)) if n else np.nan, np.nan
    e = v - v.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(v.mean()), float(v.mean() / np.sqrt(s / n))


def quintile_spread(signal: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
                    sign: int) -> tuple[pd.Series, pd.Series]:
    """Top-minus-bottom-quintile mean forward return per day, oriented by the expected ``sign``.

    Args:
        signal: Signal.
        fwd: Forward (residual) return.
        mask: Names scored.
        sign: +1 when a high signal should outperform, −1 when it should underperform.

    Returns:
        ``(spread per day, one-way turnover of the two legs per day)``: the share of names that
        enter or leave the legs between consecutive days, averaged over both legs.
    """
    ok = mask & signal.notna() & fwd.notna()
    pct = (sign * signal).where(ok).rank(axis=1, pct=True)
    top, bot = pct > 0.8, pct <= 0.2
    spread = fwd.where(top).mean(axis=1) - fwd.where(bot).mean(axis=1)
    turn = []
    for leg in (top, bot):
        n = leg.sum(axis=1).replace(0, np.nan)
        changed = (leg & ~leg.shift(1, fill_value=False)).sum(axis=1)
        turn.append(changed / n)
    return spread, (turn[0] + turn[1]) / 2


def rebalanced_spread(signal: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame, sign: int,
                      h: int, cost_bp: float = 5.0) -> dict:
    """A long-short quintile book rebalanced every ``h`` sessions: gross and net of costs.

    On every h-th signal day the top fifth (by ``sign × signal``) is bought and the bottom fifth
    sold, equal-weighted, and held for the ``h``-session forward return. A name entering a leg is
    one trade in and one out of the name it replaces, at ``cost_bp`` each.

    Args:
        signal: Signal.
        fwd: The h-session forward (residual) return.
        mask: Names scored.
        sign: +1 when a high signal should outperform, −1 when it should underperform.
        h: Holding period (sessions) = rebalancing interval.
        cost_bp: One-way cost per trade (basis points).

    Returns:
        ``periods``, ``gross`` and ``net`` (annualized, 252 sessions), ``turnover`` (mean share of a
        leg replaced per rebalance), ``hit`` (share of periods with a positive gross spread).
    """
    ok = mask & signal.notna() & fwd.notna()
    days = ok.index[ok.sum(axis=1) >= 50][::h]
    pct = (sign * signal).where(ok).loc[days].rank(axis=1, pct=True)
    top, bot = pct > 0.8, pct <= 0.2
    f = fwd.loc[days]
    gross = f.where(top).mean(axis=1) - f.where(bot).mean(axis=1)
    turn = []
    for leg in (top, bot):
        new = (leg & ~leg.shift(1, fill_value=False)).sum(axis=1) / leg.sum(axis=1)
        turn.append(new.iloc[1:])
    tau = (turn[0] + turn[1]) / 2
    cost = 2 * 2 * tau.reindex(gross.index).fillna(1.0) * cost_bp / 1e4
    return {"periods": len(gross), "gross": float(gross.mean() * 252 / h),
            "net": float((gross - cost).mean() * 252 / h), "turnover": float(tau.mean()),
            "hit": float((gross > 0).mean())}


def score_signal(signal: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
                 controls: list[pd.DataFrame], dollar_vol: pd.DataFrame, sign: int, h: int) -> dict:
    """Every pre-registered number of one signal × horizon cell.

    Args:
        signal: Signal (sessions × symbols).
        fwd: Residual h-session forward return (:func:`residual_forward`).
        mask: The window's universe.
        controls: Frames the partial IC regresses the signal on.
        dollar_vol: Median dollar volume (for the terciles).
        sign: The published direction.
        h: Horizon (sessions; also the Newey-West lag).

    Returns:
        Dict: mean IC and its t, partial IC and t, monthly share, the minimum detectable IC, IC by
        dollar-volume tercile, the rebalanced quintile book, the pass flags and ``ic`` (per day).
    """
    ic = daily_ic(signal, fwd, mask)
    mean, t = nw_t(ic, h)
    pic = daily_ic(residualize(signal, controls, mask), fwd, mask)
    pmean, pt = nw_t(pic, h)
    months = ic.groupby(ic.index.to_period("M")).agg(["mean", "count"])
    months = months[months["count"] >= 10]
    share = float((sign * months["mean"] > 0).mean()) if len(months) else np.nan
    rank = dollar_vol.where(mask & signal.notna()).rank(axis=1, pct=True)
    terc = {f"ic_t{k + 1}": float(daily_ic(signal, fwd, mask & (rank > k / 3)
                                           & (rank <= (k + 1) / 3), min_names=30).mean())
            for k in range(3)}
    book = rebalanced_spread(signal, fwd, mask, sign, h)
    res = dict(days=int(ic.notna().sum()), ic=mean, t=t, pic=pmean, pt=pt, months=len(months),
               month_share=share, mde=(T_BAR + 0.84) * abs(mean / t) if t else np.nan, **terc,
               **{f"book_{k}": v for k, v in book.items()})
    res["passed"] = bool(sign * t >= T_BAR and sign * pt >= T_PARTIAL and share >= MONTH_SHARE)
    res["ic_series"] = ic
    return res
