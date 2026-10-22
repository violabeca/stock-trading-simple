"""Daily option-chain summaries from MarketData.app's end-of-day chains on local disk.

``scripts/marketdata_dump.py`` downloads every optionable name's whole chain once per
session (16:00 New York quotes, all expiries) into ``var/marketdata/chains/<SYMBOL>/`` —
one gzip CSV per month. The files carry no implied volatilities, so this module solves
them (Black-Scholes on the mid quote, no dividends, :data:`RATE`) and reduces each
name-day to a handful of numbers (:data:`COLUMNS`). ``features.py`` serves them as the
``opt_*`` features, point-in-time like every other level.

Reading ~23,000 chain files takes minutes, so :func:`build_cache` keeps one summary per
source file under ``var/options/`` and recomputes only files that changed (the dump
appends to the current month's file every day), then writes the combined table that
:func:`load_summaries` reads.

The reduction, per name and date:

* the expiry nearest 30 calendar days among 10 to 60 days out (a shorter one wins a tie);
* ``atm_iv``: the mean of the call and put IV at the strike nearest the price among strikes
  quoted on both sides (bid > 0, ask > bid);
* ``skew``: the IV of the put nearest 95% of the price (strikes at 85-97%) minus the ATM
  call's IV — what downside protection costs over the at-the-money option (Xing, Zhang &
  Zhao 2010); ``put_skew``: the same put minus the ATM put, which no dividend can bend;
* ``cpiv``: the open-interest-weighted mean of call IV minus put IV over strikes quoted on
  both sides at 90-110% of the price (Cremers & Weinbaum 2010; equal weights without open
  interest). Dividends make calls look cheap, so its LEVEL is biased; changes are not;
* ``call_volume`` / ``put_volume`` / ``call_oi`` / ``put_oi``: contracts over every expiry.

Contracts on a non-standard OCC root (adjusted after a split or merger) are left out.
"""

from __future__ import annotations

import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from trading_simple.config import PROJECT_ROOT

CHAINS = PROJECT_ROOT / "var" / "marketdata" / "chains"
CACHE = PROJECT_ROOT / "var" / "options"
RATE = 0.04            # the yearly risk-free rate option prices are read with (US bills 2025-26)
NY = "America/New_York"
COLUMNS = ("spot", "atm_iv", "skew", "put_skew", "cpiv", "call_volume", "put_volume",
           "call_oi", "put_oi")
_READ = ["date", "optionSymbol", "expiration", "side", "strike", "bid", "ask", "mid",
         "openInterest", "volume", "underlyingPrice"]


def _ncdf(x):
    """The standard normal distribution function (Abramowitz & Stegun 7.1.26, within 1.5e-7)."""
    z = np.abs(x) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * erf)


def bs_price(s, k, t, sigma, call, r=RATE):
    """European option value by Black-Scholes without dividends (arrays welcome).

    Args:
        s, k: Stock price and strike.
        t: Years to expiry.
        sigma: Yearly volatility (0.3 = 30%).
        call: True for a call, False for a put.
        r: Yearly risk-free rate.

    Returns:
        The option value per share.
    """
    sq = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / sq
    disc = k * np.exp(-r * t)
    return np.where(call, s * _ncdf(d1) - disc * _ncdf(d1 - sq),
                    disc * _ncdf(sq - d1) - s * _ncdf(-d1))


def implied_vol(price, s, k, t, call, r=RATE, *, lo: float = 0.01, hi: float = 5.0,
                steps: int = 50):
    """The yearly volatility at which :func:`bs_price` gives ``price``, by bisection.

    Args:
        price, s, k, t, call, r: As :func:`bs_price` (arrays welcome).
        lo, hi: The volatility range searched.
        steps: Halvings of the range (50 is far below a quote's tick).

    Returns:
        The implied volatility; NaN where no volatility in ``[lo, hi]`` gives the price (a
        quote below the option's value at ``lo``: stale or crossed).
    """
    price, s, k, t = (np.asarray(v, float) for v in (price, s, k, t))
    call = np.asarray(call, bool)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(steps):
        m = (a + b) / 2
        high = bs_price(s, k, t, m, call, r) > price
        a, b = np.where(high, a, m), np.where(high, m, b)
    inside = (price > bs_price(s, k, t, lo, call, r)) & (price < bs_price(s, k, t, hi, call, r))
    return np.where(inside, (a + b) / 2, np.nan)


def option_root(symbol: str) -> str:
    """The OCC root of ``symbol``'s standard contracts (punctuation dropped: BRK.B -> BRKB)."""
    return symbol.replace(".", "").replace("-", "")


def summarize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reduce one name's end-of-day chain rows to one row per date (see the module docstring).

    Args:
        raw: MarketData chain rows (at least the columns in ``_READ``), any number of dates.
        symbol: The ticker; contracts on another OCC root are dropped.

    Returns:
        Columns ``symbol``, ``d`` (the quote date) and :data:`COLUMNS`; NaN where a number
        cannot be read that day (no expiry in range, no quoted strike near the money).
    """
    d = raw[raw["optionSymbol"].astype(str).str[:-15] == option_root(symbol)]
    if d.empty:
        return pd.DataFrame(columns=["symbol", "d", *COLUMNS])
    date = pd.to_datetime(d["date"])
    exp = pd.to_datetime(d["expiration"], unit="s", utc=True).dt.tz_convert(NY)
    d = d.assign(date=date, dte=(exp.dt.tz_localize(None).dt.normalize() - date).dt.days)
    day = d.groupby("date").agg(spot=("underlyingPrice", "median"))
    flows = d.pivot_table(index="date", columns="side", values=["volume", "openInterest"],
                          aggfunc="sum")
    for col, key in (("call_volume", ("volume", "call")), ("put_volume", ("volume", "put")),
                     ("call_oi", ("openInterest", "call")), ("put_oi", ("openInterest", "put"))):
        day[col] = flows[key] if key in flows.columns else 0.0

    q = d[(d["bid"] > 0) & (d["ask"] > d["bid"]) & d["dte"].between(10, 60)]
    q = q.join(day["spot"], on="date")
    q = q.assign(m=q["strike"] / q["spot"])
    q = q[q["m"].between(0.85, 1.10)]
    key = (q["dte"] - 30).abs() * 1000 + q["dte"]                   # nearest 30 days, shorter tie
    q = q[key == key.groupby(q["date"]).transform("min")]
    q = q.assign(iv=implied_vol(q["mid"], q["spot"], q["strike"], q["dte"] / 365.0,
                                (q["side"] == "call").to_numpy()))
    calls = q[q["side"] == "call"].set_index(["date", "strike"])[["iv", "openInterest"]]
    puts = q[q["side"] == "put"].set_index(["date", "strike"])[["iv", "openInterest"]]
    both = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner").reset_index()
    both = both.join(day["spot"], on="date")
    both["dist"] = (both["strike"] - both["spot"]).abs()
    atm = both.sort_values(["date", "dist", "strike"]).drop_duplicates("date").set_index("date")
    day["atm_iv"] = atm[["iv_c", "iv_p"]].mean(axis=1)
    otm = q[(q["side"] == "put") & q["m"].between(0.85, 0.97) & q["iv"].notna()]
    otm = otm.assign(dist=(otm["m"] - 0.95).abs()).sort_values(["date", "dist"])
    otm = otm.drop_duplicates("date").set_index("date")["iv"]
    day["skew"] = otm - atm["iv_c"]
    day["put_skew"] = otm - atm["iv_p"]
    near = both[(both["strike"] / both["spot"]).between(0.90, 1.10)
                & both["iv_c"].notna() & both["iv_p"].notna()]
    near = near.assign(diff=near["iv_c"] - near["iv_p"],
                       w=(near["openInterest_c"] + near["openInterest_p"]).astype(float))
    wsum = near.groupby("date")["w"].sum()
    weighted = (near["diff"] * near["w"]).groupby(near["date"]).sum() / wsum
    day["cpiv"] = weighted.where(wsum > 0, near.groupby("date")["diff"].mean())
    out = day.reindex(columns=list(COLUMNS)).reset_index().rename(columns={"date": "d"})
    out.insert(0, "symbol", symbol)
    return out


def _summarize_file(path: Path, symbol: str) -> pd.DataFrame:
    """:func:`summarize` of one chain file (an empty or unreadable file gives no rows)."""
    try:
        raw = pd.read_csv(path, usecols=_READ)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=["symbol", "d", *COLUMNS])
    return summarize(raw, symbol)


def _symbol_cache(args) -> pd.DataFrame:
    """One name: reuse each cached file summary whose source is unchanged, redo the rest."""
    symbol, root, cache = args
    store = Path(cache) / f"{symbol}.pkl"
    old = pickle.loads(store.read_bytes()) if store.exists() else {}
    new, changed = {}, False
    for f in sorted((Path(root) / symbol).glob("*.csv.gz")):
        stamp = (f.stat().st_size, f.stat().st_mtime_ns)
        hit = old.get(f.name)
        if hit is None or hit[0] != stamp:
            hit, changed = (stamp, _summarize_file(f, symbol)), True
        new[f.name] = hit
    if changed or set(new) != set(old):
        store.write_bytes(pickle.dumps(new))
    frames = [v[1] for v in new.values() if len(v[1])]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_cache(symbols: list[str] | None = None, *, root: Path = CHAINS, cache: Path = CACHE,
                workers: int = 8) -> pd.DataFrame:
    """Summarize every chain file not yet cached (or changed since), and write the table.

    Local files only (no API calls), so the names run in parallel. The first build reads
    every file (several minutes); later builds redo only the files the daily dump touched.

    Args:
        symbols: Names to build; every folder under ``root`` when omitted.
        root: The chains folder (``var/marketdata/chains``).
        cache: Where the per-name summaries and ``summary.pkl`` (the combined table) live.
        workers: Processes.

    Returns:
        The combined table (:func:`load_summaries`' format), also written to
        ``cache / "summary.pkl"``.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    symbols = sorted(symbols or [p.name for p in Path(root).iterdir() if p.is_dir()])
    jobs = [(s, str(root), str(cache)) for s in symbols]
    if workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(_symbol_cache, jobs, chunksize=4)
    else:
        parts = [_symbol_cache(j) for j in jobs]
    table = pd.concat([p for p in parts if len(p)], ignore_index=True)
    table["d"] = pd.to_datetime(table["d"])
    tmp = cache / "summary.pkl.tmp"
    tmp.write_bytes(pickle.dumps(table))
    os.replace(tmp, cache / "summary.pkl")
    return table


_LOADED: dict = {}


def load_summaries(cache: Path = CACHE) -> pd.DataFrame:
    """The combined summary table written by :func:`build_cache` (kept in memory per file stamp).

    Returns:
        One row per (``symbol``, ``d``) with :data:`COLUMNS`; ``d`` is the quote date (16:00
        New York close of that session).

    Raises:
        FileNotFoundError: If no cache was built yet (run :func:`build_cache` — notebook
            ``03_altdata_update``, section "Option chains").
    """
    path = Path(cache) / "summary.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: build it with "
                                "providers.marketdata.build_cache() (notebook 03)")
    stamp = path.stat().st_mtime_ns
    if _LOADED.get("key") != (str(path), stamp):
        _LOADED.update(key=(str(path), stamp), table=pd.read_pickle(path))
    return _LOADED["table"]
