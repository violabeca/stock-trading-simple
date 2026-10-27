"""proposals_lab — one shared yardstick for the STRATEGY_OVERVIEW.md proposals (2026-10-22).

Every proposal and cross-strategy is judged by the SAME code: point-in-time universes, warm
per-period backtests on panels built once, the same metrics, luck band, optimizer, noise check and
bear table. A research folder (no ``strategy.py``): load it by path, or use the CLI.

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lab", "/Users/janmarti/Documents/code/stock-trading-simple/strategies/proposals_lab/"
        "lab.py")
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)

CLI (run from the repo root with ``uv run python strategies/proposals_lab/lab.py ...``):

    build                                         once: universes, panels, earnings, SPY
    run <strategy> <label> '<overrides>' [tests] [--costs2] [--hedge]   per-period books
                                                  (tests default OPEN; --hedge adds SPY)
    report <strategy> <base> <label>... [--tests E0,TB,...]
    luck <strategy> <label> [test] [n]            shuffled-name books (default T1, 100)
    noise <strategy> <label> [k] [copies]         the same config on panels missing k names
    tune <strategy> <label> '<overrides>' '<space>' [n_trials] [tests]
    bear <strategy> <label>...                    falling-market episodes + calendar years
    verdict <strategy> <label> [--not-neutral]    the README's gates G1-G8

Tests (``TESTS``): E0 2016-07 → 2018-06, TB 2018-07 → 2020-06, T3 2020-07 → 2021-12 (the TRAIN
era), T1 2022-01 → 2026-10-16 (top 500), T2 names 501-1000 (2022-26) — the OPEN tests — and the
LOCKED T4 names 1001-1500, TL names 1501-2000 (2022-26). Each is a list of half-year periods; the
period's universe is ranked on the 6 months before it (informed_flow_ls's rolling top 500).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
VAR = REPO / "var" / "proposals_lab"
#: Lab version 2 (2026-10-23): the fixes the first round's adversarial reviews found — NYSE
#: sessions only (Swiss listings added three empty US-holiday rows to the 2026 panels, on which
#: books exited everything), each period's day 0 counted from the initial cash (book and
#: benchmark; it holds the opening fees and the engine's up-front borrow), ETFs typed "equity"
#: kept out of the stock universes, and the lab's own portfolio runner (:func:`run_portfolio`:
#: short covers execute before new shorts; small short re-trims defer their borrow). Version-1
#: results stay in ``results/``; version 2 writes ``results2/``.
LAB_VERSION = 2
PANELS, RESULTS, STUDIES = VAR / "panels", VAR / "results2", VAR / "studies2"
#: Sector SPDRs stored with security_type "equity" (no name) — ETFs, not stocks.
EXCLUDE = frozenset({"XLE", "XLF", "XLI", "XLK", "XLV", "XLY"})
LAST = "2026-10-16"          # the last daily bar in the DB
START = "2015-01-01"         # every panel starts here: 1.5 years of warm-up before E0
FEATURES = ["short_interest", "fund_shares", "fund_equity", "fund_assets", "fund_ni_ttm",
            "fund_gp_ttm"]
ROLL = {"top_n": 500, "months": 6, "min_dollar_volume": 50e6, "min_price": 5.0, "min_days": 100}
N_BOOT, BOOT_BLOCK = 2000, 21
EPISODES = {"Q4 2018": ("2018-09-20", "2018-12-24"), "COVID crash": ("2020-02-19", "2020-03-23"),
            "2022 bear": ("2022-01-03", "2022-10-12"), "Feb-Apr 2025": ("2025-02-19", "2025-04-08")}


def halves(first: str, last_origin: str) -> list[tuple[str, str]]:
    """Half-year test periods ``(start, end)`` from ``first`` to the period starting at
    ``last_origin``; the last one ends at :data:`LAST` at the latest."""
    out = []
    for o in pd.date_range(first, last_origin, freq="6MS"):
        end = min(o + pd.DateOffset(months=6) - pd.Timedelta(days=1), pd.Timestamp(LAST))
        out.append((str(o.date()), str(end.date())))
    return out


_T1 = halves("2022-01-01", "2026-07-01")
TESTS = {"E0": ("top", halves("2016-07-01", "2018-01-01")),
         "TB": ("top", halves("2018-07-01", "2020-01-01")),
         "T3": ("top", halves("2020-07-01", "2021-07-01")),
         "T1": ("top", _T1),
         "T2": ("band", 500, 1000, 10e6, _T1),
         "T4": ("band", 1000, 1500, 3e6, _T1),
         "TL": ("band", 1500, 2000, 1e6, _T1)}
TRAIN, VALID, LOCKED = ["E0", "TB", "T3"], ["T1", "T2"], ["T4", "TL"]
OPEN = TRAIN + VALID


def periods(test: str) -> list[tuple[str, str]]:
    return TESTS[test][-1]


# --- data: universes and panels ---------------------------------------------------------------

def _engine():
    from trading_simple.db import make_engine
    return make_engine()


def _settings():
    from trading_simple.config import load_settings
    return load_settings()


def ranked(te0: str, engine=None) -> pd.DataFrame:
    """Every equity's median dollar volume, median price and session count over the
    ``ROLL["months"]`` before ``te0`` (cached in ``ranked.pkl``)."""
    from sqlalchemy import text
    store = VAR / "ranked.pkl"
    cache = pickle.loads(store.read_bytes()) if store.exists() else {}
    if te0 not in cache:
        start = pd.Timestamp(te0) - pd.DateOffset(months=ROLL["months"])
        cache[te0] = pd.read_sql(text("""
            SELECT p.symbol,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close * p.volume) AS dv,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close) AS price,
                   count(*) AS days
            FROM prices_ohlcv p JOIN securities s ON s.symbol = p.symbol
            WHERE p.bar = '1d' AND p.ts >= :a AND p.ts < :b AND s.security_type = 'equity'
            GROUP BY p.symbol"""), engine or _engine(),
            params={"a": start, "b": pd.Timestamp(te0)}).set_index("symbol")
        VAR.mkdir(parents=True, exist_ok=True)
        store.write_bytes(pickle.dumps(cache))
    return cache[te0]


def names(test: str, te0: str, engine=None) -> list[str]:
    """The test's universe for the period starting ``te0`` (point in time)."""
    r = ranked(te0, engine)
    r = r[~r.index.isin(EXCLUDE)]
    spec = TESTS[test]
    if spec[0] == "top":
        pool = r[(r["days"] >= ROLL["min_days"]) & (r["dv"] >= ROLL["min_dollar_volume"])
                 & (r["price"] >= ROLL["min_price"])]
        return sorted(pool.nlargest(ROLL["top_n"], "dv").index)
    _, lo, hi, floor, _ = spec
    pool = r[(r["days"] >= ROLL["min_days"]) & (r["dv"] >= floor) & (r["price"] >= 5)]
    return sorted(pool.sort_values("dv", ascending=False).index[lo:hi])


def _earnings_frames(engine, index: pd.DatetimeIndex, columns) -> dict[str, pd.DataFrame]:
    """yfinance earnings (the sibling repo's ``earnings`` table) as daily frames, each value on
    its USABLE bar: the second session after the report date (decided at that bar's close, so
    a report the source dated one day early still cannot be traded before it is out).
    ``earn_flag`` 1.0, ``earn_surprise`` (actual - estimate) / |estimate|, ``earn_actual``,
    ``earn_estimate``; NaN elsewhere. The report session itself is ``earn_day`` (1.0) — for
    measuring reactions only, never as a signal date."""
    from sqlalchemy import text
    rows = pd.read_sql(text("""SELECT symbol, report_date, eps_estimate, eps_actual, surprise_pct
                               FROM earnings WHERE symbol = ANY(:s)"""), engine,
                       params={"s": list(columns)})
    days = index.tz_convert(None).normalize() if index.tz is not None else index.normalize()
    pos = np.searchsorted(days.to_numpy(), pd.to_datetime(rows["report_date"]).to_numpy())
    out = {k: pd.DataFrame(np.nan, index=index, columns=columns)
           for k in ("earn_flag", "earn_surprise", "earn_actual", "earn_estimate", "earn_day")}
    col = {c: i for i, c in enumerate(columns)}
    for k in out:
        arr = out[k].to_numpy()
        for (_, row), p in zip(rows.iterrows(), pos, strict=True):
            j = col.get(row["symbol"])
            if j is None or p >= len(index):
                continue
            q = p if k == "earn_day" else p + 2
            if q >= len(index):
                continue
            arr[q, j] = {"earn_flag": 1.0, "earn_day": 1.0,
                         "earn_surprise": row["surprise_pct"], "earn_actual": row["eps_actual"],
                         "earn_estimate": row["eps_estimate"]}[k]
        out[k] = pd.DataFrame(arr, index=index, columns=columns)
    return out


def build(workers: int = 1) -> None:
    """Build every test period's panel once: one load of every name any test uses (2015 ->
    LAST) with :data:`FEATURES`, the earnings frames and the sector codes; then each period's
    slice (its names, rows up to its end) is saved to ``panels/<test>_<start>.pkl``."""
    from sqlalchemy import text

    from trading_simple import features
    from trading_simple.data import load_panel
    engine, settings = _engine(), _settings()
    want = {(t, te0): names(t, te0, engine) for t in TESTS for te0, _ in periods(t)}
    union = sorted(set().union(*want.values()))
    print(f"{len(want)} periods, {len(union)} names; loading 2015 -> {LAST}", flush=True)
    parts = []                       # chunks: one 2,200-name query once killed the DB backend
    for i in range(0, len(union) + 1, 300):
        chunk = (union + ["SPY"])[i:i + 300]
        if chunk:
            parts.append(load_panel(engine, chunk, "1d", START, LAST, session="regular",
                                    exchanges=settings.exchanges))
            print(f"  loaded {i + len(chunk)} / {len(union) + 1} names", flush=True)
    frames = {k: pd.concat([getattr(q, k) for q in parts], axis=1).sort_index()
              for k in ("close", "open_", "high", "low", "volume")}
    p = replace(parts[0], **frames)
    sessions = p.close.index[p.close["SPY"].notna()]          # NYSE sessions: SPY trades them all
    dropped = p.close.index.difference(sessions)
    print(f"dropping {len(dropped)} non-NYSE rows: {[str(d.date()) for d in dropped][:20]}",
          flush=True)
    p = replace(p, close=p.close.loc[sessions], open_=p.open_.loc[sessions],
                high=p.high.loc[sessions], low=p.low.loc[sessions],
                volume=p.volume.loc[sessions])
    spy = p.close["SPY"].copy()
    (VAR / "spy.pkl").write_bytes(pickle.dumps(spy))
    p = replace(p, close=p.close.drop(columns="SPY"), open_=p.open_.drop(columns="SPY"),
                high=p.high.drop(columns="SPY"), low=p.low.drop(columns="SPY"),
                volume=p.volume.drop(columns="SPY"))
    features.attach_features(p, engine, FEATURES, settings=settings)
    p.features.update(_earnings_frames(engine, p.close.index, p.close.columns))
    sec = pd.read_sql(text("SELECT symbol, sector FROM securities"), engine).set_index(
        "symbol")["sector"].reindex(p.close.columns)
    codes = {s: i for i, s in enumerate(sorted(sec.dropna().unique()))}
    p.features["sector"] = pd.DataFrame(
        np.tile(sec.map(codes).to_numpy(dtype=float), (len(p.close), 1)),
        index=p.close.index, columns=p.close.columns)
    (VAR / "sectors.json").write_text(json.dumps(codes, indent=1))
    PANELS.mkdir(parents=True, exist_ok=True)
    for (t, te0), syms in want.items():
        te1 = dict(periods(t))[te0]
        cut = p.close.index <= pd.Timestamp(te1, tz="UTC")
        sl = lambda f, cut=cut, syms=syms: f.loc[cut, syms]  # noqa: E731
        q = replace(p, close=sl(p.close), open_=sl(p.open_), high=sl(p.high), low=sl(p.low),
                    volume=sl(p.volume), features={k: sl(v) for k, v in p.features.items()})
        _save_split(q, PANELS / f"{t}_{te0}")
        print(f"  {t} {te0}: {len(syms)} names", flush=True)


PRICE_FRAMES = ("close", "open_", "high", "low", "volume")


def _save_split(q, folder: Path) -> None:
    """One pickle per frame, so a strategy loads only the features it reads."""
    (folder / "features").mkdir(parents=True, exist_ok=True)
    for k in PRICE_FRAMES:
        (folder / f"{k}.pkl").write_bytes(pickle.dumps(getattr(q, k)))
    for k, v in (q.features or {}).items():
        (folder / "features" / f"{k}.pkl").write_bytes(pickle.dumps(v))
    (folder / "meta.json").write_text(json.dumps({"bar": q.bar, "tz": q.tz}))


def split_existing() -> None:
    """Convert whole-panel pickles (``panels/<test>_<start>.pkl``) to the per-frame layout."""
    for f in sorted(PANELS.glob("*.pkl")):
        q = pickle.loads(f.read_bytes())
        _save_split(q, PANELS / f.stem)
        f.unlink()
        print("split", f.stem, flush=True)


def available_features(test: str, te0: str) -> list[str]:
    return sorted(x.stem for x in (PANELS / f"{test}_{te0}" / "features").glob("*.pkl"))


_PANELS: dict = {}
_WARNED: set = set()


def panel(test: str, te0: str, features=None):
    """The saved panel of one test period (names point in time, rows 2015 -> period end) with
    the price frames and ``features`` (a list; ``None`` = every saved feature). ``sector`` is
    always included."""
    from trading_simple.data import Panel
    folder = PANELS / f"{test}_{te0}"
    feats = available_features(test, te0) if features is None else \
        sorted(set(features) | {"sector"})
    key = (test, te0, tuple(feats))
    if key not in _PANELS:
        if len(_PANELS) >= 16:                  # every TRAIN period fits (the optimizer)
            _PANELS.pop(next(iter(_PANELS)))
        frames = {k: pickle.loads((folder / f"{k}.pkl").read_bytes()) for k in PRICE_FRAMES}
        meta = json.loads((folder / "meta.json").read_text())
        have = set(available_features(test, te0))
        missing = [f for f in feats if f not in have]
        if missing and tuple(missing) not in _WARNED:
            _WARNED.add(tuple(missing))
            print(f"lab: feature(s) {missing} not in the lab panels — the strategy runs "
                  f"without them (saved: {sorted(have)})", flush=True)
        fs = {k: pickle.loads((folder / "features" / f"{k}.pkl").read_bytes())
              for k in feats if k in have}
        _PANELS[key] = Panel(close=frames["close"], open_=frames["open_"], high=frames["high"],
                             low=frames["low"], volume=frames["volume"], bar=meta["bar"],
                             tz=meta["tz"], features=fs)
    return _PANELS[key]


def strategy_features(strat) -> list[str]:
    """The features a strategy reads in the lab: ``data.lab_features`` if declared (for
    frames no ``features.py`` builder serves, e.g. ``earn_*``), else ``data.features``."""
    d = strat.data or {}
    return list(d.get("lab_features") or d.get("features") or [])


def spy() -> pd.Series:
    """SPY's daily adjusted close (2015 -> LAST)."""
    return pickle.loads((VAR / "spy.pkl").read_bytes())


def with_spy(p):
    """A panel with SPY's bars added as a column (for strategies that trade it as a hedge;
    SPY has no features, so it is never ranked)."""
    if "SPY" in p.close.columns:
        return p
    s = spy().reindex(p.close.index)
    add = {k: getattr(p, k).assign(SPY=s) for k in ("close", "open_", "high", "low")}
    vol = p.volume.assign(SPY=np.nan)
    return replace(p, volume=vol, **add)


# --- statistics (copied from informed_flow_ls/research.py so this file stands alone) ---------

def years_of(index) -> float:
    from trading_simple import backtest as bt
    return bt._years(index)


def alpha_of(r, bench, years):
    """Geometric CAGR, buy-and-hold CAGR, beta and alpha (= CAGR − beta × B&H CAGR) of aligned
    daily return rows (1-d or 2-d: one row per resample)."""
    r, bench = np.atleast_2d(r), np.atleast_2d(bench)
    cagr = np.expm1(np.log1p(r).sum(axis=1) / years)
    bench_cagr = np.expm1(np.log1p(bench).sum(axis=1) / years)
    rc, bc = r - r.mean(axis=1, keepdims=True), bench - bench.mean(axis=1, keepdims=True)
    var = (bc * bc).sum(axis=1)
    beta = np.divide((rc * bc).sum(axis=1), var, out=np.zeros_like(var), where=var > 0)
    return cagr, bench_cagr, beta, cagr - beta * bench_cagr


def block_index(size: int, n: int = N_BOOT, block: int = BOOT_BLOCK, seed: int = 7):
    """Stationary block-bootstrap resamples (Politis-Romano) of positions 0..size-1."""
    rng = np.random.default_rng(seed)
    pos = np.arange(size)
    new = rng.random((n, size)) < 1.0 / block
    new[:, 0] = True
    start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    origin = np.take_along_axis(np.where(new, rng.integers(0, size, (n, size)), 0), start,
                                axis=1)
    return (origin + pos - start) % size


def bootstrap(r, bench, years, seed: int = 7) -> dict:
    r, bench = np.asarray(r, float), np.asarray(bench, float)
    idx = block_index(len(r), seed=seed)
    a = alpha_of(r[idx], bench[idx], years)[3]
    return {"P(alpha > 0)": float((a > 0).mean()), "alpha p5": float(np.percentile(a, 5)),
            "alpha p95": float(np.percentile(a, 95))}


def paired(r, base, bench, years, seed: int = 11) -> dict:
    """Alpha of ``r`` minus alpha of ``base`` on the same days and bootstrap blocks."""
    r, base, bench = (np.asarray(x, float) for x in (r, base, bench))
    idx = block_index(len(r), seed=seed)
    diff = alpha_of(r[idx], bench[idx], years)[3] - alpha_of(base[idx], bench[idx], years)[3]
    point = alpha_of(r, bench, years)[3][0] - alpha_of(base, bench, years)[3][0]
    return {"delta alpha": float(point), "P(delta > 0)": float((diff > 0).mean())}


def stitched(pairs) -> dict:
    """One continuous record from per-period ``(book, buy-and-hold)`` daily returns."""
    r = pd.concat([x for x, _ in pairs])
    b = pd.concat([y for _, y in pairs]).reindex(r.index).fillna(0.0)
    years = years_of(r.index)
    cagr, bench_cagr, beta, alpha = (float(x[0]) for x in alpha_of(r.to_numpy(), b.to_numpy(),
                                                                   years))
    curve = (1 + r).cumprod()
    return {"years": years, "cagr": cagr, "buy & hold": bench_cagr, "beta": beta, "alpha": alpha,
            "volatility": float(r.std() * np.sqrt(len(r) / years)),
            "max drawdown": float((curve / curve.cummax() - 1).min()),
            **bootstrap(r.to_numpy(), b.to_numpy(), years), "returns": r, "bench": b}


def period_alpha(r: pd.Series, b: pd.Series) -> float:
    return float(alpha_of(r.to_numpy(), b.to_numpy(), years_of(r.index))[3][0])


def robust(alphas) -> float:
    """Mean of per-period alphas minus half their spread: steady beats one lucky stretch."""
    a = np.asarray(alphas, float)
    return float(a.mean() - 0.5 * a.std())


def pbo(matrix: np.ndarray) -> float:
    """Probability of backtest overfitting (Bailey et al. 2017) on a trials x periods matrix of
    per-period scores: over every half/half split of the periods, how often does the
    in-sample best trial land in the out-of-sample bottom half?"""
    m = np.asarray(matrix, float)
    if m.shape[0] < 3 or m.shape[1] < 4:
        return float("nan")
    k = m.shape[1] - (m.shape[1] % 2)
    m = m[:, -k:]
    below = []
    for ins in itertools.combinations(range(k), k // 2):
        out = [i for i in range(k) if i not in ins]
        best = int(np.nanargmax(m[:, list(ins)].mean(axis=1)))
        oos = pd.Series(m[:, out].mean(axis=1)).rank(pct=True).to_numpy()
        below.append(oos[best] <= 0.5)
    return float(np.mean(below))


# --- one book ---------------------------------------------------------------------------------

def load_strategy(name: str, overrides: dict | None = None):
    """A loaded strategy with ``overrides`` merged into its params (validated)."""
    from trading_simple import strategies
    s = strategies.load(name)
    s = replace(s, params={**s.params, **(overrides or {})})
    if not s.valid(s.params):
        raise ValueError(f"{name}: overrides {overrides} fail valid()")
    return s


def with_costs(settings, multiple: float = 1.0, late_bps: float = 0.0):
    """Settings with every cost scaled by ``multiple`` and ``late_bps`` added a leg."""
    c = settings.costs
    costs = c.model_copy(update={"commission_per_share": c.commission_per_share * multiple,
                                 "min_commission": c.min_commission * multiple,
                                 "spread_bps": c.spread_bps * multiple,
                                 "slippage_bps": c.slippage_bps * multiple + late_bps})
    return settings.model_copy(update={"costs": costs})


def _call_order(targets: pd.DataFrame) -> np.ndarray:
    """Per-bar execution order (a vectorbt ``call_seq`` array) for :func:`run_portfolio`.

    Each order is classed against the name's IN-FORCE target before the bar (its last emitted
    target): 0 a long cut or exit, 1 a short cover or cut, 2 a new or larger short (a long
    flipped short included), 3 a long bought or added (a short flipped long included); names
    with no order last. Sales and covers free the cash that new shorts need as collateral and
    longs spend. The engine's ``call_seq="auto"`` runs a new short (a sale) BEFORE the cover
    (a buy) that would release locked cash, so on a full book vectorbt rejects or part-fills it
    and, since targets are emitted only on change, never re-sends it.

    Returns:
        ``call_seq[row, k]`` = the column called k-th in ``row`` (vectorbt's convention: "the
        position of the column in the group to call next").
    """
    t = targets.to_numpy(dtype=float)
    prev = pd.DataFrame(t).ffill().shift(1).fillna(0.0).to_numpy()
    has = ~np.isnan(t)
    x = np.where(has, t, prev)
    cat = np.full(t.shape, 4, dtype=np.int64)
    cat[has & (x > 0) & (x > prev)] = 3
    cat[has & (x < 0) & (x < prev)] = 2
    cat[has & (prev < 0) & (x >= prev) & (x <= 0)] = 1
    cat[has & (prev > 0) & (x <= prev) & (x >= 0)] = 0
    return np.argsort(cat, axis=1, kind="stable")


def borrow_fees(targets: pd.DataFrame, borrow, fees: float) -> pd.DataFrame:
    """``backtest._borrow_fees`` with one fix: a short re-sized by less than a quarter of its
    size while open does not trigger the charge (the accrual carries on to its next
    substantial buy-back or its end). The engine's version turns a hair-sized re-trim into a
    fee FRACTION ``accrued × |w| / |x − w|`` that vectorbt applies to an order whose real size
    comes from drift (etf_trend's review: borrow over-charged 4-5×)."""
    index = targets.index
    if isinstance(borrow, pd.DataFrame):
        rate = borrow.reindex(index=index, columns=targets.columns).fillna(0.0).to_numpy()
    else:
        rate = np.full(targets.shape, float(borrow))
    years = np.r_[0.0, np.diff(index.asi8) / 86_400e9] / 365.0
    accrued = np.cumsum(rate * years[:, None], axis=0)
    t = targets.to_numpy(dtype=float)
    out = np.zeros_like(t)
    for j in range(t.shape[1]):
        orders = np.flatnonzero(~np.isnan(t[:, j]))
        w, since, opened = 0.0, 0, None
        for k in orders:
            x = t[k, j]
            small = x < 0 and (x - w) < 0.25 * (-w)            # a hair re-trim of the short
            if w < 0 and x > w and not small:
                out[k, j] += (accrued[k, j] - accrued[since, j]) * (-w) / (x - w)
                since = k
            if x < 0 and w >= 0:
                opened, since = (k, w), k
            w = x
        if w < 0 and opened is not None:
            k, before = opened
            out[k, j] += (accrued[-1, j] - accrued[since, j]) * (-w) / (before - t[k, j])
    return pd.DataFrame(fees + out, index=index, columns=targets.columns)


def run_portfolio(panel, weights: pd.DataFrame, *, price, fees, slippage, init_cash: float,
                  retrim_k=None, borrow=None, size_granularity=None):
    """The engine's ``backtest._run_portfolio`` with the lab's two fixes: :func:`_call_order`
    (covers before new shorts) and :func:`borrow_fees`. Everything else — event-driven
    targets, lock_cash, cash sharing, the retrim loop — is the engine's own code."""
    import vectorbt as vbt

    from trading_simple import backtest as bt
    whole = {} if size_granularity is None else {"size_granularity": size_granularity}
    targets = bt.rotation_targets(weights)
    if borrow is not None:
        fees = borrow_fees(targets, borrow, fees)

    def build(tg):
        return vbt.Portfolio.from_orders(
            panel.close, size=tg, size_type="targetpercent", price=price, fees=fees,
            slippage=slippage, init_cash=init_cash, **whole, group_by=True, cash_sharing=True,
            lock_cash=True, call_seq=_call_order(tg), freq=panel.freq)

    if retrim_k is None:
        return build(targets)
    return bt._retrim_loop(build, targets, weights, retrim_k * weights)


def engine_run(strat, w, sig: dict, settings):
    """``backtest.run`` on a scored window: portfolio mode through :func:`run_portfolio`,
    signal mode (entries/exits) through the engine unchanged. Returns ``(pf, fees, slip)``."""
    from trading_simple import backtest as bt
    bt.configure()
    fees, slip = bt.strategy_costs(strat, w, settings)
    init = settings.backtest.init_cash
    if bt.WEIGHTS_KEY in sig:
        price, slippage = bt.fill_prices(w, slip, strat.fill)
        whole = {"size_granularity": 1.0} if strat.whole_shares else {}
        pf = run_portfolio(w, sig[bt.WEIGHTS_KEY], price=price, fees=fees, slippage=slippage,
                           init_cash=init, retrim_k=sig.get(bt.RETRIM_K_KEY),
                           borrow=sig.get(bt.BORROW_KEY), **whole)
    else:
        pf = bt.run(w, sig, fees=fees, slippage=slip, init_cash=init, fill=strat.fill,
                    whole_shares=strat.whole_shares)
    return pf, fees, slip


def book(strat, p, settings, te0, signals: dict | None = None) -> dict:
    """One warm backtest of a period: signals on the whole panel, scored from ``te0``.
    ``signals`` given: run those (a luck book) instead of the strategy's own.

    Day 0 counts from the initial cash, for the book and the benchmark alike (lab v2): it
    holds the opening orders' costs and the borrow the engine charges up front for shorts held
    to the window's end. A continuous book would not re-buy itself every half-year, so this
    over-charges by one book rebuild per period (~0.1-0.15 pt of alpha a year at the base
    costs, ~0.5 under the late-fill stress) — conservative, never flattering."""
    from trading_simple import backtest as bt
    if signals is None:
        signals = strat.signals(p, **strat.params)
    w, sig = bt._score_window(p, signals, te0)
    pf, fees, slip = engine_run(strat, w, sig, settings)
    total, bench = bt.benchmark(w, fees=fees, slippage=slip, fill=strat.fill)
    value = bt.equity_curve(pf)["value"]
    r = value / value.shift(1).fillna(settings.backtest.init_cash) - 1.0
    bench = bench.reindex(r.index).fillna(0.0).copy()
    day0 = (1.0 + total) / float(np.prod(1.0 + bench.to_numpy())) - 1.0
    bench.iloc[0] = (1.0 + bench.iloc[0]) * (1.0 + day0) - 1.0
    return {"returns": r, "bench": bench, "pf": pf, "window": w,
            "stats": bt.portfolio_stats(pf)}


def _anatomy(res: dict) -> dict:
    """Orders, exposures and per-name P&L contributions of one book (top 20 kept daily)."""
    pf, w = res["pf"], res["window"]
    value = pf.value()
    assets = pf.asset_value(group_by=False)
    held = assets.div(value, axis=0).shift(1).fillna(0.0)
    rets = w.close.pct_change(fill_method=None).reindex(held.index).fillna(0.0)
    contrib = (held * rets[held.columns]) if set(held.columns) <= set(rets.columns) else held * 0
    total = contrib.sum()
    top = total.abs().nlargest(20).index
    return {"orders": int(pf.orders.count().sum()),
            "long": float((assets.clip(lower=0).sum(axis=1) / value).mean()),
            "short": float((assets.clip(upper=0).sum(axis=1) / value).mean()),
            "contrib total": total, "contrib daily": contrib[top],
            "trades_per_year": float(res["stats"].get("trades_per_year", np.nan))}


def _ex_ante_beta(sig: dict, p, te0, te1) -> tuple[float, float]:
    """Mean and largest |Σ target weight × trailing 126-bar beta| over the window."""
    w = sig.get("weights")
    if w is None:
        return float("nan"), float("nan")
    w = w.fillna(0.0)
    ret = p.close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    beta = ret.rolling(126, min_periods=63).cov(mkt).div(
        mkt.rolling(126, min_periods=63).var(), axis=0).shift(1)
    idx = p.close.index
    win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
    net = (w * beta.reindex(columns=w.columns)).sum(axis=1)[win]
    return float(net.mean()), float(net.abs().max())


def results_path(strategy: str, label: str) -> Path:
    return RESULTS / strategy / f"{label}.pkl"


def run(strategy: str, label: str, overrides: dict | None = None, tests=None, *,
        late: bool = True, costs2: bool = False, hedge: bool = False, drop: int = 0,
        seed: int = 0) -> dict:
    """Run a config through ``tests`` (default OPEN), period by period, and save
    ``results/<strategy>/<label>.pkl`` (added to a file saved with the SAME overrides).

    ``late`` adds +50 bp a leg books, ``costs2`` 2× cost books, ``hedge`` merges SPY into
    every panel, ``drop`` removes that many random names per period (the noise check).
    """
    tests = tests or OPEN
    strat = load_strategy(strategy, overrides)
    settings = _settings()
    late_s, c2_s = with_costs(settings, 1.0, 50.0), with_costs(settings, 2.0)
    path = results_path(strategy, label)
    old = pickle.loads(path.read_bytes()) if path.exists() else {}
    meta = {"overrides": overrides or {}, "hedge": hedge, "drop": drop, "seed": seed,
            "lab": LAB_VERSION}
    if old and old.get("meta") != meta:
        raise SystemExit(f"{strategy}/{label} was saved with {old.get('meta')}, not {meta}")
    out = {**old, "meta": meta}
    rng = np.random.default_rng(seed)
    for t in tests:
        rows = []
        for te0, te1 in periods(t):
            p = panel(t, te0, strategy_features(strat))
            if drop:
                keep = sorted(set(p.close.columns) - set(rng.choice(p.close.columns, drop,
                                                                     replace=False)))
                p = replace(p, close=p.close[keep], open_=p.open_[keep], high=p.high[keep],
                            low=p.low[keep], volume=p.volume[keep],
                            features={k: v[keep] for k, v in p.features.items()})
            if hedge:
                p = with_spy(p)
            sig = strat.signals(p, **strat.params)          # once: every book below reuses it
            res = book(strat, p, settings, te0, signals=sig)
            row = {"period": te0, "end": te1, "returns": res["returns"], "bench": res["bench"],
                   **_anatomy(res)}
            row["beta ex-ante"], row["beta ex-ante max"] = _ex_ante_beta(sig, p, te0, te1)
            if late:
                row["late"] = book(strat, p, late_s, te0, signals=sig)["returns"]
            if costs2:
                row["2x costs"] = book(strat, p, c2_s, te0, signals=sig)["returns"]
            rows.append(row)
            print(f"{strategy}/{label} {t} {te0}: alpha "
                  f"{period_alpha(row['returns'], row['bench']):+.3f}  orders {row['orders']}",
                  flush=True)
        out[t] = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def load_results(strategy: str, label: str) -> dict:
    return pickle.loads(results_path(strategy, label).read_bytes())


# --- reading results ---------------------------------------------------------------------------

def _stitch(rows, key="returns"):
    return stitched([(r[key], r["bench"]) for r in rows])


def top_out(rows, k: int = 5) -> float:
    """Stitched alpha after removing the daily P&L of the test's ``k`` biggest contributors."""
    tot = pd.concat([r["contrib total"] for r in rows], axis=1).sum(axis=1)
    worst = tot.sort_values(ascending=False).index[:k]
    pairs = []
    for r in rows:
        d = r["contrib daily"]
        cut = d[[c for c in worst if c in d.columns]].sum(axis=1).reindex(r["returns"].index)
        pairs.append((r["returns"] - cut.fillna(0.0), r["bench"]))
    return stitched(pairs)["alpha"]


def monthly(rows):
    r = pd.concat([x["returns"] for x in rows])
    b = pd.concat([x["bench"] for x in rows]).reindex(r.index).fillna(0.0)
    key = r.index.tz_convert(None).to_period("M")
    return (1 + r).groupby(key).prod() - 1, (1 + b).groupby(key).prod() - 1


def down_months(res, tests=("E0", "TB", "T3", "T1")):
    """Book and market mean over the market's down months, the tests stitched."""
    books, mkts = [], []
    for t in tests:
        if t in res:
            bk, mk = monthly(res[t])
            books.append(bk[mk < 0])
            mkts.append(mk[mk < 0])
    if not books:
        return {}
    bk, mk = pd.concat(books), pd.concat(mkts)
    return {"down months": int(len(bk)), "book mean": float(bk.mean()),
            "market mean": float(mk.mean()), "book > 0": float((bk > 0).mean())}


def episodes(res) -> dict:
    """The book's and the market's return over each falling-market episode the tests cover."""
    r = pd.concat([x["returns"] for t in ("E0", "TB", "T3", "T1") if t in res for x in res[t]])
    b = pd.concat([x["bench"] for t in ("E0", "TB", "T3", "T1") if t in res for x in res[t]])
    r, b = r[~r.index.duplicated()].sort_index(), b[~b.index.duplicated()].sort_index()
    out = {}
    for k, (a, z) in EPISODES.items():
        m = (r.index >= pd.Timestamp(a, tz="UTC")) & (r.index <= pd.Timestamp(z, tz="UTC"))
        if m.sum() < 5:
            continue
        out[k] = {"book": float((1 + r[m]).prod() - 1), "market": float((1 + b[m]).prod() - 1)}
    return out


def summary(res, tests=None) -> dict:
    """Per test: alpha, P, beta, CAGR, B&H, vol, max DD, orders/yr, late / 2x-cost alpha,
    top-5-out alpha, ex-ante beta, per-period alphas."""
    out = {}
    for t in tests or [t for t in TESTS if t in res]:
        if t not in res:
            continue
        rows = res[t]
        s = _stitch(rows)
        col = {"alpha": s["alpha"], "P": s["P(alpha > 0)"], "beta": s["beta"], "cagr": s["cagr"],
               "B&H": s["buy & hold"], "vol": s["volatility"], "maxDD": s["max drawdown"],
               "orders/yr": sum(r["orders"] for r in rows) / s["years"],
               "top5-out alpha": top_out(rows),
               "ex-ante beta": float(np.nanmean([r["beta ex-ante"] for r in rows])),
               "periods": [round(period_alpha(r["returns"], r["bench"]), 3) for r in rows]}
        for key in ("late", "2x costs"):
            if key in rows[0]:
                col[f"{key} alpha"] = _stitch(rows, key)["alpha"]
        out[t] = col
    return out


def report(strategy: str, base: str, labels: list[str], tests=None) -> pd.DataFrame:
    """Every label against ``base`` on the same days: per-test alpha and Δ (paired bootstrap),
    periods better, plus the falling-month mean and the episodes."""
    ref = load_results(strategy, base)
    table = {}
    for name in [base, *labels]:
        res = load_results(strategy, name)
        col = {}
        for t in tests or [t for t in TESTS if t in res]:
            if t not in res:
                continue
            rows = res[t]
            s = _stitch(rows)
            col[f"{t} alpha"] = s["alpha"]
            col[f"{t} P"] = s["P(alpha > 0)"]
            col[f"{t} beta"] = s["beta"]
            col[f"{t} maxDD"] = s["max drawdown"]
            if "late" in rows[0]:
                col[f"{t} late"] = _stitch(rows, "late")["alpha"]
            col[f"{t} orders/yr"] = sum(r["orders"] for r in rows) / s["years"]
            if name != base and t in ref:
                b = _stitch(ref[t])
                pr = paired(s["returns"], b["returns"].reindex(s["returns"].index).fillna(0.0),
                            b["bench"].reindex(s["returns"].index).fillna(0.0), b["years"])
                col[f"{t} Δ"] = pr["delta alpha"]
                col[f"{t} P(Δ>0)"] = pr["P(delta > 0)"]
                d = [period_alpha(x["returns"], x["bench"]) - period_alpha(y["returns"], y["bench"])
                     for x, y in zip(rows, ref[t], strict=False)]
                col[f"{t} better"] = f"{sum(v > 0 for v in d)}/{len(d)}"
        dm = down_months(res)
        if dm:
            col["down months: book / market"] = (
                f"{dm['book mean']:+.2%} / {dm['market mean']:+.2%}"
                f" (n {dm['down months']}, >0 {dm['book > 0']:.0%})")
        for k, v in episodes(res).items():
            col[k] = f"{v['book']:+.1%} ({v['market']:+.1%})"
        table[name] = col
    out = pd.DataFrame(table)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 200):
        print(out.to_string(float_format=lambda x: f"{x:.3f}"))
    return out


# --- luck, noise, the optimizer ---------------------------------------------------------------

def _tenths(p, te0) -> pd.Series:
    before = p.close.index < pd.Timestamp(te0, tz="UTC")
    dollars = (p.close * p.volume)[before].tail(126).median()
    return np.ceil(dollars.rank(pct=True) * 10).fillna(0)


def luck(strategy: str, label: str, test: str = "T1", n: int = 100, seed: int = 29,
         fixed=("SPY",)) -> dict:
    """``n`` books per period that trade the strategy's OWN signals but on randomly relabelled
    names: every per-name frame (weights / entries / exits / cash_frac) is column-permuted
    within dollar-volume tenths (ranked on the 126 sessions before the period), so timing,
    turnover, holding times and liquidity match and only the names are random. Never per-bar
    random ranks (informed_flow v25: they churn daily and cannot fail). Returns the real
    alpha's percentile among the books and the band."""
    res = load_results(strategy, label)
    strat = load_strategy(strategy, res["meta"]["overrides"])
    settings = _settings()
    key = hashlib.sha1(repr((sorted(strat.params.items()), test, n, seed)).encode()
                       ).hexdigest()[:10]
    cache = RESULTS / strategy / f"luck-{label}-{key}.pkl"
    if cache.exists():
        books = pickle.loads(cache.read_bytes())
    else:
        rng = np.random.default_rng(seed)
        books = []
        for te0, _ in periods(test):
            p = panel(test, te0, strategy_features(strat))
            if res["meta"].get("hedge"):
                p = with_spy(p)
            sig = strat.signals(p, **strat.params)
            tenth = _tenths(p, te0)
            cols = p.close.columns.to_numpy()
            movable = np.array([c not in fixed for c in cols])
            per = []
            for _ in range(n):
                perm = cols.copy()
                for g in np.unique(tenth.to_numpy()):
                    inside = np.flatnonzero((tenth.to_numpy() == g) & movable)
                    perm[inside] = cols[rng.permutation(inside)]
                mapping = dict(zip(cols, perm, strict=True))
                shuf = {}
                for k, v in sig.items():
                    if (isinstance(v, pd.DataFrame) and k != "borrow"
                            and set(v.columns) <= set(cols)):
                        shuf[k] = v.rename(columns=mapping)[v.columns]
                    else:
                        shuf[k] = v
                per.append(book(strat, p, settings, te0, signals=shuf)["returns"])
            books.append(per)
            print(f"luck {strategy}/{label} {test} {te0}: {n} books", flush=True)
        cache.write_bytes(pickle.dumps(books))
    bench = [r["bench"] for r in res[test]]
    alphas = np.array([stitched([(books[i][k], bench[i]) for i in range(len(books))])["alpha"]
                       for k in range(n)])
    real = _stitch(res[test])["alpha"]
    out = {"real": real, "median": float(np.median(alphas)),
           "p95": float(np.percentile(alphas, 95)),
           "percentile": float((alphas < real).mean() * 100)}
    print(out)
    return out


def noise(strategy: str, label: str, k: int = 5, copies: int = 3, tests=None) -> dict:
    """Path noise: the same config re-run ``copies`` times on panels missing ``k`` random names
    a period. Returns per test the spread (sd) of the copies' alpha minus the label's — a Δ
    between two variants smaller than ~2 of these is noise."""
    res = load_results(strategy, label)
    over = res["meta"]["overrides"]
    tests = tests or [t for t in OPEN if t in res]
    deltas = {t: [] for t in tests}
    for c in range(copies):
        name = f"{label}-noise{k}-{c}"
        path = results_path(strategy, name)
        r = pickle.loads(path.read_bytes()) if path.exists() else run(
            strategy, name, over, tests, late=False, hedge=res["meta"].get("hedge", False),
            drop=k, seed=100 + c)
        for t in tests:
            deltas[t].append(_stitch(r[t])["alpha"] - _stitch(res[t])["alpha"])
    out = {t: {"deltas": [round(x, 4) for x in v], "sd": float(np.std(v, ddof=1))}
           for t, v in deltas.items()}
    print(json.dumps(out, indent=1))
    return out


def tune(strategy: str, label: str, overrides: dict, space: dict, n_trials: int = 120,
         tests=None, seed: int = 42, min_trades: float = 0.0) -> dict:
    """TPE over ``space`` ({knob: {type, low, high, step} | {type: categorical, choices}}) on the
    ``tests`` periods (default TRAIN), every trial scored by :func:`robust` over its
    per-period alphas. Journal in ``studies/<strategy>-<label>.log`` (resumable); the trials'
    per-period alphas give the PBO. Returns the best params, its score and the PBO."""
    import optuna
    tests = tests or TRAIN
    settings = _settings()
    STUDIES.mkdir(parents=True, exist_ok=True)
    name = f"{strategy}-{label}"
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        str(STUDIES / f"{name}.log")))
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(
                                    seed=seed, n_startup_trials=min(20, n_trials // 3)),
                                load_if_exists=True)
    stamp = {"space": repr(space), "tests": tests, "overrides": repr(overrides)}
    if study.user_attrs and study.user_attrs.get("stamp") != stamp:
        raise ValueError(f"study {name} was run with another space/tests/overrides")
    study.set_user_attr("stamp", stamp)
    store = STUDIES / f"{name}.alphas.pkl"
    matrix = pickle.loads(store.read_bytes()) if store.exists() else {}

    def suggest(trial):
        out = {}
        for k, s in space.items():
            if s["type"] == "int":
                out[k] = trial.suggest_int(k, s["low"], s["high"], step=s.get("step", 1))
            elif s["type"] == "float":
                out[k] = trial.suggest_float(k, s["low"], s["high"], step=s.get("step"))
            else:
                out[k] = trial.suggest_categorical(k, s["choices"])
        return out

    def objective(trial):
        params = {**overrides, **suggest(trial)}
        try:
            strat = load_strategy(strategy, params)
        except ValueError:
            raise optuna.TrialPruned() from None
        alphas, trades = [], []
        for t in tests:
            for te0, _ in periods(t):
                res = book(strat, panel(t, te0, strategy_features(strat)), settings, te0)
                alphas.append(period_alpha(res["returns"], res["bench"]))
                trades.append(res["stats"].get("trades_per_year", np.nan))
        if min_trades and np.nanmean(trades) < min_trades:
            raise optuna.TrialPruned()
        matrix[trial.number] = alphas
        store.write_bytes(pickle.dumps(matrix))
        trial.set_user_attr("mean alpha", float(np.mean(alphas)))
        return robust(alphas)

    left = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    m = np.array([matrix[t.number] for t in done if t.number in matrix])
    best = study.best_trial
    out = {"best": best.params, "score": best.value,
           "mean alpha": best.user_attrs.get("mean alpha"), "pbo": pbo(m),
           "trials": len(done)}
    print(json.dumps(out, indent=1, default=str))
    return out


# --- the gates --------------------------------------------------------------------------------

def verdict(strategy: str, label: str, market_neutral: bool = True) -> pd.DataFrame:
    """The README's gates G1-G8 for one saved label (luck read from its cache if run)."""
    res = load_results(strategy, label)
    s = summary(res)
    train = stitched([(r["returns"], r["bench"]) for t in TRAIN if t in res for r in res[t]]) \
        if any(t in res for t in TRAIN) else None
    lk = None
    if any((RESULTS / strategy).glob(f"luck-{label}-*.pkl")):
        lk = luck(strategy, label)
    ep = episodes(res)
    dm = down_months(res)
    t1 = s.get("T1", {})
    rows = [
        ("G1 TRAIN-era alpha > 0", train["alpha"] if train else np.nan,
         bool(train and train["alpha"] > 0)),
        ("G2 T1 alpha > 0, P >= 0.90", t1.get("alpha"), bool(t1 and t1["alpha"] > 0
                                                           and t1["P"] >= 0.90)),
        ("G3 T2 alpha > 0", s.get("T2", {}).get("alpha"),
         bool(s.get("T2", {}).get("alpha", -1) > 0)),
        ("G4 T1 beats >= 95% of shuffled books", lk["percentile"] if lk else np.nan,
         bool(lk and lk["percentile"] >= 95)),
        ("G5 T1 late-fill and 2x-cost alpha > 0",
         min(t1.get("late alpha", np.nan), t1.get("2x costs alpha", np.inf)),
         bool(t1.get("late alpha", -1) > 0 and t1.get("2x costs alpha", 1) > 0)),
        ("G6 beats the market in every bear episode, down months > 0",
         dm.get("book mean", np.nan),
         bool(ep and all(v["book"] > v["market"] for v in ep.values())
              and dm.get("book mean", -1) > 0)),
        ("G7 |T1 beta| <= 0.3" if market_neutral else "G7 (not market-neutral: skipped)",
         t1.get("beta"), bool((not market_neutral) or abs(t1.get("beta", 9)) <= 0.3)),
        ("G8 T1 alpha without its top 5 names > 0", t1.get("top5-out alpha"),
         bool(t1.get("top5-out alpha", -1) > 0)),
    ]
    out = pd.DataFrame(rows, columns=["gate", "value", "passed"])
    print(out.to_string())
    return out


def bear(strategy: str, labels: list[str]) -> None:
    """Episodes, calendar years, CAGR and max drawdown over the stitched E0 + TB + T3 + T1."""
    rows, yrs = {}, {}
    for lab in labels:
        res = load_results(strategy, lab)
        r = pd.concat([x["returns"] for t in ("E0", "TB", "T3", "T1") if t in res for x in res[t]])
        b = pd.concat([x["bench"] for t in ("E0", "TB", "T3", "T1") if t in res for x in res[t]])
        r, b = r[~r.index.duplicated()].sort_index(), b[~b.index.duplicated()].sort_index()
        ep = episodes(res)
        rows[lab] = {k: v["book"] for k, v in ep.items()}
        if "market" not in rows:
            rows["market (equal weight)"] = {k: v["market"] for k, v in ep.items()}
            sp = spy().pct_change()
            inside = lambda a, z, sp=sp: (sp.index >= pd.Timestamp(a, tz="UTC")) & (  # noqa: E731
                sp.index <= pd.Timestamp(z, tz="UTC"))
            rows["SPY"] = {k: float((1 + sp[inside(a, z)]).prod() - 1)
                           for k, (a, z) in EPISODES.items() if k in ep}
            yrs["market (equal weight)"] = (1 + b).groupby(b.index.year).prod() - 1
        curve = (1 + r).cumprod()
        rows[lab]["CAGR"] = float(curve.iloc[-1] ** (252 / len(r)) - 1)
        rows[lab]["max DD"] = float((curve / curve.cummax() - 1).min())
        yrs[lab] = (1 + r).groupby(r.index.year).prod() - 1
    with pd.option_context("display.width", 250):
        print((pd.DataFrame(rows).T * 100).round(1).to_string())
        print((pd.DataFrame(yrs).T * 100).round(1).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if cmd == "build":
        build()
    elif cmd == "split":
        split_existing()
    elif cmd == "run":
        run(a[1], a[2], json.loads(a[3]) if len(a) > 3 else {},
            a[4].split(",") if len(a) > 4 else None,
            costs2="--costs2" in a, hedge="--hedge" in a)
    elif cmd == "report":
        tests = None
        if "--tests" in a:
            i = a.index("--tests")
            tests, a = a[i + 1].split(","), a[:i] + a[i + 2:]
        report(a[1], a[2], a[3:], tests)
    elif cmd == "luck":
        luck(a[1], a[2], a[3] if len(a) > 3 else "T1", int(a[4]) if len(a) > 4 else 100)
    elif cmd == "noise":
        noise(a[1], a[2], int(a[3]) if len(a) > 3 else 5, int(a[4]) if len(a) > 4 else 3)
    elif cmd == "tune":
        tune(a[1], a[2], json.loads(a[3]), json.loads(a[4]), int(a[5]) if len(a) > 5 else 120,
             a[6].split(",") if len(a) > 6 else None)
    elif cmd == "bear":
        bear(a[1], a[2:])
    elif cmd == "verdict":
        verdict(a[1], a[2], market_neutral="--not-neutral" not in a)
    else:
        raise SystemExit(__doc__)
