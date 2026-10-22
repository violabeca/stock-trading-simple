"""darts_forecast explorer helpers (``darts_forecast_explorer.ipynb``): which bar size
(granularity), input length and prediction length let a Darts model predict the next PRICE PATH
best?

The notebook is a workbench, not a strategy run. It

1. picks a small, diverse set of stocks (large / mid / small market cap x slow / fast moving), each
   with complete 1-minute data (:func:`candidates`, :func:`pick`);
2. loads six months of them and splits the months into TRAIN, VALIDATION and TEST
   (:func:`load_window`);
3. builds bars of every granularity from the 1-minute bars (:func:`path_inputs`) and lets one model
   predict the whole path of the next ``H`` bars from the last ``L`` bars (:func:`fit_paths`,
   :func:`predict_paths`), at the same decision times for every granularity
   (:data:`DECISION_TIMES`);
4. grades every predicted path against the actual one, point by point: the out-of-sample R² of the
   path against the "no change" forecast, per lead time, averaged over the lead times and equally
   over the stocks, beside a luck band and a one-number AR(1) baseline (:func:`score_paths`);
5. lets an Optuna search choose granularity x input length x prediction length on VALIDATION from
   coarse steps (:data:`GRANULARITIES`, :data:`INPUT_LENGTHS`, :data:`PRED_LENGTHS`), skipping the
   combinations the data cannot support (:func:`feasible`) (:func:`search_paths`), and grades the
   winners once on TEST, refit on everything before it (:func:`test_paths`).

ONE model is fit per configuration, for all the stocks (which stock: a static input). Its inputs
are ``strategy.build_inputs``' (bar by bar and daily, no options) on the chosen grid; its target is
:func:`path_inputs`' ``y``. Loaded by path from the notebook. It builds on ``research.py``, which
it loads once and exposes as :data:`rs`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A helper module of this folder, loaded by path once (the notebook reuses the same one)."""
    key = f"darts_{name}"
    path = HERE / f"{name}.py"
    mod = sys.modules.get(key)
    if mod is not None and Path(getattr(mod, "__file__", "")) == path:
        return mod
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _sibling("research")
SM = rs.SM
SESSION = 390
#: Part of every saved path result's key: editing this file or research.py recomputes them.
EXPLORER_VERSION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
RESEARCH_VERSION = hashlib.sha1((HERE / "research.py").read_bytes()).hexdigest()[:10]
#: Market-cap groups (today's cap, Nasdaq screener): lower bound in dollars.
CAP_GROUPS = {"large": 10e9, "mid": 2e9, "small": 0.0}
#: How many names of each (cap group, speed) the suggestion picks.
PLAN = {("large", "slow"): 2, ("large", "fast"): 2, ("mid", "slow"): 2, ("mid", "fast"): 2,
        ("small", "slow"): 2, ("small", "fast"): 2}
#: The volatility percentile (inside its cap group) a "slow" / "fast" pick sits nearest to: near
#: the ends but not at them (the very wildest names are one-off stories, not "fast movers").
SPEED = {"slow": 0.10, "fast": 0.90}
#: The data rules a candidate must pass, in order (the first one failed is its ``why``).
RULES = {
    "1-minute data on every session of the window": lambda c: c["missing"] == 0,
    "a trade in >= 90% of the window's minutes": lambda c: c["traded"] >= 0.90,
    "1-minute data over the last 6 months (>= 95% of sessions)": lambda c: c["recent"] >= 0.95,
    "listed on a US exchange, with a market cap": lambda c: c["market_cap"] > 0,
    "price >= $5 and >= 40 daily returns before the window":
        lambda c: (c["price"] >= 5) & (c["n_returns"] >= 40),
}
#: The searched hyperparameters: the steps the optimizer may pick (each step about doubles).
GRANULARITIES = (1, 2, 5, 10, 30, 60)            # minutes per bar
INPUT_LENGTHS = (50, 100, 200, 400, 1000)        # bars the model reads
PRED_LENGTHS = (10, 20, 50, 100)                 # bars it predicts
#: Every configuration is graded at these decision times: bar closes on every granularity's grid.
DECISION_TIMES = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30")
#: A configuration is tried only with at least this many training windows per stock and this many
#: independent VALIDATION paths per stock (validation bars / prediction length).
MIN_WINDOWS, MIN_OUTCOMES = 200, 10
#: Training windows per stock one epoch samples: a finer grid takes every k-th window.
WINDOW_BUDGET = 2000
#: The last sessions before the graded part are held out of training: training stops when their
#: error has not improved for 2 epochs (the best epoch's weights are kept), and they set the scale
#: of the paths. At least ES_DAYS sessions, more when ES_PATHS independent paths need more.
ES_DAYS, ES_PATHS = 5, 5
#: A bar's move beyond this many typical moves counts as this many (a jump would decide alone).
CLIP = 10.0
#: The multi-step Darts models (point forecasts, MSE loss: the loss the grade is built on).
PATH_MODELS = ("tide", "tsmixer", "dlinear")
#: The fixed settings of every fit (the notebook's §0 explains them).
SETTINGS = {"vol_days": 20, "seed": 0, "device": "cpu", "lr": 1e-3}
#: The smaller settings of a smoke run (``knobs(smoke=True)``): under a minute end to end.
SMOKE = {"names": 4, "months": 3, "train_months": 1, "granularities": (5, 30),
         "inputs": (50, 100), "outputs": (10, 20), "n_trials": 4, "draws": 30, "epochs": 1,
         "compare": ("dlinear", "tide"), "default": (30, 50, 10), "min_windows": 50,
         "min_outcomes": 5}
#: Every input in plain words: (what it is, when it is known, why it could help).
FEATURES = {
    "ret": ("the bar's return", "the bar's close", "short-term momentum or reversal"),
    "rv": ("the bar's volatility, from its 1-minute moves", "the bar's close",
           "volatility clusters; calm and wild hours behave differently"),
    "range": ("the bar's high-low range", "the bar's close", "a range measure of volatility"),
    "clv": ("where the bar closed in its range (-1 low .. +1 high)", "the bar's close",
            "buying or selling pressure into the close of the bar"),
    "flow": ("up-minute minus down-minute volume, as a share of the bar's volume",
             "the bar's close", "order-flow imbalance precedes short-term moves"),
    "rvol": ("the bar's volume vs the same bar of the last 10 sessions", "the bar's close",
             "unusual activity: news, a breakout, an exhaustion"),
    "ret_open": ("the return since today's open", "the bar's close",
                 "intraday momentum: the day's direction so far"),
    "vwap_dist": ("the price vs today's volume-weighted average price", "the bar's close",
                  "a reference price that intraday traders lean on"),
    "mkt": ("the other names' average return over the bar", "the bar's close",
            "the market leads single stocks; separates a stock's own move from the tide"),
    "prev_ret": ("yesterday's return", "the open", "one-day reversal"),
    "ret5": ("the last 5 sessions' return", "the open", "weekly reversal"),
    "ret20": ("the last 20 sessions' return", "the open", "monthly momentum or reversal"),
    "vol20": ("the last 20 sessions' volatility (log)", "the open", "the volatility regime"),
    "pos20": ("yesterday's close in the 20-day high-low range (0 low .. 1 high)", "the open",
              "breakouts and pullbacks"),
    "prev_clv": ("where yesterday closed in its range", "the open",
                 "late-day pressure carrying over"),
    "prev_rvol": ("yesterday's volume vs its 20-day average", "the open",
                  "attention: heavy days are followed by different days"),
    "gap": ("today's overnight gap", "the open", "gaps partly fill or extend"),
}
CALENDAR = {"tod_sin": "time of day (sine)", "tod_cos": "time of day (cosine)",
            "tod_frac": "time of day (fraction of the session)", "dow": "day of the week",
            "opex_week": "monthly option-expiry week"}


# ---------------------------------------------------------------------------------------------
# The run settings.
# ---------------------------------------------------------------------------------------------
def knobs(*, smoke: bool, **run) -> dict:
    """The run's settings: the notebook's knobs, with the smaller smoke settings on top when
    ``smoke`` (:data:`SMOKE`).

    Args:
        smoke: A quick end-to-end test.
        **run: ``model``, ``window``, ``train_months``, ``test_months``, ``granularities``,
            ``inputs``, ``outputs``, ``n_trials``, ``epochs``, ``draws``, ``compare``, ``default``
            (the (granularity, input, prediction) of the worked example).

    Returns:
        The settings dict, plus ``smoke``, ``min_windows`` and ``min_outcomes``.
    """
    k = {"min_windows": MIN_WINDOWS, "min_outcomes": MIN_OUTCOMES, **run, "smoke": smoke}
    if smoke:
        a, b = (pd.Timestamp(x) for x in k["window"])
        start = (b + pd.Timedelta(days=1) - pd.DateOffset(months=SMOKE["months"])).date()
        k.update(window=(str(max(a.date(), start)), str(b.date())),
                 **{x: SMOKE[x] for x in ("train_months", "granularities", "inputs", "outputs",
                                          "n_trials", "draws", "epochs", "compare", "default",
                                          "min_windows", "min_outcomes")})
    return k


def knob_table(k: dict) -> pd.DataFrame:
    """The run's settings as a table (one row per knob)."""
    show = {"smoke": "quick end-to-end test (smaller everything)",
            "model": "the Darts model of the search",
            "window": "the months studied (TRAIN -> VALIDATION -> TEST)",
            "train_months": "months the first fit learns from",
            "test_months": "last months, graded once at the end",
            "granularities": "bar sizes the optimizer may pick (minutes)",
            "inputs": "input lengths it may pick (bars the model reads)",
            "outputs": "prediction lengths it may pick (bars predicted)",
            "n_trials": "configurations the optimizer measures",
            "epochs": "training passes per fit (over a sample of windows on fine grids)",
            "draws": "make-believe futures behind each luck band",
            "compare": "models compared in §6",
            "default": "(granularity, input, prediction) of the worked example",
            "min_windows": "training windows per stock a configuration needs",
            "min_outcomes": "independent VALIDATION paths per stock it needs"}
    return pd.DataFrame([{"knob": n, "value": str(k.get(n)), "meaning": w}
                         for n, w in show.items()]).set_index("knob")


# ---------------------------------------------------------------------------------------------
# Picking the stocks.
# ---------------------------------------------------------------------------------------------
def _digest(names) -> str:
    """A short, order-free fingerprint of a list of symbols (part of a cache file name)."""
    return hashlib.sha1(",".join(sorted(map(str, names))).encode()).hexdigest()[:10]


def _cached(name: str, make, refresh: bool):
    """A pickled result in the strategy's (git-ignored) cache folder, made once."""
    path = rs.CACHE / f"explorer-{name}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    out = make()
    rs.CACHE.mkdir(exist_ok=True)
    out.to_pickle(path)
    return out


def coverage(engine, window, *, recent_months: int = 6, pool=None,
             refresh: bool = False) -> pd.DataFrame:
    """How complete every stored name's regular-session 1-minute data is: over the window, and
    over the last ``recent_months`` months stored. Counts bars only (no prices are read). Saved
    in the cache folder on first use (the scan takes a minute or two).

    Args:
        engine: Database engine.
        window: ``(first, last)`` dates.
        recent_months: The "last months" span.
        pool: Only these symbols (None: all).
        refresh: Re-query instead of reading the saved result.

    Returns:
        One row per symbol with data in the window: ``sessions`` (with data), ``missing``
        (market sessions without), ``stored`` / ``traded`` (share of the market's minutes stored /
        with a trade), ``recent`` (share of the recent market sessions with data); ``attrs``:
        ``sessions`` / ``recent_from`` / ``recent_to``.
    """
    a, b = window
    tag = f"coverage-{a}-{b}-{recent_months}m" + (f"-{_digest(pool)}" if pool else "")

    def make() -> pd.DataFrame:
        only = "AND symbol = ANY(:pool)" if pool else ""
        day = "(ts AT TIME ZONE 'America/New_York')::date"
        q = text(f"SELECT symbol, {day} AS day, count(*) AS n, "
                 "count(*) FILTER (WHERE volume > 0) AS traded FROM prices_ohlcv "
                 f"WHERE bar = '1m' AND rth AND ts >= :a AND ts < :b {only} GROUP BY 1, 2")
        stop = pd.Timestamp(b) + pd.Timedelta(days=1)
        with engine.connect() as c:
            w = pd.read_sql(q, c, params={"a": pd.Timestamp(a), "b": stop, "pool": pool})
            syms = sorted(w["symbol"].unique())
            since = pd.Timestamp.now().normalize() - pd.DateOffset(months=recent_months + 1)
            r = pd.read_sql(text(f"SELECT symbol, {day} AS day, count(*) AS n FROM prices_ohlcv "
                                 "WHERE bar = '1m' AND rth AND symbol = ANY(:s) AND ts >= :a "
                                 "GROUP BY 1, 2"), c, params={"s": syms, "a": since})
        return _coverage(w, r, recent_months, window, SM["HOLIDAYS"])

    return _cached(tag, make, refresh)


def _sessions(a, b, holidays, seen) -> pd.DatetimeIndex:
    """The exchange's sessions from ``a`` to ``b``: weekdays that are not in ``holidays`` and on
    which some name has data (an unscheduled closure has none). A calendar, not a vote of the
    names: the stored names change (about 515 until Feb 2026, about 1,750 from March)."""
    days = pd.bdate_range(pd.Timestamp(a), pd.Timestamp(b))
    days = days[~days.strftime("%Y-%m-%d").isin(list(holidays))]
    return days[days.isin(pd.DatetimeIndex(seen))]


def _coverage(w: pd.DataFrame, r: pd.DataFrame, recent_months: int, window,
              holidays) -> pd.DataFrame:
    """:func:`coverage`'s table from the per-day counts (``w``: window, ``r``: recent), against
    the exchange's calendar (:func:`_sessions`)."""
    w, r = w.assign(day=pd.to_datetime(w["day"])), r.assign(day=pd.to_datetime(r["day"]))
    market = _sessions(window[0], window[1], holidays, w["day"].unique())
    full = w[w["day"].isin(market)].groupby("day")["n"].max().sum()
    g = w[w["day"].isin(market)].groupby("symbol")
    out = pd.DataFrame({"sessions": g["day"].nunique(), "missing": len(market) - g["day"].nunique(),
                        "stored": g["n"].sum() / full, "traded": g["traded"].sum() / full})
    last = r["day"].max()
    first = last - pd.DateOffset(months=recent_months) + pd.Timedelta(days=1)
    rec = _sessions(first, last, holidays, r["day"].unique())
    out["recent"] = (r[r["day"].isin(rec)].groupby("symbol")["day"].nunique() / len(rec)).reindex(
        out.index).fillna(0.0)
    out.attrs.update(sessions=len(market), recent_from=str(first.date()),
                     recent_to=str(last.date()))
    return out


def listing(engine=None, symbols=None, *, refresh: bool = False) -> pd.DataFrame:
    """Today's market cap, sector and company name of every US-listed stock (the free Nasdaq
    screener, as in ``notebooks/01_universe_screener.ipynb``), saved in the cache folder on first
    use so the picks stay put. If the screener is down: SEC share counts x the last daily close
    for ``symbols`` (no sector, no name).

    Args:
        engine: Database engine (the fallback only).
        symbols: The fallback's symbols.
        refresh: Fetch again.

    Returns:
        Indexed by symbol: ``name``, ``sector``, ``market_cap``; ``attrs["source"]``.
    """
    from trading_simple import universe

    def screener() -> pd.DataFrame:
        out = universe.fetch_listed_stocks().set_index("symbol")[["name", "sector",
                                                                  "market_cap"]].copy()
        out.attrs["source"] = f"Nasdaq screener, {pd.Timestamp.now():%Y-%m-%d}"
        return out

    try:
        return _cached("listing", screener, refresh)
    except RuntimeError:
        if engine is None or symbols is None:
            raise
        return _cached(f"listing-sec-{_digest(symbols)}",
                       lambda: _sec_caps(engine, list(symbols)), refresh)


def _sec_caps(engine, symbols: list[str]) -> pd.DataFrame:
    """The fallback market cap: the latest SEC share count x the latest daily close."""
    from trading_simple import features

    with engine.connect() as c:
        d = pd.read_sql(text("SELECT symbol, ts, close FROM prices_ohlcv WHERE bar = '1d' AND "
                             "symbol = ANY(:s) AND ts >= now() - interval '60 days'"), c,
                        params={"s": symbols})
    close = d.pivot_table(index="ts", columns="symbol", values="close")
    close.index = pd.to_datetime(close.index, utc=True)
    shares = features.load_features(engine, close.index, list(close.columns),
                                     ["fund_shares"])["fund_shares"]
    out = pd.DataFrame({"name": None, "sector": None,
                        "market_cap": (close.ffill() * shares).iloc[-1]})
    out.attrs["source"] = "SEC share count x last close (screener unavailable)"
    return out


def daily_profile(engine, symbols, before, *, days: int = 63) -> pd.DataFrame:
    """How fast each name moved BEFORE the window (so the choice uses nothing from it), from the
    stored daily bars: annualized volatility of the daily (dividend-adjusted) returns, the median
    daily dollar volume and the last close.

    Args:
        engine: Database engine.
        symbols: The names.
        before: The window's first day (excluded).
        days: Daily returns used.

    Returns:
        Indexed by symbol: ``vol`` (a year, 0.3 = 30%), ``n_returns``, ``dollar_volume``,
        ``price``.
    """
    a = pd.Timestamp(before) - pd.Timedelta(days=int(days * 1.6) + 10)
    with engine.connect() as c:
        d = pd.read_sql(text("SELECT symbol, ts, close, adj_close, volume FROM prices_ohlcv "
                             "WHERE bar = '1d' AND symbol = ANY(:s) AND ts >= :a AND ts < :b"),
                        c, params={"s": list(symbols), "a": a, "b": pd.Timestamp(before)})
    wide = {k: d.pivot_table(index="ts", columns="symbol", values=k).sort_index().tail(days + 1)
            for k in ("close", "adj_close", "volume")}
    r = np.log(wide["adj_close"]).diff().iloc[1:]
    return pd.DataFrame({"vol": r.std() * math.sqrt(252), "n_returns": r.notna().sum(),
                         "dollar_volume": (wide["close"] * wide["volume"]).median(),
                         "price": wide["close"].ffill().iloc[-1]}).reindex(list(symbols))


def candidates(engine, window, *, pool=None, refresh: bool = False) -> pd.DataFrame:
    """Every name with 1-minute data in the window, with what the pick needs: data completeness
    (:func:`coverage`), today's market cap and its group (:func:`listing`), how fast it moved
    before the window (:func:`daily_profile`) and its volatility percentile inside its cap group,
    and whether it passes every rule of :data:`RULES` (``ok``; ``why``: the first rule failed).

    Args:
        engine: Database engine.
        window: ``(first, last)`` dates.
        pool: Only these symbols (None: all).
        refresh: Re-query instead of reading the saved results.

    Returns:
        One row per name.
    """
    cov = coverage(engine, window, pool=pool, refresh=refresh)
    lst = listing(engine, cov.index, refresh=refresh)
    prof = _cached(f"profile-{window[0]}-{_digest(cov.index)}",
                   lambda: daily_profile(engine, cov.index, window[0]), refresh)
    c = cov.join(lst[~lst.index.duplicated()], how="left").join(prof, how="left")
    c["cap_group"] = pd.cut(c["market_cap"], [*sorted(CAP_GROUPS.values()), np.inf], right=False,
                            labels=sorted(CAP_GROUPS, key=CAP_GROUPS.get)).astype(object)
    passed = pd.Series(True, index=c.index)
    c["why"] = ""
    for rule, test in RULES.items():
        fail = passed & ~test(c).fillna(False).astype(bool)
        c.loc[fail, "why"] = rule
        passed &= ~fail
    c["ok"] = passed
    c["vol_pct"] = c[c["ok"]].groupby("cap_group")["vol"].rank(pct=True)
    c.attrs = {**cov.attrs, "listing": lst.attrs.get("source", "")}
    return c.sort_values("market_cap", ascending=False)


def rule_funnel(c: pd.DataFrame) -> pd.DataFrame:
    """How many names survive each data rule, in order (the pick's funnel)."""
    rows, left = [{"rule": "names with 1-minute data in the window", "names left": len(c)}], len(c)
    for rule in RULES:
        left -= int((c["why"] == rule).sum())
        rows.append({"rule": rule, "names left": left})
    out = pd.DataFrame(rows)
    for g in CAP_GROUPS:
        out.loc[len(out)] = {"rule": f"... of which {g} cap",
                             "names left": int((c["ok"] & (c["cap_group"] == g)).sum())}
    return out.set_index("rule")


def pick(c: pd.DataFrame, plan: dict = PLAN, speed: dict = SPEED, *, near: int = 8) -> pd.DataFrame:
    """The suggested diverse set: for each (cap group, speed) of ``plan``, the names whose
    volatility percentile inside their cap group is nearest to ``speed`` (slow: low, fast: high),
    preferring, among the ``near`` nearest, a sector not picked yet - or, once the nearby sectors
    are all taken, one not picked yet for this (cap group, speed).

    Args:
        c: :func:`candidates`.
        plan: Names per (cap group, speed).
        speed: Target percentile per speed.
        near: How far down the list a new sector may be looked for.

    Returns:
        One row per pick (in plan order): ``symbol``, ``cap_group``, ``speed``, ``name``,
        ``sector``, ``market_cap``, ``vol``, ``vol_pct``, ``traded``.
    """
    rows, used = [], set()
    for (grp, spd), n in plan.items():
        pool = c[c["ok"] & (c["cap_group"] == grp)].copy()
        pool["dist"] = (pool["vol_pct"] - speed[spd]).abs()
        pool = pool.sort_values("dist")
        here = set()
        for _ in range(n):
            pool = pool[~pool.index.isin([r["symbol"] for r in rows])]
            if pool.empty:
                break
            head = pool.head(near)
            fresh = head[~head["sector"].isin(used)]
            if fresh.empty:
                fresh = head[~head["sector"].isin(here)]
            sym = (fresh if len(fresh) else head).index[0]
            used.add(pool.loc[sym, "sector"])
            here.add(pool.loc[sym, "sector"])
            rows.append({"symbol": sym, "cap_group": grp, "speed": spd,
                         **pool.loc[sym, ["name", "sector", "market_cap", "vol", "vol_pct",
                                          "traded"]].to_dict()})
    return pd.DataFrame(rows)


def default_names(picks: pd.DataFrame, *, smoke: bool) -> list[str]:
    """The names to study: every pick, or for a smoke run one each of large-slow, large-fast,
    small-slow and small-fast (the extremes, so every step still sees a slow and a fast name)."""
    if not smoke:
        return list(picks["symbol"])
    first = picks.drop_duplicates(["cap_group", "speed"]).set_index(["cap_group", "speed"])
    want = [("large", "slow"), ("large", "fast"), ("small", "slow"), ("small", "fast")]
    return [first.loc[k, "symbol"] for k in want if k in first.index][:SMOKE["names"]]


def example(picks: pd.DataFrame, names, sp: dict, *, k: int = 4,
            clock: str = "12:30") -> tuple[str, str]:
    """The worked example: the first fast large cap among ``names`` (else the first name) at
    ``clock`` on the ``k``-th session after VALIDATION starts (12:30 closes a bar on a 30- and a
    60-minute grid)."""
    fast = set(picks.loc[(picks["speed"] == "fast") & (picks["cap_group"] == "large"), "symbol"])
    name = next((n for n in names if n in fast), list(names)[0])
    day = sp["sessions"][sp["sessions"].index(sp["validation"][0]) + k]
    return name, f"{day} {clock}"


def pick_table(picks: pd.DataFrame) -> pd.DataFrame:
    """The picks in readable units."""
    return pd.DataFrame({
        "cap group": picks["cap_group"], "speed": picks["speed"], "company": picks["name"].str[:40],
        "sector": picks["sector"], "market cap $B": (picks["market_cap"] / 1e9).round(1),
        "volatility % a year": (100 * picks["vol"]).round(0),
        "volatility percentile in its group": (100 * picks["vol_pct"]).round(0),
        "minutes with a trade %": (100 * picks["traded"]).round(1)}).set_index(picks["symbol"])


# ---------------------------------------------------------------------------------------------
# The window and its split.
# ---------------------------------------------------------------------------------------------
def months(window, *, train_months: int = 2, test_months: int = 1) -> dict:
    """The window cut by calendar month: TRAIN (the first ``train_months``), TEST (the last
    ``test_months``) and VALIDATION (the months between), as inclusive date pairs."""
    a, b = (pd.Timestamp(x) for x in window)
    val_from = a + pd.DateOffset(months=train_months)
    test_from = b + pd.Timedelta(days=1) - pd.DateOffset(months=test_months)
    if not val_from < test_from:
        raise ValueError(f"the window {window} leaves no VALIDATION month")
    day = pd.Timedelta(days=1)
    return {"train": (a, val_from - day), "validation": (val_from, test_from - day),
            "test": (test_from, b)}


def load_window(settings, engine, symbols, window, *, train_months: int = 2,
                test_months: int = 1, warmup: int | None = None, read_locked: bool = False):
    """The names' 1-minute panel over the window, with exactly ``warmup`` sessions before it
    (the inputs' 20-day history; the model never learns from them), dividend-adjusted, and the
    split as session dates. ``warmup`` defaults to ``max(21, vol_days + 1)`` with
    :data:`SETTINGS`' ``vol_days``: the inputs' 20-day history and the typical-move scale.

    The default window sits inside the darts study's TUNE months. A window reaching into its
    locked TEST / HOLDOUT months (``research.WINDOWS``) raises unless ``read_locked``: whatever
    is chosen here from those months can no longer be tested on them.

    Args:
        settings: Global settings.
        engine: Database engine.
        symbols: The names.
        window: ``(first, last)`` dates.
        train_months: Months before the first fit.
        test_months: Last months, graded once at the end.
        warmup: Sessions before the window (the inputs' warm-up).
        read_locked: Allow a window past the TUNE months.

    Raises:
        ValueError: A window past TUNE without ``read_locked``, a name without 1-minute data in
            it, or too little history before it.

    Returns:
        ``(panel, split)``; ``split`` holds ``warmup`` / ``train`` / ``validation`` / ``test``
        (first and last session, "YYYY-MM-DD"), ``train_sessions``, ``sessions`` (all, in order)
        and ``window``.
    """
    if str(window[1]) > rs.WINDOWS["tune"][1] and not read_locked:
        raise ValueError(f"the window ends {window[1]}, after the TUNE months (to "
                         f"{rs.WINDOWS['tune'][1]}): it reads the locked TEST / HOLDOUT months. "
                         "Set READ_LOCKED = True to accept that.")
    warmup = warmup or max(21, SETTINGS["vol_days"] + 1)
    cut = months(window, train_months=train_months, test_months=test_months)
    start = (pd.Timestamp(window[0]) - pd.Timedelta(days=int(warmup * 1.6) + 14)).date()
    panel = rs.load(settings, engine, start=str(start), end=str(window[1]), options=False,
                    symbols=list(symbols), adjust=True)
    missing = sorted(set(symbols) - set(panel.symbols))
    if missing:
        raise ValueError(f"no 1-minute data in the window for {missing}: drop them from SYMBOLS")
    day = np.asarray(panel.close.index.tz_convert(panel.tz).date).astype("datetime64[D]")
    sess = np.unique(day)
    first = int(np.searchsorted(sess, np.datetime64(pd.Timestamp(window[0]).date())))
    if first < warmup:
        raise ValueError(f"only {first} sessions stored before {window[0]}: the warm-up needs "
                         f"{warmup}")
    keep = day >= sess[first - warmup]
    panel = replace(panel, **{k: getattr(panel, k)[keep]
                              for k in ("open_", "high", "low", "close", "volume")})
    sess = [str(d) for d in sess[first - warmup:]]
    out = {"window": tuple(str(x) for x in window), "sessions": sess,
           "warmup": (sess[0], sess[warmup - 1])}
    for k, (a, b) in cut.items():
        inside = [d for d in sess if str(a.date()) <= d <= str(b.date())]
        out[k] = (inside[0], inside[-1])
    out["train_sessions"] = sum(out["train"][0] <= d <= out["train"][1] for d in sess)
    return panel, out


def split_table(sp: dict) -> pd.DataFrame:
    """The split as a table: each part's sessions and what it is used for."""
    use = {"warmup": "inputs only: the 20-day inputs and the volatility scale need history",
           "train": "the first fit learns from these outcomes",
           "validation": "graded out of sample; the optimizer chooses on these scores",
           "test": "graded ONCE at the end, for the chosen settings only"}
    rows = []
    for k, w in use.items():
        a, b = sp[k]
        rows.append({"part": k.upper(), "from": a, "to": b,
                     "sessions": sum(a <= d <= b for d in sp["sessions"]), "use": w})
    return pd.DataFrame(rows).set_index("part")


def data_quality(panel, picks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per name, on the loaded panel: sessions, the share of regular minutes stored and with a
    trade, the largest close-to-close day move (a split slipping through would show as one huge
    move) and, when ``picks`` is given, its cap group and speed.

    Args:
        panel: The 1-minute panel (dividend-adjusted).
        picks: :func:`pick` (optional).

    Returns:
        One row per name.
    """
    reg = SM["regular_rows"](panel)
    day = pd.Index(panel.close.index.tz_convert(panel.tz).date)
    expected = reg.sum()                      # every session's real minutes, stored or not
    close = panel.close[reg]
    daily = close.groupby(day[reg]).last()
    out = pd.DataFrame({
        "sessions": close.notna().groupby(day[reg]).any().sum(),
        "minutes stored %": 100 * close.notna().sum() / expected,
        "minutes with a trade %": 100 * (panel.volume[reg] > 0).sum() / expected,
        "largest day move %": 100 * np.log(daily / daily.shift(1)).abs().max()})
    if picks is not None:
        out = picks.set_index("symbol")[["cap_group", "speed"]].join(out, how="right")
    return out


# ---------------------------------------------------------------------------------------------
# The path forecaster: what it sees, what it predicts, fit and predict.
# ---------------------------------------------------------------------------------------------
_PIN: dict = {}


def path_inputs(panel, g: int, *, vol_days: int = SETTINGS["vol_days"],
                extra: int = max(PRED_LENGTHS) + 1) -> dict:
    """Everything a path forecaster sees and predicts on a ``g``-minute grid (kept in-process per
    panel content and grid, the last 8).

    The target ``y`` is each bar's close-to-close log return (a session's first bar holds the
    night) in TYPICAL MOVES: divided by ``u``, the root-mean-square of the same clock position's
    returns over the previous ``vol_days`` sessions (causal: the open's bar is scaled by the
    open's own, gap-inclusive, size, a midday bar by the midday size), floored at a quarter of the
    stock's typical bar move over the same previous sessions (``build_inputs``' ``scale``, causal
    too), and clipped at ±:data:`CLIP`. A half-day's empty bars after 13:00 are missing.
    The inputs are ``strategy.build_inputs``' bar-by-bar, market and daily inputs (no options);
    the calendar runs ``extra`` bars past the panel end.

    Args:
        panel: The 1-minute panel.
        g: Minutes per bar.
        vol_days: Sessions behind ``u``.
        extra: Calendar bars past the panel end.

    Returns:
        ``{"g": grid, "names", "y", "valid", "u", "close", "times" (each bar's close, New York),
        "past" {name: (bars x k)}, "past_names", "groups", "future" ((bars + extra) x 5),
        "future_names", "bar_minutes"}``.
    """
    key = (SM["fingerprint"](panel), g, vol_days, extra)
    if key in _PIN:
        return _PIN[key]
    inp = SM["build_inputs"](panel, bar_minutes=g, horizon=1, options="none", vol_days=vol_days)
    gr, close = inp["g"], inp["close"]
    r = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = np.log(close[1:] / close[:-1])
    r[~gr["regular"]] = np.nan
    ms = pd.DataFrame(r ** 2).groupby(gr["pos"]).transform(
        lambda x: x.shift(1).rolling(vol_days, min_periods=max(3, vol_days // 2)).mean())
    u = np.fmax(np.sqrt(ms.to_numpy()), 0.25 * inp["scale"])     # both from earlier sessions
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.clip(r / u, -CLIP, CLIP)
    out = {"g": gr, "names": inp["names"], "y": y, "valid": np.isfinite(y), "u": u,
           "close": close, "times": rs.decision_times(gr), "past": inp["past"],
           "past_names": inp["past_names"], "groups": inp["groups"],
           "future": SM["time_features"](gr, extra=extra), "future_names": inp["future_names"],
           "bar_minutes": g}
    _PIN[key] = out
    while len(_PIN) > 8:
        _PIN.pop(next(iter(_PIN)))
    return out


def _covariates(pin: dict) -> tuple:
    """Per-name past-input series, the calendar series and the static "which stock" frames, built
    once per inputs."""
    if "_cov" not in pin:
        from darts import TimeSeries

        n = len(pin["g"]["code"])
        pc = [TimeSeries.from_times_and_values(pd.RangeIndex(0, n), np.nan_to_num(pin["past"][s]),
                                               columns=pin["past_names"]) for s in pin["names"]]
        fc = TimeSeries.from_times_and_values(pd.RangeIndex(0, len(pin["future"])),
                                              pin["future"], columns=pin["future_names"])
        stat = [pd.DataFrame({f"is_{x}": [float(x == s)] for x in pin["names"]})
                for s in pin["names"]]
        pin["_cov"] = (pc, fc, stat)
    return pin["_cov"]


def windows(n_before: int, lo: int, L: int, H: int) -> int:
    """Training windows of one stock: ``L`` input bars and ``H`` labels, the labels reaching past
    the first trained bar ``lo`` and ending before ``n_before`` (the first graded bar). Inputs may
    reach back before ``lo`` (into the warm-up)."""
    return max(0, n_before - H - max(L, lo - H + 1) + 1)


def held_out_sessions(g: int, H: int) -> int:
    """Sessions held out at the end of the training data: :data:`ES_DAYS`, or more when
    :data:`ES_PATHS` independent paths of ``H`` bars need more."""
    return max(ES_DAYS, math.ceil(ES_PATHS * H / math.ceil(SESSION / g)))


def feasible(g: int, L: int, H: int, sp: dict, *, min_windows: int = MIN_WINDOWS,
             min_outcomes: int = MIN_OUTCOMES) -> tuple[bool, str]:
    """Can the data support this configuration? It needs ``min_windows`` training windows per
    stock (:func:`windows`, before the held-out slice that ends the training data,
    :func:`held_out_sessions`) and ``min_outcomes`` independent VALIDATION paths per stock
    (validation bars / prediction length).

    Returns:
        ``(ok, why)``.
    """
    n_pos = math.ceil(SESSION / g)
    sess = sp["sessions"]
    lo = sess.index(sp["train"][0]) * n_pos
    hi = (sess.index(sp["validation"][0]) - held_out_sessions(g, H)) * n_pos
    n_val = sum(sp["validation"][0] <= d <= sp["validation"][1] for d in sess) * n_pos
    w, o = windows(hi, lo, L, H), n_val / H
    if w < min_windows:
        return False, f"{w:,} training windows per stock (needs {min_windows})"
    if o < min_outcomes:
        return False, f"{o:.0f} independent validation paths per stock (needs {min_outcomes})"
    return True, f"{w:,} training windows, {o:.0f} independent validation paths per stock"


def feasibility_table(sp: dict, grans, inputs, outputs, **kw) -> pd.DataFrame:
    """Which (granularity, input, prediction) the data supports: one row per granularity and
    input length, one column per prediction length ("ok" or why not)."""
    rows = []
    for g in grans:
        for L in inputs:
            row = {"granularity (min)": g, "input (bars)": L,
                   "input span (sessions)": round(L / math.ceil(SESSION / g), 1)}
            for H in outputs:
                ok, why = feasible(g, L, H, sp, **kw)
                row[f"predict {H}"] = "ok" if ok else why.split(" (")[0]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["granularity (min)", "input (bars)"])


class _KeepBest:
    """A Lightning callback keeping (in memory) the weights of the epoch with the lowest
    validation error; :func:`fit_paths` restores them after training."""

    def __new__(cls):
        import pytorch_lightning as pl

        class KeepBest(pl.Callback):
            def __init__(self):
                self.best, self.state = np.inf, None

            def on_validation_epoch_end(self, trainer, pl_module):
                v = trainer.callback_metrics.get("val_loss")
                if v is not None and not trainer.sanity_checking and float(v) < self.best:
                    self.best = float(v)
                    self.state = {k: t.detach().clone() for k, t in pl_module.state_dict().items()}

        return KeepBest()


def make_path_model(kind: str, L: int, H: int, *, epochs: int, n_samples: int, seed: int = 0,
                    device: str = "cpu"):
    """A fresh multi-step Darts model: it reads ``L`` bars and predicts the next ``H`` at once
    (point forecasts, mean-squared-error loss), with the calendar known ahead and the stock as a
    static input; at most ``epochs`` passes, stopped early on a held-out slice (:data:`ES_DAYS`).
    TiDE / TSMixer as in ``strategy.make_model``; DLinear splits the input into trend and
    remainder and maps each linearly to the path."""
    import pytorch_lightning  # noqa: F401  (imported first so the quieting sticks)
    from darts.models import DLinearModel, TiDEModel, TSMixerModel

    SM["_quiet"]()
    trainer = {"accelerator": device, "enable_progress_bar": False, "enable_model_summary": False,
               "logger": False, "enable_checkpointing": False, "num_sanity_val_steps": 0,
               "callbacks": [SM["_early_stopping"](), _KeepBest()]}
    common = dict(input_chunk_length=L, output_chunk_length=H, n_epochs=epochs,
                  batch_size=int(np.clip(n_samples // 40, 32, 512)), random_state=seed,
                  use_static_covariates=True, pl_trainer_kwargs=trainer,
                  optimizer_kwargs={"lr": SETTINGS["lr"]})
    if kind == "tide":
        return TiDEModel(hidden_size=64, num_encoder_layers=1, num_decoder_layers=1,
                         decoder_output_dim=8, temporal_decoder_hidden=16, dropout=0.1, **common)
    if kind == "tsmixer":
        return TSMixerModel(hidden_size=32, ff_size=32, num_blocks=2, dropout=0.1, **common)
    if kind == "dlinear":
        return DLinearModel(kernel_size=25, **common)
    raise ValueError(f"model must be one of {PATH_MODELS}, got {kind!r}")


def fit_paths(pin: dict, kind: str, L: int, H: int, lo: int, hi: int, *, epochs: int,
              seed: int = 0, device: str = "cpu") -> tuple:
    """ONE model for every stock, fit on the labels of bars ``lo .. hi-1``.

    The last :func:`held_out_sessions` before ``hi`` (the first graded bar) are held out to stop
    training early: training windows end before them (``cut``), the held-out windows' labels lie
    in ``cut .. hi-1`` (their inputs may read the training bars, never the reverse), and the
    weights of the epoch with the lowest held-out error are kept. Training inputs may reach back
    before ``lo`` (labels there weigh 0). A fine grid samples every k-th window (``stride``) so
    one epoch sees about :data:`WINDOW_BUDGET` windows per stock.

    Returns:
        ``(model, info)``; ``info``: ``windows`` (per stock), ``stride``, ``samples`` (per epoch,
        all stocks), ``epochs`` (run), ``best_val_loss`` (kept), ``seconds``, ``train_to`` (the
        last training label's session), ``held_out`` (first and last session).
    """
    from darts import TimeSeries

    pc, fc, stat = _covariates(pin)
    sf, code = pin["g"]["sess_first"], pin["g"]["code"]
    cut = int(sf[max(int(code[lo]) + 1, int(code[hi]) - held_out_sessions(pin["bar_minutes"], H))])
    y = np.nan_to_num(pin["y"]).astype(np.float32)
    w = (pin["valid"] & (np.arange(len(y)) >= lo)[:, None]).astype(np.float32)

    def part(a: int, b: int, first: int) -> tuple[list, list]:
        idx = pd.RangeIndex(a, b)
        ww = np.where(np.arange(a, b)[:, None] >= first, w[a:b], 0.0).astype(np.float32)
        ser = [TimeSeries.from_times_and_values(idx, y[a:b, k:k + 1], columns=["y"])
               .with_static_covariates(stat[k]) for k in range(len(pin["names"]))]
        wts = [TimeSeries.from_times_and_values(idx, ww[:, k:k + 1], columns=["w"])
               for k in range(len(pin["names"]))]
        return ser, wts

    series, weights = part(SM["series_start"](lo, L, H, True), cut, lo)
    val, val_w = part(max(0, cut - L), hi, cut)
    n_win = windows(cut, lo, L, H)
    stride = max(1, math.ceil(n_win / WINDOW_BUDGET))
    samples = (n_win // stride) * len(series)
    est = make_path_model(kind, L, H, epochs=epochs, n_samples=samples, seed=seed, device=device)
    fut = [fc] * len(series)
    kw = dict(future_covariates=fut, val_future_covariates=fut) \
        if est.supports_future_covariates else {}
    t0 = time.time()
    est.fit(series, past_covariates=pc, sample_weight=weights, val_series=val,
            val_past_covariates=pc, val_sample_weight=val_w, stride=stride, verbose=False, **kw)
    SM["_quiet"]()
    keep = next(c for c in est.trainer_params["callbacks"] if hasattr(c, "state"))
    if keep.state is not None:
        est.model.load_state_dict(keep.state)
    run = int(est.epochs_trained) if hasattr(est, "epochs_trained") else epochs
    return est, {"windows": n_win, "stride": stride, "samples": samples, "epochs": run,
                 "best_val_loss": round(keep.best, 5), "seconds": round(time.time() - t0, 1),
                 "train_to": str(pin["g"]["date"][cut - 1]),
                 "held_out": (str(pin["g"]["date"][cut]), str(pin["g"]["date"][hi - 1]))}


def decisions(pin: dict, part, H: int, *, times=DECISION_TIMES) -> tuple[np.ndarray, np.ndarray]:
    """The graded decisions of ``part`` (first, last session): the bars closing at ``times``
    (None: every bar) whose next ``H`` bars are all real, valid and inside ``part``, one row per
    (bar, stock).

    Returns:
        ``(bars, names)``: grid bar and stock column per decision, by bar.
    """
    t, valid = pin["times"], pin["valid"]
    day, clock = np.asarray(t.strftime("%Y-%m-%d")), np.asarray(t.strftime("%H:%M"))
    n = len(day)
    at = np.isin(clock, times) if times is not None else np.ones(n, bool)
    bars_ = np.flatnonzero(at & (day >= part[0]) & (day <= part[1]) & pin["g"]["regular"])
    bars_ = bars_[bars_ + H < n]
    bars_ = bars_[day[bars_ + H] <= part[1]]
    cs = np.vstack([np.zeros((1, valid.shape[1]), int), np.cumsum(valid, axis=0)])
    rows, cols = np.nonzero(cs[bars_ + H + 1] - cs[bars_ + 1] == H)     # bars s+1 .. s+H valid
    return bars_[rows], cols


def predict_paths(est, pin: dict, s: np.ndarray, j: np.ndarray, L: int, H: int, *,
                  batch: int = 1024) -> np.ndarray:
    """The fitted model's path from each decision: the next ``H`` bar moves (typical moves) of
    stock ``j`` predicted from its last ``L`` bars up to and including bar ``s``.

    Returns:
        A (decisions x H) array.
    """
    from darts import TimeSeries

    pc, fc, stat = _covariates(pin)
    y = np.nan_to_num(pin["y"]).astype(np.float32)
    series = [TimeSeries.from_times_and_values(pd.RangeIndex(b - L + 1, b + 1),
                                               y[b - L + 1:b + 1, k:k + 1], columns=["y"])
              .with_static_covariates(stat[k]) for b, k in zip(s, j, strict=True)]
    kw = {"future_covariates": [fc] * len(series)} if est.supports_future_covariates else {}
    out = est.predict(n=H, series=series, past_covariates=[pc[k] for k in j], batch_size=batch,
                      verbose=False, **kw)
    return np.stack([p.values()[:, 0] for p in out]).astype(np.float32)


def actual_paths(pin: dict, s: np.ndarray, j: np.ndarray, H: int) -> np.ndarray:
    """The bar moves that really followed each decision (typical moves), (decisions x H)."""
    return pin["y"][s[:, None] + np.arange(1, H + 1), j[:, None]].astype(np.float32)


def ar1(pin: dict, lo: int, hi: int) -> np.ndarray:
    """Per stock: the lag-1 autocorrelation of the bar moves of bars ``lo .. hi-1``, pairs inside
    one session (on 1-minute bars mostly the bid-ask bounce)."""
    y, code = pin["y"][lo:hi], pin["g"]["code"][lo:hi]
    same = code[1:] == code[:-1]
    rho = []
    for k in range(y.shape[1]):
        a, b = y[:-1, k], y[1:, k]
        ok = same & np.isfinite(a) & np.isfinite(b)
        rho.append(float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else 0.0)
    return np.array(rho)


def ar1_paths(pin: dict, s: np.ndarray, j: np.ndarray, rho: np.ndarray, H: int) -> np.ndarray:
    """The AR(1) baseline: each next bar repeats ``rho`` x the previous one, so the path from the
    decision bar's move ``y_s`` is ``y_s * rho^k`` (no input but that one number per stock)."""
    last = np.nan_to_num(pin["y"][s, j])
    return (last[:, None] * rho[j][:, None] ** np.arange(1, H + 1)).astype(np.float32)


def calibration(pred: np.ndarray, act: np.ndarray) -> float:
    """The scale ``beta`` (0 .. 1) that makes predicted paths agree best with the actual ones the
    way they are graded: it maximizes the mean over lead times of R² (each lead's squared errors
    divided by that lead's sum of squared actual paths, so the long leads do not dominate). A
    network's paths come out too loud for what it knows (it fits noise); ``beta`` = 0 says the
    paths did not help at all (predict "no change"), 1 keeps them as they are."""
    p, c = np.cumsum(pred, 1, dtype=np.float64), np.cumsum(act, 1, dtype=np.float64)
    w = 1.0 / np.maximum((c * c).sum(axis=0), 1e-12)          # per lead, as the grade weighs it
    den = float(((p * p).sum(axis=0) * w).sum())
    return float(np.clip(((p * c).sum(axis=0) * w).sum() / den, 0.0, 1.0)) if den > 0 else 0.0


def scaled(pred: np.ndarray, beta: float) -> np.ndarray:
    """Predicted bar moves with the whole path scaled by ``beta``."""
    return (beta * pred).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# The grade: path R² against "no change", per stock and lead time, with a luck band.
# ---------------------------------------------------------------------------------------------
def _r2(pc: np.ndarray, ac: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Per stock (rows) and lead (columns): 1 - sum (predicted - actual)² / sum actual² of the
    cumulative paths ``pc`` / ``ac`` (draws x decisions x H or decisions x H)."""
    out = []
    for k in np.unique(j):
        m = j == k
        err = ((pc[..., m, :] - ac[..., m, :]) ** 2).sum(axis=-2)
        base = (ac[..., m, :] ** 2).sum(axis=-2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(1 - err / np.where(base > 0, base, np.nan))
    return np.stack(out, axis=-2)


def _corr(pc: np.ndarray, ac: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Per stock (rows) and lead (columns): the Pearson correlation of the predicted and actual
    cumulative paths ``pc`` (decisions x H) / ``ac`` (draws x decisions x H or decisions x H)
    across the decisions."""
    out = []
    for k in np.unique(j):
        m = j == k
        p = pc[m] - pc[m].mean(axis=0)
        a = ac[..., m, :] - ac[..., m, :].mean(axis=-2, keepdims=True)
        den = np.sqrt((p * p).sum(axis=0) * (a * a).sum(axis=-2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((p * a).sum(axis=-2) / np.where(den > 0, den, np.nan))
    return np.stack(out, axis=-2)


def score_paths(pred: np.ndarray, act: np.ndarray, s: np.ndarray, j: np.ndarray, names, *,
                draws: int = 0, seed: int = 0) -> dict:
    """The grade of predicted paths against the actual ones, point by point.

    Both are cumulated from the decision (the path in typical moves). Per stock and lead time
    ``h``: R²(h) = 1 - Σ (predicted - actual)² / Σ actual², the out-of-sample R² against the "no
    change" forecast (0 = as good as saying nothing, above 0 = better, below = worse); the stock's
    score is its mean over the lead times, the grade the mean over the stocks (equal weights).
    Direction: ``corr``, the correlation of predicted and actual change per stock and lead time,
    averaged (the scale of the paths cannot move it). Its luck band: the same predictions against
    ``draws`` make-believe futures in which every bar's move keeps its size but gets a random
    sign, one coin per bar for all stocks (overlap and co-movement stay, direction is gone);
    ``z`` = how many luck-SDs ``corr`` sits above luck's centre.

    Returns:
        ``r2`` (the grade), ``r2_h`` (mean over stocks, per lead), ``r2_by_name``, ``share_pos``
        (stocks above 0), ``worst``, ``corr`` / ``corr_h`` (per lead), ``hit_h`` (same sign, %),
        ``n_paths``, ``outcomes`` (independent paths per stock, roughly), and with ``draws``
        ``null_mean`` / ``null_sd`` / ``z`` (of ``corr``).
    """
    pc, ac = np.cumsum(pred, 1, dtype=np.float64), np.cumsum(act, 1, dtype=np.float64)
    H = pred.shape[1]
    r2 = _r2(pc, ac, j)                                       # stocks x H
    corr = _corr(pc, ac, j)                                   # stocks x H
    keys = [names[k] for k in np.unique(j)]
    by_name = pd.Series(np.nanmean(r2, axis=1), index=keys)
    hit = [100 * ((pc[j == k] > 0) == (ac[j == k] > 0)).mean(axis=0) for k in np.unique(j)]
    per_name = pd.Series(j).value_counts()
    span = (s.max() - s.min() + H) / H if len(s) else 0.0
    with np.errstate(invalid="ignore"):
        out = {"r2": float(by_name.mean()), "r2_h": np.nanmean(r2, axis=0),
               "r2_by_name": by_name, "share_pos": float((by_name > 0).mean()),
               "worst": float(by_name.min()), "corr": float(np.nanmean(corr)) if np.isfinite(
                   corr).any() else np.nan, "corr_h": np.nanmean(corr, axis=0)
               if np.isfinite(corr).any() else np.full(H, np.nan),
               "hit_h": np.mean(hit, axis=0), "n_paths": len(s),
               "outcomes": float(min(per_name.mean(), span))}
    if draws and np.isfinite(out["corr"]):
        rng = np.random.default_rng(seed)
        idx = s[:, None] + np.arange(1, H + 1)
        null = []
        for a in range(0, draws, 25):
            sign = rng.choice(np.array([-1.0, 1.0], np.float32),
                              size=(min(25, draws - a), int(idx.max()) + 1))
            flipped = np.cumsum(sign[:, idx] * act[None], axis=2, dtype=np.float64)
            null.extend(np.nanmean(_corr(pc, flipped, j), axis=(1, 2)))
        mu, sd = float(np.mean(null)), float(np.std(null))
        out.update(null_mean=mu, null_sd=sd, z=(out["corr"] - mu) / sd if sd > 0 else np.nan)
    return out


# ---------------------------------------------------------------------------------------------
# One configuration, the search, the model comparison and the TEST check.
# ---------------------------------------------------------------------------------------------
def run_path(panel, sp: dict, cell, *, kind: str, epochs: int, draws: int = 200,
             part: str = "validation", seed: int = 0, device: str = "cpu") -> dict:
    """One configuration end to end: fit ONE model on every label before ``part`` (VALIDATION:
    TRAIN only; TEST: TRAIN and VALIDATION; its last :func:`held_out_sessions` held out to stop
    training early), scale its paths by :func:`calibration` on that held-out slice, predict the
    path from every decision of ``part`` and grade it (:func:`score_paths`) beside the unscaled
    paths and the AR(1) baseline. Saved in the cache folder, keyed by the configuration, the
    split, the panel's content, the code (explorer.py, research.py, strategy.py) and the
    module's constants.

    Args:
        panel: The panel (:func:`load_window`).
        sp: Its split.
        cell: ``(granularity, input length, prediction length)``.
        kind: The Darts model (:data:`PATH_MODELS`).
        epochs: Training passes.
        draws: Luck-band futures (0: none).
        part: ``"validation"`` or ``"test"``.
        seed: Model and luck seed.
        device: ``cpu`` | ``mps``.

    Returns:
        ``cell``, ``kind``, ``part``, ``s`` / ``j`` (decision bar and stock), ``pred`` (the
        predicted bar moves, scaled), ``raw`` (unscaled), ``beta``, ``score`` (of ``pred``: the
        R² grade), ``score_raw`` (of ``raw``, with the luck band: direction does not depend on
        the scale, so ``corr`` and ``z`` are read here, also when ``beta`` is 0), ``ar1`` (the
        baseline's score), ``rho``, ``fit`` (its info plus ``train_from`` / ``train_last`` /
        ``first_graded`` dates and ``calibration_paths``).
    """
    g, L, H = (int(x) for x in cell)
    ident = json.dumps({"cell": [g, L, H], "kind": kind, "epochs": epochs, "draws": draws,
                        "part": part, "seed": seed, "device": device,
                        "split": [sp["train"], sp["validation"], sp["test"]],
                        "panel": SM["fingerprint"](panel),
                        "code": [rs.CODE_VERSION, EXPLORER_VERSION, RESEARCH_VERSION],
                        "constants": [ES_DAYS, ES_PATHS, CLIP, WINDOW_BUDGET, DECISION_TIMES,
                                      SETTINGS]}, sort_keys=True, default=str)
    path = rs.CACHE / (f"paths-{kind}-{g}m-L{L}-H{H}-{part}-"
                       f"{hashlib.sha1(ident.encode()).hexdigest()[:12]}.pkl")
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    pin = path_inputs(panel, g)
    sess, sf, dates = sp["sessions"], pin["g"]["sess_first"], pin["g"]["date"]
    lo, hi = int(sf[sess.index(sp["train"][0])]), int(sf[sess.index(sp[part][0])])
    est, info = fit_paths(pin, kind, L, H, lo, hi, epochs=epochs, seed=seed, device=device)
    cs, cj = decisions(pin, info["held_out"], H, times=None)       # every bar of the held-out week
    keep = np.isin(cs, np.unique(cs)[::max(1, len(np.unique(cs)) // 300)])
    cs, cj = cs[keep], cj[keep]
    beta = calibration(predict_paths(est, pin, cs, cj, L, H), actual_paths(pin, cs, cj, H)) \
        if len(cs) else 0.0
    s, j = decisions(pin, sp[part], H)
    raw = predict_paths(est, pin, s, j, L, H)
    pred, act = scaled(raw, beta), actual_paths(pin, s, j, H)
    rho = ar1(pin, lo, hi)
    out = {"cell": (g, L, H), "kind": kind, "part": part, "s": s, "j": j, "pred": pred,
           "raw": raw, "beta": beta,
           "score": score_paths(pred, act, s, j, pin["names"]),
           "score_raw": score_paths(raw, act, s, j, pin["names"], draws=draws, seed=seed),
           "ar1": score_paths(ar1_paths(pin, s, j, rho, H), act, s, j, pin["names"]),
           "rho": rho, "fit": {**info, "train_from": str(dates[lo]),
                               "train_last": str(dates[hi - 1]), "calibration_paths": len(cs),
                               "first_graded": str(dates[s.min()]) if len(s) else ""}}
    rs.CACHE.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    os.replace(tmp, path)
    return out


def _row(r: dict) -> dict:
    """One configuration's numbers (a search / table row): the R² grade of the scaled paths
    (``r2``, ``r2_h``, ``share_pos``, ``worst``), the scale ``beta``, the unscaled paths' R²
    (``r2_raw``), and their direction (``corr``, ``corr_h``) with its luck band (``null_mean`` /
    ``null_sd``: ``z``)."""
    g, L, H = r["cell"]
    sc, raw, n_pos = r["score"], r["score_raw"], math.ceil(SESSION / g)
    return {"granularity": g, "input": L, "prediction": H, "r2": sc["r2"],
            "beta": r["beta"], "r2_raw": raw["r2"], "corr": raw["corr"],
            "z": raw.get("z", np.nan), "null_mean": raw.get("null_mean", np.nan),
            "null_sd": raw.get("null_sd", np.nan), "share_pos": sc["share_pos"],
            "worst": sc["worst"], "ar1_r2": r["ar1"]["r2"], "minutes_ahead": H * g,
            "input_sessions": L / n_pos, "outcomes": raw["outcomes"], "paths": raw["n_paths"],
            "r2_h": sc["r2_h"], "corr_h": raw["corr_h"], "epochs": r["fit"]["epochs"],
            "stride": r["fit"]["stride"], "fit_s": r["fit"]["seconds"]}


def search_paths(panel, sp: dict, *, kind: str, grans, inputs, outputs, n_trials: int,
                 epochs: int, draws: int = 200, seed: int = 0, device: str = "cpu",
                 min_windows: int = MIN_WINDOWS, min_outcomes: int = MIN_OUTCOMES,
                 verbose: bool = True) -> tuple[pd.DataFrame, object]:
    """Optuna's choice of granularity, input length and prediction length for ``kind``: the
    configuration whose predicted paths agree best with the actual ones on VALIDATION (the grade
    of :func:`score_paths`, averaged over all the stocks: ONE model per configuration for them
    all).

    Each trial asks Optuna's TPE sampler for one STEP of each of the three ladders (an integer
    position: nothing between two steps exists, and the sampler knows which steps are
    neighbours). A configuration the data cannot support (:func:`feasible`) or one already
    measured is not measured; its nearest unmeasured feasible neighbour is, so every trial adds a
    new point (the redirected proposal is marked failed, which the sampler ignores). The first
    quarter of the trials are spread at random. TEST is never read here.

    Returns:
        ``(results, study)``: one row per configuration measured (``trial``, :func:`_row`'s
        numbers, ``seconds``) and the Optuna study.
    """
    import optuna
    from optuna.distributions import IntDistribution
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dims = (list(grans), list(inputs), list(outputs))
    keys = ("granularity", "input", "prediction")
    space = {k: IntDistribution(0, len(d) - 1) for k, d in zip(keys, dims, strict=True)}
    ok = {c for c in np.ndindex(*map(len, dims))
          if feasible(*(d[i] for d, i in zip(dims, c, strict=True)), sp,
                      min_windows=min_windows, min_outcomes=min_outcomes)[0]}
    goal = min(n_trials, len(ok))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=max(4, n_trials // 4)))
    rows, seen, rng = [], set(), np.random.default_rng(seed)
    while len(seen) < goal:
        tr = study.ask(space)
        cell = tuple(tr.params[k] for k in keys)
        if cell in seen or cell not in ok:          # measure the nearest unmeasured neighbour
            study.tell(tr, state=TrialState.FAIL)
            free = sorted(ok - seen)
            d = np.array([sum(abs(a - b) for a, b in zip(c, cell, strict=True)) for c in free])
            near = [c for c, x in zip(free, d, strict=True) if x == d.min()]
            study.enqueue_trial(dict(zip(keys, near[rng.integers(len(near))], strict=True)))
            continue
        seen.add(cell)
        t0 = time.time()
        r = run_path(panel, sp, tuple(d[i] for d, i in zip(dims, cell, strict=True)), kind=kind,
                     epochs=epochs, draws=draws, seed=seed, device=device)
        row = {"trial": len(rows) + 1, **_row(r), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        study.tell(tr, float(row["r2"]) if np.isfinite(row["r2"]) else None,
                   state=TrialState.COMPLETE if np.isfinite(row["r2"]) else TrialState.FAIL)
        if verbose:
            print(f"{row['trial']:>3}/{goal}  {row['granularity']:>2}-min bars  input "
                  f"{row['input']:>4}  predict {row['prediction']:>3}  R² {row['r2']:+.4f} "
                  f"(scale {row['beta']:.2f}; direction corr {row['corr']:+.3f}, z "
                  f"{row['z']:+.1f})  {row['seconds']:.0f} s")
    return pd.DataFrame(rows), study


def best(res: pd.DataFrame, objective: str = "r2", k: int = 1) -> pd.DataFrame:
    """The ``k`` best configurations of a search by ``objective``, ties (several scaled to "no
    change", R² exactly 0) broken by direction beyond luck (``z``)."""
    return res.sort_values([objective, "z"], ascending=False, na_position="last").head(k)


def best_per(res: pd.DataFrame, key: str) -> pd.DataFrame:
    """The best configuration of each value of ``key`` (the same ranking as :func:`best`)."""
    return best(res, k=len(res)).drop_duplicates(key).sort_values(key)


def compare_paths(panel, sp: dict, models, cell, *, epochs: int, draws: int = 200,
                  seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Every model of ``models`` on ONE configuration, graded on VALIDATION.

    Returns:
        One row per model (:func:`_row`'s numbers).
    """
    rows = []
    for m in models:
        t0 = time.time()
        r = run_path(panel, sp, cell, kind=m, epochs=epochs, draws=draws, seed=seed)
        rows.append({"model": m, **_row(r)})
        if verbose:
            print(f"{m:>8}: R² {r['score']['r2']:+.4f}  ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows).set_index("model")


def test_paths(panel, sp: dict, configs, *, kind: str, epochs: int, draws: int = 200,
               seed: int = 0) -> pd.DataFrame:
    """The chosen configurations graded once on TEST, each refit on every label before it (TRAIN
    and VALIDATION), beside their VALIDATION grade.

    Args:
        configs: ``[(label, (granularity, input, prediction))]``.

    Returns:
        One row per configuration and part (``config``, ``part``, :func:`_row`'s numbers).
    """
    rows = []
    for label, cell in configs:
        for part in ("validation", "test"):
            r = run_path(panel, sp, cell, kind=kind, epochs=epochs, draws=draws, part=part,
                         seed=seed)
            rows.append({"config": label, "part": part, **_row(r)})
    return pd.DataFrame(rows)


def importance(study) -> pd.Series:
    """How much each hyperparameter moves the grade (Optuna's fANOVA importance over the
    completed trials; empty with too few)."""
    import optuna

    try:
        imp = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError, ZeroDivisionError):
        return pd.Series(dtype=float)
    words = {"granularity": "granularity (bar size)", "input": "input length",
             "prediction": "prediction length"}
    return pd.Series({words.get(k, k): v for k, v in imp.items()}, name="importance")


# ---------------------------------------------------------------------------------------------
# Checks, pictures and tables.
# ---------------------------------------------------------------------------------------------
def path_checks(r: dict, pin: dict, sp: dict) -> list[tuple]:
    """The boundaries of one configuration's result, as built (they document the split; the code
    enforces them, so only :func:`path_truncation` can catch a leak it did not intend): training
    labels end before the held-out sessions, those end before the first graded decision, every
    graded path starts at a decision time and ends inside its part.

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    part = sp[r["part"]]
    H = r["cell"][2]
    t = pin["times"]
    starts, ends = t[r["s"]], t[r["s"] + H]
    f = r["fit"]
    return [
        ("training labels end before the held-out sessions, those before the graded part",
         f["train_to"] < f["held_out"][0] <= f["held_out"][1] < part[0] <= f["first_graded"],
         f"training {f['train_from']} → {f['train_to']}; held out {f['held_out'][0]} → "
         f"{f['held_out'][1]}; first graded decision {f['first_graded']}"),
        ("every graded path starts at a decision time",
         bool(np.isin(starts.strftime("%H:%M"), DECISION_TIMES).all()),
         f"{len(r['s']):,} paths at {', '.join(DECISION_TIMES)}"),
        ("every graded path lies inside its part (no later price is read)",
         bool((starts.strftime("%Y-%m-%d") >= part[0]).all()
              and (ends.strftime("%Y-%m-%d") <= part[1]).all()),
         f"{r['part'].upper()} {part[0]} → {part[1]}; last path ends {ends.max():%Y-%m-%d %H:%M}"),
    ]


def path_truncation(panel, sp: dict, cell, *, kind: str, epochs: int, cut_session: int = 10,
                    seed: int = 0) -> list[tuple]:
    """Nothing reads the future: every 1-minute bar after a cut (the 12:30 decision of the
    ``cut_session``-th VALIDATION session) is replaced by random bars and the WHOLE configuration
    re-run (inputs, fit, scale, predictions). Every path the network predicted at or before the
    cut must be identical (and the scale too), and the later ones must change (else the check
    could not fail).

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    ck = _sibling("checks")
    base = run_path(panel, sp, cell, kind=kind, epochs=epochs, draws=0, seed=seed)
    pin = path_inputs(panel, cell[0])
    day = sp["sessions"][sp["sessions"].index(sp["validation"][0]) + cut_session]
    cut = rs.bucket_at(pin, f"{day} 12:30")
    rows = np.arange(len(panel.close)) >= pin["g"]["end"][cut] + 1
    alt = run_path(ck.scrambled(panel, rows, seed), sp, cell, kind=kind, epochs=epochs, draws=0,
                   seed=seed)
    key = lambda r: {(int(a), int(b)): i for i, (a, b) in enumerate(zip(r["s"], r["j"],  # noqa: E731
                                                                        strict=True))}
    kb, ka = key(base), key(alt)
    early = [k for k in kb if k[0] <= cut and k in ka]
    late = [k for k in kb if k[0] > cut and k in ka]
    # the network's own paths (the scale can be 0, which would hide any change)
    same = all(np.array_equal(base["raw"][kb[k]], alt["raw"][ka[k]]) for k in early)
    moved = any(not np.array_equal(base["raw"][kb[k]], alt["raw"][ka[k]]) for k in late)
    return [("paths predicted before the cut are unchanged",
             bool(early) and same and base["beta"] == alt["beta"],
             f"cut after {day} 12:30; {len(early):,} paths compared, same scale "
             f"{base['beta']:.3f}"),
            ("paths predicted after the cut did change (the check can fail)", moved,
             f"{len(late):,} later paths")]


def example_paths(panel, results, name: str, ts) -> list[dict]:
    """The worked decision's predicted price paths, one per result: the decision's price
    carried along each configuration's predicted bar moves (converted back with each bar's
    typical move as known at the decision: its clock position's size in the decision's session),
    with the times they are about.

    Returns:
        ``[{"label", "times", "price", "price_raw", "granularity", "beta"}]``: the scaled path
        (the graded one) and the network's own, unscaled (configurations without that decision
        are left out).
    """
    out = []
    for label, r in results:
        g, _, H = r["cell"]
        pin = path_inputs(panel, g)
        k = pin["names"].index(name)
        try:
            b = rs.bucket_at(pin, ts)
        except KeyError:
            continue
        hit = np.flatnonzero((r["s"] == b) & (r["j"] == k))
        if not len(hit):
            continue
        g_ = pin["g"]          # each bar's typical size as known at the decision: same clock
        sess0 = g_["sess_first"][g_["code"][b]]      # position, in the decision's own session
        at = np.minimum(sess0 + g_["pos"][b + 1:b + H + 1], len(g_["code"]) - 1)
        u, p0 = np.nan_to_num(pin["u"][at, k]), pin["close"][b, k]
        out.append({"label": label, "granularity": g, "beta": r["beta"],
                    "times": pin["times"][b:b + H + 1],
                    "price": np.r_[p0, p0 * np.exp(np.cumsum(r["pred"][hit[0]] * u))],
                    "price_raw": np.r_[p0, p0 * np.exp(np.cumsum(r["raw"][hit[0]] * u))]})
    return out


def microstructure(panel, grans, *, before: str | None = None) -> pd.DataFrame:
    """Why 1-minute bars need a baseline: per granularity, the lag-1 autocorrelation of bar
    returns inside a session (mean over the stocks, and how many stocks are negative; a negative
    value is mostly the bid-ask bounce and stale prints), and the share of 1-minute bars without a
    trade (stored as a copy of the last price); only sessions before ``before`` (TEST unread)."""
    rows = []
    for g in grans:
        pin = path_inputs(panel, g)
        n = len(pin["g"]["code"]) if before is None else int(np.searchsorted(
            np.asarray(pin["g"]["date"]).astype("datetime64[D]"), np.datetime64(before)))
        y, code = pin["y"][:n], pin["g"]["code"][:n]
        same = code[1:] == code[:-1]
        ac = []
        for k in range(y.shape[1]):
            a, b = y[:-1, k], y[1:, k]
            ok = same & np.isfinite(a) & np.isfinite(b)
            ac.append(np.corrcoef(a[ok], b[ok])[0, 1])
        ac = np.array(ac)
        rows.append({"granularity (min)": g, "lag-1 autocorrelation (mean)": round(ac.mean(), 3),
                     "stocks negative": f"{int((ac < 0).sum())} of {len(ac)}",
                     "R² it alone explains at the next bar": round(float(np.mean(ac ** 2)), 4)})
    out = pd.DataFrame(rows).set_index("granularity (min)")
    reg = SM["regular_rows"](panel)
    if before is not None:
        reg &= np.asarray(panel.close.index.tz_convert(panel.tz).date).astype(
            "datetime64[D]") < np.datetime64(before)
    out.attrs["no-trade minutes %"] = (100 * (panel.volume[reg] == 0).mean()).round(1).to_dict()
    return out


def feature_table(pin: dict, L: int) -> pd.DataFrame:
    """Every input one forecast reads: its group, what it is, when it is known, why it could
    help, and how many bars of it the model reads (the input length for everything bar by bar;
    the calendar also for the predicted bars)."""
    rows = [{"input": "y (the target)", "group": "target",
             "what it is": "the bar's move in typical moves: the path the model continues",
             "known at": "the bar's close", "why it might predict": "momentum, reversal, bounce",
             "bars read": L}]
    for n in pin["past_names"]:
        what, known, why = FEATURES.get(n, (n, "", ""))
        rows.append({"input": n, "group": pin["groups"][n], "what it is": what, "known at": known,
                     "why it might predict": why, "bars read": L})
    rows += [{"input": k, "group": "calendar", "what it is": w, "known at": "in advance",
              "why it might predict": "intraday and weekly patterns",
              "bars read": f"{L} + the predicted bars"} for k, w in CALENDAR.items()]
    rows.append({"input": "stock", "group": "static", "what it is": "which stock (one-hot)",
                 "known at": "always", "why it might predict": "one model, per-stock differences",
                 "bars read": 1})
    return pd.DataFrame(rows).set_index("input")


def results_table(res: pd.DataFrame, objective: str = "r2") -> pd.DataFrame:
    """A search's results, best first, in readable units."""
    x = best(res, objective, k=len(res))
    return pd.DataFrame({
        "granularity (min)": x["granularity"], "input (bars)": x["input"],
        "predict (bars)": x["prediction"], "predicts ahead (min)": x["minutes_ahead"],
        "input span (sessions)": x["input_sessions"].round(1),
        "R² vs no change (scaled)": x["r2"].round(4), "scale β": x["beta"].round(2),
        "stocks R² > 0": (100 * x["share_pos"]).round(0).astype(int).astype(str) + "%",
        "R² unscaled": x["r2_raw"].round(4), "direction corr": x["corr"].round(3),
        "direction z vs luck": x["z"].round(1), "AR(1) R²": x["ar1_r2"].round(4),
        "independent paths": x["outcomes"].round(0), "seconds": x["seconds"]}).reset_index(
        drop=True).rename_axis("rank").rename(index=lambda i: i + 1)


def best_of_bar(k: int, level: float = 0.975) -> float:
    """The z that the BEST of ``k`` tries exceeds by luck alone only ``1 - level`` of the time:
    Φ⁻¹(level^(1/k)) - 1.96 for one try, 2.69 for 7, 3.14 for 30. It counts the tries as
    independent; neighbouring configurations share data, so the true bar is somewhat lower (this
    one is strict)."""
    from scipy.stats import norm

    return float(norm.ppf(level ** (1 / max(int(k), 1))))


def search_words(res: pd.DataFrame) -> str:
    """The search in plain words: the winner, how many configurations beat "no change" and luck,
    whether the winner clears the bar for the best of all the tries, the AR(1) baseline, and the
    best of each granularity."""
    top = best(res).iloc[0]
    bar = best_of_bar(len(res))
    pos, clear = (res["r2"] > 0).sum(), (res["z"] >= 2).sum()
    zero = int((res["beta"] == 0).sum())
    per_g = best_per(res, "granularity")
    lines = [] if top["r2"] > 0 else [
        "- **No configuration's paths agree with the actual ones better than \"no change\"** "
        f"(the best R² is {top['r2']:+.4f}); {zero} of {len(res)} were scaled to \"no change\" "
        "outright by their held-out week."]
    lines += [
        f"- **Winner on VALIDATION:** {top['granularity']:.0f}-minute bars, input "
        f"{top['input']:.0f} bars ({top['input_sessions']:.1f} sessions), predict "
        f"{top['prediction']:.0f} bars ({top['minutes_ahead']:.0f} minutes ahead): R² "
        f"{top['r2']:+.4f} against \"no change\" ({100 * top['share_pos']:.0f}% of the stocks "
        f"above 0, scale {top['beta']:.2f}); direction corr {top['corr']:+.3f}, z {top['z']:+.1f} "
        "against luck.",
        f"- **Better than \"no change\" (R² > 0):** {pos} of {len(res)} configurations; "
        f"**direction above luck (z ≥ 2):** {clear}.",
        f"- **The bar for the best of {len(res)} tries:** z {bar:.1f}. The best direction z "
        f"({res['z'].max():+.1f}) is " + ("**above it**." if res["z"].max() >= bar else
                                          "**below it**: what the best of this many tries looks "
                                          "like by luck alone."),
        f"- **The AR(1) baseline** (one number per stock) scores R² {top['ar1_r2']:+.4f} on the "
        "winner's paths.",
        "- **Best of each granularity:** " + ", ".join(
            f"{r['granularity']:.0f} min: R² {r['r2']:+.4f} (input {r['input']:.0f}, predict "
            f"{r['prediction']:.0f})" for _, r in per_g.iterrows()) + ".",
    ]
    return "\n".join(lines)


def test_words(tc: pd.DataFrame) -> str:
    """The TEST check in plain words: per configuration, VALIDATION against TEST, and whether it
    held (TEST R² above 0 and direction z >= 1.65: one-sided, 5% by luck)."""
    lines = []
    for label, x in tc.groupby("config", sort=False):
        v, t = (x[x["part"] == k].iloc[0] for k in ("validation", "test"))
        held = np.isfinite(t["r2"]) and t["r2"] > 0 and t["z"] >= 1.65
        lines.append(f"- **{label}** ({v['granularity']:.0f} min, input {v['input']:.0f}, predict "
                     f"{v['prediction']:.0f}): VALIDATION R² {v['r2']:+.4f} (direction z "
                     f"{v['z']:+.1f}) → TEST R² {t['r2']:+.4f} (direction z {t['z']:+.1f}): "
                     + ("**held**." if held else "**did not hold**."))
    return "\n".join(lines)


#: The models in one line each (the notebook's §6).
MODEL_WORDS = {
    "tide": "TiDE: a neural network of dense layers (encoder-decoder) that reads the inputs "
            "and the calendar and outputs the whole path",
    "tsmixer": "TSMixer: a neural network that mixes across time steps and across inputs",
    "dlinear": "DLinear: splits the input into trend and remainder and maps each linearly to the "
               "path: the simple baseline that often beats deep models on long inputs",
}


def models_table(models) -> pd.DataFrame:
    """The models compared, in words."""
    return pd.DataFrame({"what it is": [MODEL_WORDS.get(m, m) for m in models]},
                        index=pd.Index(list(models), name="model"))


def as_dots(rows: pd.DataFrame, label) -> pd.DataFrame:
    """Result rows as ``charts.dots`` input: the direction (``corr``) against its luck band, one
    row per ``label`` value (a column name, or labels in row order)."""
    names = rows[label].to_numpy() if isinstance(label, str) else list(label)
    return pd.DataFrame({"forecast": names, "group": "all", "metric": "corr",
                         "value": rows["corr"].to_numpy(),
                         "null_mean": rows["null_mean"].to_numpy(),
                         "null_sd": rows["null_sd"].to_numpy(), "z": rows["z"].to_numpy()})
