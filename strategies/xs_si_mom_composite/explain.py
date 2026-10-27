"""xs_si_mom_composite explained: the numbers behind ``xs_si_mom_composite_explained_marimo.py``.

Loaded by path from that notebook; ``strategy.py`` never imports it, and it changes no rule.
Every frame here is the strategy's own, recomputed step by step so the notebook can show each
step: ``ranks``, ``side_score``, informed_flow_ls_p1's ``_select`` and ``_weights``, and
adaptive_hedge's ``_on_priced_opens``. :func:`walk` asserts that the steps rebuild
``signals()``'s weights exactly. There are no figures here: plots live in the notebook (repo
rule).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

from trading_simple import backtest as bt
from trading_simple import features
from trading_simple.data import load_panel

#: The point-in-time universe of the lab's T1 test (``strategies/proposals_lab/lab.py``).
#: Every six months it holds the 500 equities with the highest median dollar volume over the six
#: months before, with a median price of at least $5, at least 100 sessions and $50M a day.
ROLL = {"top_n": 500, "months": 6, "min_dollar_volume": 50e6, "min_price": 5.0, "min_days": 100}
#: The standard universe since 2026-10-27 (version B, RESEARCH_LOG §13.5): the top 2000, at least
#: $1M a day, the same $5 and 100-session screens.
ROLL_B = {**ROLL, "top_n": 2000, "min_dollar_volume": 1e6}
#: Sector SPDRs that were typed ``equity`` in the shared DB until 2026-10-24 (engine #51).
NOT_STOCKS = frozenset({"XLE", "XLF", "XLI", "XLK", "XLV", "XLY"})
#: The only inputs A_U0 reads (the profitability rank is off in the shipped config).
FEATURES = ["short_interest", "fund_shares"]
CAUSES = {"band": "rank left the band", "max_hold": "held 126 sessions",
          "short_stop": "squeeze stop (+100%)", "crowded": "too crowded to short (>30%)",
          "ineligible": "left the universe / no data", "open": "still open"}

_RANKED_SQL = """
    SELECT p.symbol,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close * p.volume) AS dv,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close) AS price,
           count(*) AS days
    FROM prices_ohlcv p JOIN securities s ON s.symbol = p.symbol
    WHERE p.bar = '1d' AND p.ts >= :a AND p.ts < :b AND s.security_type = 'equity'
    GROUP BY p.symbol
"""


def ranked(engine, formed, months: int = 6) -> pd.DataFrame:
    """Every equity's median dollar volume, median price and session count over the ``months``
    before ``formed`` (sector SPDRs left out).

    Args:
        engine: Database engine.
        formed: Formation date (the first day the universe trades).
        months: Length of the ranking window.

    Returns:
        ``dv``, ``price`` and ``days`` per symbol.
    """
    start = pd.Timestamp(formed) - pd.DateOffset(months=months)
    r = pd.read_sql(text(_RANKED_SQL), engine,
                    params={"a": start, "b": pd.Timestamp(formed)}).set_index("symbol")
    return r[~r.index.isin(NOT_STOCKS)]


def pick(r: pd.DataFrame, roll: dict) -> list[str]:
    """The members a :func:`ranked` table gives under ``roll``'s screens (top ``top_n`` by
    dollar volume)."""
    pool = r[(r["days"] >= roll["min_days"]) & (r["dv"] >= roll["min_dollar_volume"])
             & (r["price"] >= roll["min_price"])]
    return sorted(pool.nlargest(roll["top_n"], "dv").index)


def members(engine, formed, roll: dict = ROLL) -> list[str]:
    """The universe formed on ``formed`` under ``roll`` (:data:`ROLL`: the top 500; :data:`ROLL_B`:
    the top 2000), equities only.

    Args:
        engine: Database engine.
        formed: Formation date (the first day the universe trades).
        roll: The universe's screens.

    Returns:
        The member symbols, sorted.
    """
    return pick(ranked(engine, formed, roll["months"]), roll)


def formations(engine, first, last, roll: dict = ROLL) -> dict[pd.Timestamp, list[str]]:
    """Every universe formed every ``roll["months"]`` months from ``first`` up to ``last``.

    Args:
        engine: Database engine.
        first: The first formation date (the scored window's start).
        last: The last bar of the data.
        roll: The universe's screens (:data:`ROLL` or :data:`ROLL_B`).

    Returns:
        ``{formation date: members}``.
    """
    dates = pd.date_range(first, last, freq=f"{roll['months']}MS")
    return {d: members(engine, d, roll) for d in dates}


def settlements(engine, symbol: str) -> pd.DataFrame:
    """FINRA's days-to-cover rows stored under ``symbol``, dated by settlement AND publication.

    Display only: it shows the reporting lag. The strategy reads ``panel.features``, which is
    dated by publication and screened in ``features.py``.

    Args:
        engine: Database engine.
        symbol: The ticker.

    Returns:
        ``settled`` / ``known`` (UTC dates) and ``dtc``, oldest first.
    """
    r = pd.read_sql(text("SELECT settlement_date, known_at, days_to_cover FROM short_interest "
                         "WHERE symbol = :s ORDER BY settlement_date"), engine,
                    params={"s": symbol})
    return pd.DataFrame({"settled": pd.to_datetime(r["settlement_date"]).dt.tz_localize("UTC"),
                         "known": pd.to_datetime(r["known_at"], utc=True).dt.normalize(),
                         "dtc": r["days_to_cover"]})


def member_mask(index: pd.DatetimeIndex, columns, history: dict) -> pd.DataFrame:
    """1.0 where a name was a member ON that date (the latest formation at or before it), else
    0.0. The first formation also covers the warm-up before it.

    Args:
        index: The panel's index.
        columns: The panel's symbols.
        history: :func:`formations` output.

    Returns:
        A 0/1 frame the strategy reads as its ``member`` feature.
    """
    mask = pd.DataFrame(0.0, index=index, columns=columns)
    first = min(history)
    for formed, names in sorted(history.items()):
        rows = index >= pd.Timestamp(formed, tz="UTC") if formed != first else slice(None)
        mask.loc[rows] = np.asarray(pd.Index(columns).isin(names), dtype=float)
    return mask


def cached(path, build, *, refresh: bool = False):
    """``build()``'s result, kept as a pickle at ``path`` (a DB load takes minutes).

    Args:
        path: The pickle file (git-ignored ``var/``).
        build: A zero-argument function computing the value.
        refresh: Recompute and overwrite the stored value.

    Returns:
        The value.
    """
    import pickle
    from pathlib import Path

    path = Path(path)
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return value


def load(engine, settings, history: dict, *, warmup, end=None, feats=FEATURES):
    """A daily panel of every name that was ever a member, from ``warmup`` (momentum needs
    about two years of history), with A_U0's features and the ``member`` mask.

    Args:
        engine: Database engine.
        settings: Loaded settings (venue overrides).
        history: :func:`formations` output.
        warmup: First bar loaded.
        end: Last bar loaded (``None``: everything stored).
        feats: The alt-data features to attach (``tune.py`` adds the profitability inputs).

    Returns:
        The :class:`~trading_simple.data.Panel`.
    """
    from dataclasses import replace

    names = sorted(set().union(*history.values()))
    parts = [load_panel(engine, names[i:i + 300], "1d", warmup, end,   # one huge query once
                        exchanges=settings.exchanges)                # killed the DB backend
             for i in range(0, len(names), 300)]
    panel = replace(parts[0], rth=None, features={}, **{
        k: pd.concat([getattr(q, k) for q in parts], axis=1).sort_index()
        for k in ("close", "open_", "high", "low", "volume")})
    panel = positive(panel)
    features.attach_features(panel, engine, feats, settings=settings)
    panel.features["member"] = member_mask(panel.close.index, panel.close.columns, history)
    return panel


def positive(panel):
    """The panel with every price at or below zero read as missing (a read-time screen; the
    shared DB is never edited). DEC's stored adjusted close is 0.0 on 732 days from 2021-01 to
    2023-11, a Yahoo adjustment defect, and the engine refuses an order at a price of 0.

    Args:
        panel: A daily panel.

    Returns:
        The same panel with non-positive opens, highs, lows and closes set to NaN.
    """
    from dataclasses import replace

    return replace(panel, **{k: getattr(panel, k).where(getattr(panel, k) > 0)
                             for k in ("open_", "high", "low", "close")})


def walk(strat, panel) -> dict:
    """Every intermediate step of A_U0's ``signals()``, checked to rebuild its weights exactly.

    Args:
        strat: The loaded ``xs_si_mom_composite`` strategy (config as shipped).
        panel: :func:`load` output.

    Returns:
        A dict:

        - ``eligible``;
        - ``dtc`` (FINRA days-to-cover) and ``dtc_avg`` (its average over the strategy's window);
        - ``groups`` (dollar-volume group 1-5);
        - ``si`` and ``mom`` (each signal's percentile rank);
        - ``mom_value`` (the residual momentum score), ``mom_beta``, ``market`` and ``returns``;
        - ``mom_noskip`` / ``mom_value_noskip``: the same rank (month-end rows only) and score
          with the last month counted (``skip=0``), for comparison; the strategy never uses them;
        - ``rebalance`` (month-end rows);
        - ``score`` (the combined rank);
        - ``longs`` / ``shorts`` (held at the deciding close);
        - ``exits`` (one row per closed position with its cause);
        - ``weights`` (target weights at the fill bar);
        - ``si_share``, ``borrow`` and ``beta``.
    """
    q = strat.params
    if (q["long_sleeves"], q["short_sleeves"], q["combine"], q["short_gate"], q["neutral"]) != (
            "si+mom", "si+mom", "rank", "", ""):
        raise ValueError("explain.walk mirrors the shipped A_U0 config only")
    mod = strat.signals.__globals__
    p1, rm, ah = (mod["component"](k) for k in ("p1", "rm", "ah"))
    r = mod["ranks"](panel, {"si", "mom"}, dtc_window=q["dtc_window"],
                     neutral_groups=q["neutral_groups"], short_max_si=q["short_max_si"],
                     si_mid=q["si_mid"], si_high=q["si_high"], borrow_gc=q["borrow_gc"],
                     borrow_mid=q["borrow_mid"], borrow_high=q["borrow_high"],
                     min_price=q["min_price"], neutral=q["neutral"])
    base = r["base"]
    score = mod["side_score"](r["ranks"], ["si", "mom"],
                              {"si": q["w_si"], "mom": q["w_mom"], "prof": q["w_prof"]},
                              base["eligible"], q["score_smooth"])
    longs, shorts, exits = p1._select(score, panel.close, base["shortable"], sides="both",
                                      q_in=q["q_in"], q_out=q["q_out"], slots=q["slots"],
                                      max_hold=q["max_hold"], short_stop=q["short_stop"])
    weights = p1._weights(longs, shorts, base["beta"], gross=q["gross"], long_share=0.5,
                          slots=q["slots"], sizing="equal")
    weights = ah._on_priced_opens(weights, panel.open_)
    shipped = strat.signals(panel, **q)["weights"]
    if not np.allclose(weights.to_numpy(), shipped.to_numpy(), equal_nan=True):
        raise AssertionError("the explained steps no longer rebuild signals()' weights")
    mom = rm.compose(panel, min_price=q["min_price"], factors="market")
    mom0 = rm.compose(panel, min_price=q["min_price"], factors="market", skip=0)
    dtc = panel.features["short_interest"].reindex_like(panel.close)
    exits = pd.DataFrame(exits, columns=["side", "symbol", "opened", "closed", "cause"])
    exits["side"] = exits["side"].map({1: "long", -1: "short"})
    return {"eligible": base["eligible"], "dtc": dtc,
            "dtc_avg": dtc.rolling(q["dtc_window"], min_periods=1).mean().where(dtc.notna()),
            "groups": p1._liquidity_groups(panel, q["neutral_groups"]),
            "si": r["ranks"]["si"], "mom": r["ranks"]["mom"], "mom_value": mom["value"],
            "mom_noskip": mom0["rank"], "mom_value_noskip": mom0["value"],
            "mom_beta": mom["beta"], "market": mom["market"], "returns": mom["returns"],
            "rebalance": mom["rebalance"], "score": score, "longs": longs, "shorts": shorts,
            "exits": exits, "weights": weights, "si_share": base["si_share"],
            "borrow": base["borrow"], "beta": base["beta"]}


def skip_test(w: dict, start) -> pd.DataFrame:
    """Does skipping the last month help? Month-end rank ICs against the next month's residual.

    At each month-end from ``start`` that has a full next month, three rankings of the names
    with both momentum ranks are compared with the next month's return beyond β × the market
    (Spearman correlation, the IC): the shipped rank (last month skipped), the same rank with
    the last month counted, and the last month's residual return alone.

    Args:
        w: :func:`walk` output.
        start: First month-end read (the point-in-time universe starts there).

    Returns:
        One row per ranking: average IC, its t-statistic and the months with IC > 0.
    """
    r, m, idx = w["returns"], w["market"], w["returns"].index
    ends = w["rebalance"][w["rebalance"]].index
    ics = []
    for t, nxt in zip(ends[:-1], ends[1:], strict=True):
        if t < pd.Timestamp(start, tz="UTC"):
            continue
        i, j = idx.get_loc(t), idx.get_loc(nxt)
        beta = w["mom_beta"].loc[t]
        nxt_ret = r.iloc[i + 1:j + 1].sum(min_count=1) - beta * m.iloc[i + 1:j + 1].sum()
        last = r.iloc[i - 20:i + 1].sum(min_count=1) - beta * m.iloc[i - 20:i + 1].sum()
        ok = w["mom"].loc[t].notna() & w["mom_noskip"].loc[t].notna() & nxt_ret.notna()
        y = nxt_ret[ok].rank()
        ics.append([w["mom"].loc[t][ok].rank().corr(y),
                    w["mom_noskip"].loc[t][ok].rank().corr(y), last[ok].rank().corr(y)])
    ics = pd.DataFrame(ics, columns=["last month skipped (what it trades)",
                                     "last month counted", "last month alone"])
    return pd.DataFrame({"average IC": ics.mean().round(3),
                         "t-statistic": (ics.mean() / ics.std() * np.sqrt(len(ics))).round(2),
                         "months with IC > 0": [f"{int((ics[c] > 0).sum())} of {len(ics)}"
                                                for c in ics]}).rename_axis("ranking")


def assessment(w: dict, day) -> pd.DataFrame:
    """The stock-picking assessment at one deciding close: every eligible name ranked.

    Args:
        w: :func:`walk` output.
        day: The deciding close (the last bar on or before it is used).

    Returns:
        One row per eligible name, best combined rank first, with:

        - its two ranks, days-to-cover, momentum score and short-interest share;
        - ``book``, what the strategy holds after this close: long, short or none;
        - ``action``, what trades at the next open: buy, sell short, sell, cover or hold.
    """
    idx = w["score"].index
    day = pd.Timestamp(day)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    t = int(idx.searchsorted(day, side="right")) - 1
    t = max(t, 1)
    at = idx[t]
    ok = w["eligible"].iloc[t].astype(bool)
    names = ok.index[ok]
    held = np.where(w["longs"].iloc[t], "long", np.where(w["shorts"].iloc[t], "short", ""))
    before = np.where(w["longs"].iloc[t - 1], "long",
                      np.where(w["shorts"].iloc[t - 1], "short", ""))
    book = pd.Series(held, index=w["score"].columns)
    prev = pd.Series(before, index=w["score"].columns)
    action = pd.Series("", index=book.index)
    action[(book == "long") & (prev != "long")] = "buy (new long)"
    action[(book == "short") & (prev != "short")] = "sell short (new short)"
    action[(prev == "long") & (book != "long")] = "sell (long exit)"
    action[(prev == "short") & (book != "short")] = "buy back (cover)"
    action[(book != "") & (book == prev)] = "hold"
    out = pd.DataFrame({
        "combined rank": w["score"].iloc[t], "short-interest rank": w["si"].iloc[t],
        "momentum rank": w["mom"].iloc[t],
        "days-to-cover (avg)": w["dtc_avg"].iloc[t], "momentum score": w["mom_value"].iloc[t],
        "liquidity group": w["groups"].iloc[t], "short interest / shares": w["si_share"].iloc[t],
        "book": book, "action at next open": action}).loc[names]
    out.attrs["day"] = at
    exited = (prev != "") & (book == "") & ~prev.index.isin(names)
    if exited.any():                                   # exits of names no longer eligible
        gone = pd.DataFrame({"book": "", "action at next open": action[exited]},
                            index=prev.index[exited])
        out = pd.concat([out, gone])
    return out.sort_values("combined rank", ascending=False, na_position="last")


def backtest(strat, panel, settings, start, account: float) -> dict:
    """The book from ``start`` on (signals warm from the panel's first bar), on the engine.

    Args:
        strat: The loaded strategy.
        panel: :func:`load` output.
        settings: Loaded settings (costs).
        start: First traded bar.
        account: Starting cash in dollars. The book is scale-free with fractional shares; the
            strategy's ``order_notional`` ($1,000 a slot) prices the $1 minimum commission.

    Returns:
        A dict:

        - ``pf``, the portfolio, and ``window``, the scored panel;
        - ``returns`` and ``bench``: daily returns of the book and of the equal-weight
          buy-and-hold of every name in the panel, both counting day 0 from the cash;
        - ``spy``: SPY's daily returns on the same days;
        - ``fees`` and ``slippage``;
        - ``warnings``: the engine's messages, e.g. orders it could not fill.
    """
    s = settings.model_copy(update={"backtest": settings.backtest.model_copy(
        update={"init_cash": float(account)})})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pf, window, fees, slip = bt._run_scored(strat, panel, None, s, start)
    value = bt.equity_curve(pf)["value"]
    returns = value / value.shift(1).fillna(float(account)) - 1.0
    total, bench = bt.benchmark(window, fees=fees, slippage=slip, fill=strat.fill)
    bench = bench.reindex(returns.index).fillna(0.0).copy()
    day0 = (1.0 + total) / float(np.prod(1.0 + bench.to_numpy())) - 1.0
    bench.iloc[0] = (1.0 + bench.iloc[0]) * (1.0 + day0) - 1.0
    return {"pf": pf, "window": window, "returns": returns, "bench": bench,
            "fees": fees, "slippage": slip, "account": float(account),
            "warnings": [str(w.message) for w in caught]}


def member_benchmark(window, history: dict) -> pd.Series:
    """Equal-weight buy-and-hold of the CURRENT universe: bought at each formation, held to the
    next one.

    ``backtest.benchmark`` holds every name of the panel from the first bar. Here the panel is
    every name that was EVER a member, including stocks that joined in 2026 because they had
    risen, so that benchmark would carry look-ahead. This one is point in time.
    - Each formation's members are bought equally at their first close in the period, or at
      the first bar's open in the first period.
    - The book then drifts until the next formation.
    - No costs are charged.

    Args:
        window: The scored panel (:func:`backtest` ``window``).
        history: :func:`formations` output.

    Returns:
        Daily returns on ``window``'s index.
    """
    close = window.close.ffill()
    index = close.index
    out = pd.Series(0.0, index=index)
    starts = sorted(pd.Timestamp(f, tz="UTC") for f in history)
    for k, formed in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else index[-1] + pd.Timedelta(days=1)
        rows = (index >= formed) & (index < end)
        if not rows.any():
            continue
        names = [c for c in history[formed.tz_localize(None)] if c in close.columns]
        px = close.loc[rows, names]
        first = window.open_.loc[rows, names].iloc[0] if k == 0 else close[names].shift(1).loc[
            rows].iloc[0]
        held = px.columns[first.notna() & px.iloc[0].notna()]
        value = (px[held] / first[held]).mean(axis=1)           # $1 split equally at the start
        out.loc[rows] = value / value.shift(1).fillna(1.0) - 1.0
    return out


def spy_returns(engine, settings, index: pd.DatetimeIndex) -> pd.Series:
    """SPY's daily close-to-close returns on ``index`` (the first day from its open, as a
    market-on-open buy would see it)."""
    p = load_panel(engine, ["SPY"], "1d", index[0] - pd.Timedelta(days=10), index[-1],
                   exchanges=settings.exchanges)
    close, open_ = p.close["SPY"], p.open_["SPY"]
    r = close.pct_change().reindex(index).fillna(0.0)
    r.iloc[0] = close.reindex(index).iloc[0] / open_.reindex(index).iloc[0] - 1.0
    return r


def summary(returns: pd.Series, bench: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """The headline numbers of the book against buy-and-hold and SPY.

    Args:
        returns: The book's daily returns.
        bench: Equal-weight buy-and-hold's daily returns.
        spy: SPY's daily returns.

    Returns:
        One row per series: total return, CAGR, volatility, Sharpe, max drawdown, beta and
        alpha against equal-weight buy-and-hold.
    """
    years = bt._years(returns.index)
    var = float(bench.var())
    rows = {}
    for name, r in (("the strategy", returns), ("equal-weight buy-and-hold", bench),
                    ("SPY", spy)):
        curve = (1.0 + r).cumprod()
        total = float(curve.iloc[-1] - 1.0)
        cagr = bt._cagr(total, years)
        beta = float(r.cov(bench) / var) if var else float("nan")
        bench_cagr = bt._cagr(float((1 + bench).prod() - 1), years)
        rows[name] = {"total return": total, "CAGR": cagr,
                      "volatility": float(r.std() * np.sqrt(252)),
                      "Sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() else np.nan,
                      "max drawdown": float((curve / curve.cummax() - 1.0).min()),
                      "beta vs equal-weight": beta, "alpha a year": cagr - beta * bench_cagr}
    return pd.DataFrame(rows).T


def legs(bt_out: dict) -> pd.DataFrame:
    """Daily profit and loss split into the long side, the short side and costs.

    Each name earns its previous close's shares from that close to today's open, then
    today's shares from the open to the close (the fills happen at the open). Costs are the
    remainder against the book's value change: commission, spread and slippage, and borrow.

    Args:
        bt_out: :func:`backtest` output.

    Returns:
        Columns ``long``, ``short``, ``costs`` (negative) and ``total``, in dollars per day.
    """
    pf, w = bt_out["pf"], bt_out["window"]
    a = pf.assets().reindex(columns=w.close.columns).fillna(0.0)
    prev = a.shift(1).fillna(0.0)
    o, c = w.open_.ffill(), w.close.ffill()
    pnl = (prev * (o - c.shift(1))).fillna(0.0) + (a * (c - o)).fillna(0.0)
    side = np.sign(a.where(a != 0, prev))
    value = bt.equity_curve(pf)["value"]
    change = value.diff()
    change.iloc[0] = value.iloc[0] - bt_out["account"]
    gross = pnl.sum(axis=1)
    return pd.DataFrame({"long": pnl.where(side > 0, 0.0).sum(axis=1),
                         "short": pnl.where(side < 0, 0.0).sum(axis=1),
                         "costs": change - gross, "total": change})


def positions(bt_out: dict, w: dict) -> pd.DataFrame:
    """Every position the book opened, with its decision dates, fills, P&L and exit cause.

    Args:
        bt_out: :func:`backtest` output.
        w: :func:`walk` output.

    Returns:
        One row per position, sorted by entry:

        - ``symbol`` and ``side``;
        - ``decided`` (the deciding close) and ``entry`` (the fill at the next open);
        - ``exit decided`` and ``exit``;
        - ``sessions`` held and ``status``;
        - ``entry price`` and ``exit price`` (adjusted, slippage included);
        - ``shares``;
        - ``P&L $``, with fees and borrow;
        - ``return``, on the position's entry value;
        - ``cause``, why it closed;
        - the combined, short-interest and momentum ranks at entry and at exit.
    """
    pf, window = bt_out["pf"], bt_out["window"]
    recs = pf.positions.records_readable
    index = window.close.index
    start = index[0]
    priced = window.open_.notna()
    ex = w["exits"]

    def fill_after(sym: str, decided) -> pd.Timestamp | None:
        """The first priced open after a deciding close (the strategy's fill rule)."""
        later = priced.index[(priced.index > decided) & priced[sym].to_numpy()]
        return later[0] if len(later) else None

    rows = []
    for _, p in recs.iterrows():
        sym, side = p["Column"], p["Direction"].lower()
        entry, closed = p["Entry Timestamp"], p["Status"] == "Closed"
        exit_ts = p["Exit Timestamp"] if closed else None
        mine = ex[(ex["symbol"] == sym) & (ex["side"] == side)]
        cause, decided_exit, decided = "open", None, None
        if closed:
            exact = [e for _, e in mine.iterrows() if fill_after(sym, e["closed"]) == exit_ts]
            late = mine[(mine["closed"] < exit_ts) & (mine["closed"] >= entry)]
            e = exact[0] if exact else (late.iloc[-1] if len(late) else None)
            if e is not None:          # (a late match: the exit order filled bars later)
                cause, decided_exit, decided = e["cause"], e["closed"], e["opened"]
        if decided is None:                            # the entry's deciding close
            held = (w["longs"] if side == "long" else w["shorts"])[sym]
            before = held.loc[:entry].iloc[:-1]
            off = before[~before]
            later = held.index[held.index > off.index[-1]] if len(off) else held.index
            decided = later[0]
        sessions = int(((index >= entry) & (index <= (exit_ts or index[-1]))).sum())

        def at(frame, when, sym=sym):
            return float(frame[sym].loc[:when].iloc[-1]) if when is not None else np.nan

        value = float(p["Size"]) * float(p["Avg Entry Price"])
        rows.append({
            "symbol": sym, "side": side,
            "decided": decided if decided >= start else pd.NaT, "entry": entry,
            "exit decided": decided_exit, "exit": exit_ts, "sessions": sessions,
            "status": "closed" if closed else "open",
            "entry price": float(p["Avg Entry Price"]),
            "exit price": float(p["Avg Exit Price"]) if closed else float(
                window.close[sym].ffill().iloc[-1]),
            "shares": float(p["Size"]), "P&L $": float(p["PnL"]),
            "return": float(p["PnL"]) / value if value else np.nan,
            "cause": CAUSES.get(cause, cause),
            "rank at entry": at(w["score"], decided), "SI rank at entry": at(w["si"], decided),
            "momentum rank at entry": at(w["mom"], decided),
            "rank at exit": at(w["score"], decided_exit),
            "held from before the start": decided < start})
    return pd.DataFrame(rows).sort_values(["entry", "symbol"]).reset_index(drop=True)
