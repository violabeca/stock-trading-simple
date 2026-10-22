"""Results of the classic_signals strategy: summaries by market condition and side, the profit
curve and its worst drawdown, and every trade's reason in plain words.

The market condition of a trade is the stock's own daily trend at the entry (judged at the
previous close): up, sideways or down. The strategy is meant to hold up in all three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONDITIONS = ("up", "sideways", "down")


def summary(trades: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per group: how many trades, the share of winners, the average and median return
    per trade (percent of the position's value, after costs), the total P&L in dollars, the profit
    factor (dollars won / dollars lost) and the average number of sessions held."""
    def row(g: pd.DataFrame) -> pd.Series:
        won, lost = g.loc[g["P&L $"] > 0, "P&L $"].sum(), -g.loc[g["P&L $"] < 0, "P&L $"].sum()
        return pd.Series({"trades": len(g), "winners": (g["P&L $"] > 0).mean(),
                          "average return %": g["return %"].mean(),
                          "median return %": g["return %"].median(),
                          "total P&L $": g["P&L $"].sum(),
                          "profit factor": won / lost if lost > 0 else np.nan,
                          "sessions held (average)": g["sessions"].mean()})
    if trades.empty:
        return pd.DataFrame(columns=[*by, "trades"])
    out = trades.groupby(by, observed=True).apply(row, include_groups=False).reset_index()
    out["trades"] = out["trades"].astype(int)
    order = {"condition": {c: i for i, c in enumerate(CONDITIONS)},    # unknown last
             "stock": {s: i for i, s in enumerate(pd.unique(trades["stock"]))}}
    return out.sort_values(by, key=lambda c: c.map(order[c.name]) if c.name in order else c,
                           kind="stable").reset_index(drop=True)


def equity(trades: pd.DataFrame) -> pd.DataFrame:
    """The profit curve: the cumulative P&L in dollars at every exit, per stock and in total
    (one column each), indexed by exit time."""
    if trades.empty:
        return pd.DataFrame()
    pnl = trades.pivot_table(index="exit time", columns="stock", values="P&L $", aggfunc="sum")
    curve = pnl.fillna(0.0).cumsum()
    curve["all stocks"] = curve.sum(axis=1)
    return curve


def drawdown(curve: pd.Series) -> pd.Series:
    """How far the profit curve is below its best level so far (0 or negative), in dollars."""
    return curve - curve.cummax().clip(lower=0.0)


def triggers(trades: pd.DataFrame, decisions: pd.DataFrame, rules: dict,
             switches: dict) -> pd.Series:
    """The switched-on trigger(s) behind each trade of ONE stock: those that proposed the trade's
    side at its signal candle ("band fade", "band breakout", or both joined by " + ")."""
    names = [n for n, r in rules.items() if r.KIND == "trigger" and switches.get(n)]
    side = np.where(trades["side"] == "long", 1, -1)
    said = decisions.loc[trades["signal time"], names].to_numpy()
    return pd.Series([" + ".join(n for n, v in zip(names, row, strict=True) if v == s)
                      for row, s in zip(said, side, strict=True)], index=trades.index,
                     dtype=object)


def bottom_line(trades: pd.DataFrame, sims: dict) -> pd.DataFrame:
    """The numbers the optimizer will judge, per stock and for all of them together: trades, the
    share of the time in a trade, the total P&L in dollars, the worst drawdown of the profit
    curve (read at every exit) and the score, profit ÷ worst drawdown (NaN when the curve never
    fell below its best).

    Args:
        trades: Every stock's trades.
        sims: Stock -> its ``simulate.run`` (for the time in a trade).
    """
    curve = equity(trades)
    time_in = {s: float((sim["position"] != 0).mean()) for s, sim in sims.items()}
    time_in["all stocks"] = float(np.mean(list(time_in.values()))) if time_in else np.nan
    rows = []
    for name in [*sims, "all stocks"]:
        n = len(trades) if name == "all stocks" else int((trades["stock"] == name).sum())
        total = float(curve[name].iloc[-1]) if name in curve else 0.0
        worst = float(drawdown(curve[name]).min()) if name in curve else 0.0
        rows.append({"stock": name, "trades": n, "in a trade, share of the time": time_in[name],
                     "total P&L $": total, "worst drawdown $": worst,
                     "profit ÷ worst drawdown": total / -worst if worst < 0 else np.nan})
    return pd.DataFrame(rows)


def holding(name: str, candles: pd.DataFrame, *, notional: float, commission: float,
            slippage: float, per_share: float = 0.0, min_slip: float = 0.0) -> dict:
    """The reference: ``notional`` dollars of the stock bought at the first open and sold at the
    last close, with the same costs as a trade (``simulate.run``), as one row of
    :func:`bottom_line`. Its drawdown is read at every session's close."""
    first, last = candles["open"].iloc[0], candles["close"].iloc[-1]
    entry = first + max(first * slippage, min_slip)
    shares = notional / entry
    closes = candles["close"].groupby(candles.index.tz_localize(None).normalize()).last()
    fees = 2 * max(commission, shares * per_share)
    total = shares * (last - max(last * slippage, min_slip)) - notional - fees
    worst = float(drawdown(shares * closes - notional).min())
    return {"stock": f"{name}, just held", "trades": 1, "in a trade, share of the time": 1.0,
            "total P&L $": float(total), "worst drawdown $": worst,
            "profit ÷ worst drawdown": float(total) / -worst if worst < 0 else np.nan}


def _score(trades: pd.DataFrame, stock: str) -> dict:
    """Trades, total P&L, worst drawdown and profit ÷ worst drawdown of one stock's trades (or of
    ``"all stocks"``), on the profit curve of :func:`equity`."""
    t = trades if stock == "all stocks" else trades[trades["stock"] == stock]
    if t.empty:
        return {"trades": 0, "P&L": 0.0, "worst": 0.0, "ratio": np.nan}
    curve = equity(t)["all stocks"]
    total, worst = float(curve.iloc[-1]), float(drawdown(curve).min())
    return {"trades": len(t), "P&L": total, "worst": worst,
            "ratio": total / -worst if worst < 0 else np.nan}


def _on_off(base: pd.DataFrame, other: pd.DataFrame, on: bool) -> tuple:
    """(trades with the rule off, trades with it on), from the run as set and the flipped one."""
    return (other, base) if on else (base, other)


def switch_effects(base: pd.DataFrame, flipped: dict[str, pd.DataFrame], switches: dict,
                   stocks: list[str]) -> pd.DataFrame:
    """What each switch changes: for every rule, the trades and the profit with it switched off
    and on, every other switch as set, per stock and for all stocks together.

    Args:
        base: The trades with every switch as set.
        flipped: Rule name -> the trades with only that rule's switch flipped.
        switches: Rule name -> on / off as set.
        stocks: The stocks, in their order.

    Returns:
        One row per rule and stock (and ``"all stocks"``): whether the rule is on as set, the
        trades, the total P&L $ and profit ÷ worst drawdown with it off and with it on, and the
        change in P&L from switching it on.
    """
    rows = []
    for rule, other in flipped.items():
        off, on = _on_off(base, other, bool(switches.get(rule)))
        for stock in [*stocks, "all stocks"]:
            a, b = _score(off, stock), _score(on, stock)
            rows.append({"rule": rule, "stock": stock,
                         "as set": "on" if switches.get(rule) else "off",
                         "trades off": a["trades"], "trades on": b["trades"],
                         "P&L off $": a["P&L"], "P&L on $": b["P&L"],
                         "change $": b["P&L"] - a["P&L"],
                         "profit ÷ drawdown off": a["ratio"], "profit ÷ drawdown on": b["ratio"]})
    return pd.DataFrame(rows)


def _groups(trades: pd.DataFrame) -> pd.Series:
    """Total P&L by market condition (up, sideways, down) and by half-year of the exit."""
    if trades.empty:
        return pd.Series(dtype=float)
    out = trades["exit time"]
    half = out.dt.year.astype(str) + np.where(out.dt.month <= 6, " H1", " H2")
    return pd.concat([trades.groupby("condition")["P&L $"].sum().reindex(list(CONDITIONS)),
                      trades.groupby(half)["P&L $"].sum().sort_index()])


def effect_breakdown(base: pd.DataFrame, flipped: dict[str, pd.DataFrame], switches: dict,
                     stocks: list[str]) -> pd.DataFrame:
    """Where each switch's change comes from: the change in P&L from switching each rule on, per
    stock, by market condition (up, sideways, down) and by half-year of the exit ("2025 H2").
    A switch that really helps should help in most of them, not only in the total.

    Args:
        base: The trades with every switch as set.
        flipped: Rule name -> the trades with only that rule's switch flipped.
        switches: Rule name -> on / off as set.
        stocks: The stocks, in their order.

    Returns:
        One row per rule, stock and group: ``rule``, ``stock``, ``group``, ``change $`` (groups
        without a trade either way are left out).
    """
    rows = []
    for rule, other in flipped.items():
        off, on = _on_off(base, other, bool(switches.get(rule)))
        for stock in stocks:
            change = _groups(on[on["stock"] == stock]).sub(_groups(off[off["stock"] == stock]),
                                                           fill_value=0.0)
            rows += [{"rule": rule, "stock": stock, "group": g, "change $": float(v)}
                     for g, v in change.items() if np.isfinite(v)]
    return pd.DataFrame(rows, columns=["rule", "stock", "group", "change $"])


def walk_table(walks: dict, control: str = "no tuning") -> pd.DataFrame:
    """The walk forward's bottom line: one row per stock and walk (and for all stocks
    together): trades, time in a trade, total P&L $, worst drawdown $ and profit ÷ worst
    drawdown over the traded sessions, and the P&L against the ``control`` walk.

    Args:
        walks: (stock, walk name) -> ``tuning.walk``.
        control: The walk every other one is compared with.
    """
    rows, pooled = [], {}
    for (stock, name), w in walks.items():
        pooled.setdefault(name, []).append(w)
        sc = _score(w["trades"], "all stocks")
        rows.append({"stock": stock, "walk": name, "trades": sc["trades"],
                     "in a trade, share of the time": float((w["position"] != 0).mean()),
                     "total P&L $": sc["P&L"], "worst drawdown $": sc["worst"],
                     "profit ÷ worst drawdown": sc["ratio"]})
    for name, ws in pooled.items():
        sc = _score(pd.concat([w["trades"] for w in ws], ignore_index=True), "all stocks")
        rows.append({"stock": "all stocks", "walk": name, "trades": sc["trades"],
                     "in a trade, share of the time": float(
                         np.mean([(w["position"] != 0).mean() for w in ws])),
                     "total P&L $": sc["P&L"], "worst drawdown $": sc["worst"],
                     "profit ÷ worst drawdown": sc["ratio"]})
    out = pd.DataFrame(rows)
    if out.empty or control not in set(out["walk"]):
        return out
    base = out[out["walk"] == control].set_index("stock")["total P&L $"]
    out[f"vs {control} $"] = out["total P&L $"] - out["stock"].map(base)
    return out


def explain_trade(trade: pd.Series, ctx: dict, decisions: pd.DataFrame, rules: dict,
                  switches: dict, params: dict, *, exit_: dict) -> str:
    """One trade's story in plain words: which trigger fired and why, what every switched-on
    filter said, the entry, the first stop, the exit and the result.

    Args:
        trade: A row of ``simulate.run(...)["trades"]``.
        ctx: The stock's ``combine.make_context``.
        decisions: Its ``combine.decide``.
        rules: ``combine.load_rules``.
        switches: Rule name -> on / off.
        params: The rules' knobs.
        exit_: The exit settings (``stop_kind``, ``stop_moves``, ``stop_days``).
    """
    i = ctx["candles"].index.get_loc(trade["signal time"])
    side = 1 if trade["side"] == "long" else -1
    fired = [rule.explain(ctx, params, i) for name, rule in rules.items()
             if rule.KIND == "trigger" and switches.get(name) and decisions[name].iloc[i] == side]
    checks = [rule.explain(ctx, params, i) for name, rule in rules.items()
              if rule.KIND == "filter" and switches.get(name)]
    gap = abs(ctx["candles"]["open"].loc[trade["entry time"]] - trade["first stop"])  # the open
    how = {"moves": f"{exit_['stop_moves']:g} typical moves = {gap:.2f} away, trailing",
           "daily trailing": f"{exit_['stop_days']:g} typical daily moves = {gap:.2f} away, "
                             "trailing",
           "daily fixed": f"{exit_['stop_days']:g} typical daily moves = {gap:.2f} away, "
                          "fixed"}[exit_["stop_kind"]]
    goal = float(trade.get("target", np.nan))
    target = f", the profit target {goal:.2f}" if np.isfinite(goal) else ""
    return (f"**{trade['side'].upper()} {trade['stock']}**, signal at the close of "
            f"{trade['signal time']:%Y-%m-%d %H:%M}: {'; '.join(fired) or 'no trigger'}. "
            f"Filters: {'; '.join(checks) if checks else 'none switched on'}. "
            f"Entered at {trade['entry time']:%H:%M} at {trade['entry price']:.2f}; the first "
            f"stop was {trade['first stop']:.2f} ({how}){target}. Exit "
            f"{trade['exit time']:%Y-%m-%d %H:%M} at "
            f"{trade['exit price']:.2f}, after {trade['sessions']} session(s): "
            f"{trade['exit reason']}. Result **{trade['return %']:+.2f}%** "
            f"({trade['P&L $']:+.0f} $ after costs). Market condition at the entry: "
            f"{trade['condition']}.")
