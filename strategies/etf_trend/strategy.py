"""etf_trend — cross-asset time-series momentum on ETFs (proposal 7, STRATEGY_OVERVIEW.md §6).

Moskowitz, Ooi & Pedersen (2012), Hurst, Ooi & Pedersen (2017), Faber (2007): hold each asset
while its own trailing return beats cash, step aside (or go short) when it does not, and size
every asset by the inverse of its volatility, so a bond and an equity fund carry about the same
risk. The textbook "crisis alpha": in a prolonged bear market equities trend down while
Treasuries, gold or the dollar trend up, and the book follows them.

**The book**, decided once a month (the close of the month's ``offset``-th trading day, 0 =
the first: a new month is known when its first bar prints, while the LAST bar of a month needs
the calendar of the bar after it) and filled at the next open (``shift(1)``); weights are
constant between decisions (the engine trades a name only when its own target changes):

1. **State** of every live risky asset (every column but ``cash``): ``+1`` when its total return
   over ``lookback`` bars (ending ``skip`` bars ago) beats the cash asset's over the same window
   (``hurdle: cash``; 0 where the cash asset is not live, or with ``hurdle: zero``), else ``0``
   (``long_flat``) or ``-1`` (``long_short``). ``blend`` averages the states of several
   lookbacks (a fraction of a position). ``rule: sma`` compares the close with its
   ``lookback``-bar simple average instead (Faber's 10-month rule). ``overlay: SPY`` also
   switches every equity fund off while SPY's own state is off (long_flat only). ``assets``
   restricts the risky universe (``"SPY"`` = the SPY / cash switch).
2. **Size**: ``gross × (1/σ_i) / Σ_j (1/σ_j)`` over EVERY live risky asset, on or off
   (``budget: asset``; ``budget: class`` gives each asset class an equal share and splits it by
   inverse vol inside the class). σ is the EWMA standard deviation of daily returns (centre of
   mass ``vol_com``), so a fully-on book is naive risk parity; an off asset's budget goes to the
   cash asset (the engine pays 0% on idle cash). ``vol_target > 0`` scales the risky book down
   (never up) to that ex-ante annual volatility, from the trailing ``cov_window``-bar covariance.
3. **Band**: a held asset whose state did not change keeps its weight, bit for bit, unless the
   new target is more than ``band`` (relative) away — tiny monthly re-trims pay the real $1
   minimum commission, which the engine's fee FRACTION undercharges. When kept weights push the
   gross above ``gross``, only the assets that trade anyway are scaled down to fit, and a
   target change smaller than ``min_change`` (absolute, the cash asset's included) is not
   traded at all; an exit always trades (:func:`build_weights`).

An asset is live at a decision bar when it has ``min_history`` closes and its whole lookback
window. Shorts (``long_short``) borrow at ``borrow`` a year (liquid ETFs are general collateral);
long + short gross stays <= ``gross`` (the engine locks short proceeds).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

#: Asset class of each ETF the research panel trades (any other column counts as equity).
CLASSES = {
    **dict.fromkeys(["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "XLB", "XLE", "XLF", "XLI",
                     "XLK", "XLP", "XLU", "XLV", "XLY"], "equity"),
    **dict.fromkeys(["TLT", "IEF", "LQD", "HYG"], "bond"),
    **dict.fromkeys(["GLD", "SLV", "DBC"], "commodity"),
    "UUP": "currency",
}
ANNUAL = 252


def signals(panel, *, lookback: int = 252, skip: int = 0, blend: str = "",
            hurdle: str = "cash", vol_com: int = 60, mode: str = "long_flat",
            cash: str = "SHY", gross: float = 0.98, band: float = 0.25,
            min_change: float = 0.0025, offset: int = 0, vol_target: float = 0.0,
            cov_window: int = 126, budget: str = "asset", overlay: str = "",
            borrow: float = 0.005, min_history: int = 252, rule: str = "return",
            assets: str = "", **_) -> dict:
    """Target weights of the monthly trend book (long > 0, short < 0) and the borrow rate.

    Args:
        panel: Daily ETF panel (adjusted = total-return prices).
        lookback: Bars of the trailing return that sets an asset's state.
        skip: Bars skipped at the end of the lookback window (0 = up to the decision bar).
        blend: Comma-separated lookbacks whose states are averaged ("" = ``lookback`` alone).
        hurdle: ``"cash"`` (beat the cash asset's return) or ``"zero"`` (beat 0).
        vol_com: Centre of mass (bars) of the EWMA volatility.
        mode: ``"long_flat"`` or ``"long_short"``.
        cash: Column held with the unused budget and used as the hurdle ("" = idle cash).
        gross: Long plus short exposure, including the cash asset (<= 1: shorts lock cash).
        band: Relative change below which a held asset keeps its weight.
        min_change: Absolute target change (share of the book) below which no order is
            placed: the held weight stays, and a smaller entry is not opened. Exits always
            trade.
        offset: Trading day of the month the book is decided on (0 = the first).
        vol_target: Ex-ante annual volatility the risky book is capped at (0 = off).
        cov_window: Bars of the covariance behind ``vol_target``.
        budget: ``"asset"`` (inverse vol over every asset) or ``"class"`` (equal per class).
        overlay: ``"SPY"`` = equity funds also need SPY's state on ("" = off).
        borrow: Yearly borrow rate charged on shorts.
        min_history: Closes an asset needs before it can be held.
        rule: ``"return"`` (the trailing ``lookback``-bar return) or ``"sma"`` (the close
            against its ``lookback``-bar simple average: Faber's 10-month rule at 210).
        assets: Comma-separated risky columns the book may hold ("" = every column but cash).

    Returns:
        ``{"weights", "borrow"}`` for the engine's portfolio mode.
    """
    parts = compose(panel, lookback=lookback, skip=skip, blend=blend, hurdle=hurdle,
                    vol_com=vol_com, mode=mode, cash=cash, gross=gross, vol_target=vol_target,
                    cov_window=cov_window, budget=budget, overlay=overlay,
                    min_history=min_history, offset=offset, rule=rule, assets=assets)
    weights = build_weights(parts["state"], parts["size"], parts["decision"],
                            panel.close.columns, cash=parts["cash"], gross=gross, band=band,
                            min_change=min_change)
    return {"weights": weights, "borrow": float(borrow)}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def _lookbacks(blend: str, lookback: int) -> list[int]:
    return [int(x) for x in blend.split(",")] if blend else [int(lookback)]


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys = ``signals()``'s defaults)."""
    p = {**_DEFAULTS, **params}
    try:
        looks = _lookbacks(p["blend"], p["lookback"])
    except ValueError:
        return False
    return (all(x >= 1 for x in looks) and p["lookback"] >= 1 and p["skip"] >= 0
            and p["hurdle"] in ("cash", "zero") and p["vol_com"] >= 1
            and p["mode"] in ("long_flat", "long_short") and 0.0 < p["gross"] <= 1.0
            and 0.0 <= p["band"] < 1.0 and 0.0 <= p["min_change"] < 0.1
            and 0 <= p["offset"] <= 15 and p["vol_target"] >= 0.0
            and p["cov_window"] >= 21 and p["budget"] in ("asset", "class")
            and p["overlay"] in ("", "SPY")
            and (p["overlay"] == "" or p["mode"] == "long_flat")
            and p["borrow"] >= 0.0 and p["min_history"] >= 1
            and p["rule"] in ("return", "sma"))


def decision_bars(index: pd.DatetimeIndex, offset: int = 0) -> np.ndarray:
    """True on the ``offset``-th bar (0-based) of every calendar month of ``index``.

    Daily bars are stamped 00:00 UTC of their trade date, so the month is read in UTC. Causal:
    a bar's position inside its month depends only on the bars before it.
    """
    month = pd.Series(index.year * 12 + index.month, index=index)
    return (month.groupby(month.to_numpy()).cumcount() == offset).to_numpy()


def compose(panel, *, lookback: int, skip: int, blend: str, hurdle: str, vol_com: int,
            mode: str, cash: str, gross: float, vol_target: float, cov_window: int,
            budget: str, overlay: str, min_history: int, offset: int, rule: str = "return",
            assets: str = "") -> dict:
    """Everything the book reads at its decision bars (pure; also what research.py inspects).

    Returns:
        ``decision`` (bool array over the panel's bars), ``state`` (risky assets: -1..+1 at
        decision bars, NaN where not live or not a decision bar), ``size`` (the inverse-vol
        budget × the vol-target scale, NaN where not live), ``vol`` (annualised EWMA vol),
        ``cash`` (the cash column or ``None``) and ``risky`` (the risky columns).
    """
    close = panel.close
    cash = cash if cash and cash in close.columns else None
    only = {a.strip() for a in assets.split(",") if a.strip()}
    risky = [c for c in close.columns if c != cash and (not only or c in only)]
    decision = decision_bars(close.index, offset)
    looks = _lookbacks(blend, lookback)
    count = close.notna().cumsum()
    state = pd.DataFrame(0.0, index=close.index, columns=risky)
    live = count[risky].ge(min_history)
    for look in looks:
        end = close.shift(skip)
        start = (close.shift(skip + look) if rule == "return"
                 else end.rolling(look, min_periods=look).mean())
        ret = end / start - 1.0
        bar = pd.Series(0.0, index=close.index)
        if hurdle == "cash" and cash is not None:
            bar = ret[cash].fillna(0.0)
        on = ret[risky].gt(bar, axis=0)
        live &= ret[risky].notna()
        state += on.astype(float) if mode == "long_flat" else np.where(on, 1.0, -1.0)
    state /= len(looks)
    if overlay == "SPY" and "SPY" in risky:
        equity = [c for c in risky if CLASSES.get(c, "equity") == "equity" and c != "SPY"]
        spy_on = state["SPY"].where(live["SPY"], 0.0)
        state[equity] = state[equity].mul(spy_on, axis=0)
    rets = close.pct_change(fill_method=None)
    vol = rets.ewm(com=vol_com, min_periods=min(vol_com, 20)).std() * np.sqrt(ANNUAL)
    live &= vol[risky].gt(0)
    inv = (1.0 / vol[risky]).where(live)
    if budget == "class":
        cls = pd.Series({c: CLASSES.get(c, "equity") for c in risky})
        size = pd.DataFrame(np.nan, index=close.index, columns=risky)
        groups = {g: list(cols) for g, cols in cls.groupby(cls).groups.items()}
        n_live = sum(inv[cols].notna().any(axis=1).astype(int) for cols in groups.values())
        for cols in groups.values():
            size[cols] = inv[cols].div(inv[cols].sum(axis=1), axis=0).div(n_live, axis=0)
    else:
        size = inv.div(inv.sum(axis=1), axis=0)
    size *= gross
    if vol_target > 0:
        size = size.mul(_vol_scale(rets[risky], size * state, decision, vol_target, cov_window),
                        axis=0)
    state = state.where(live)
    state.loc[~decision] = np.nan
    return {"decision": decision, "state": state, "size": size.where(live), "vol": vol,
            "cash": cash, "risky": risky}


def _vol_scale(rets: pd.DataFrame, book: pd.DataFrame, decision: np.ndarray, target: float,
               window: int) -> pd.Series:
    """min(1, target / ex-ante vol of ``book``) at decision bars (1 elsewhere): the trailing
    ``window``-bar sample covariance of daily returns (pairwise; missing = 0), annualised."""
    out = pd.Series(1.0, index=rets.index)
    r = rets.to_numpy(dtype=float)
    x = book.fillna(0.0).to_numpy(dtype=float)
    for t in np.flatnonzero(decision):
        if t < window:
            continue
        sub = pd.DataFrame(r[t - window + 1:t + 1]).cov(min_periods=window // 2).fillna(0.0)
        vol = float(np.sqrt(max(x[t] @ sub.to_numpy() @ x[t], 0.0) * ANNUAL))
        if vol > target:
            out.iloc[t] = target / vol
    return out


def build_weights(state: pd.DataFrame, size: pd.DataFrame, decision: np.ndarray, columns, *,
                  cash: str | None, gross: float, band: float,
                  min_change: float = 0.0) -> pd.DataFrame:
    """Execution-timed target weights from the decision-bar states and sizes.

    At each decision bar the risky target is ``size × state`` (0 where not live). A held
    asset whose sign is unchanged KEEPS its weight, bit for bit, while its new target is within
    ``band`` (relative) or ``min_change`` (absolute) of it, so the engine places no order for
    it; an entry smaller than ``min_change`` is not opened. When the kept weights push the
    gross above ``gross``, only the assets that trade anyway (entries, exits, flips, re-sizes
    beyond the band) are scaled down to fit; one whose scaled target lands inside its band
    keeps its weight too, and an entry scaled below ``min_change`` stays out (repeated until
    nothing changes). Kept weights never move: rescaling every weight instead (before the
    2026-10-23 review) re-trimmed nearly every held short by a hair in ``long_short`` (where
    the targets always fill ``gross``), and the engine's borrow surcharge, a fraction of the
    target change, charged those hairs ~375x the accrual. The fit is always feasible: kept
    weights are a subset of the previous risky book, whose gross was at most ``gross``. The
    cash asset holds ``gross`` minus the risky gross, unless that moves it by less than
    ``min_change`` (it then keeps its weight, so the book is off ``gross`` by less than
    ``min_change``). An exit (a target of 0) always trades. Targets are constant until the
    next decision, then shifted one bar: decided at the close, filled at the next open.
    """
    risky = list(state.columns)
    target = (size * state).fillna(0.0).to_numpy(dtype=float)
    out = np.zeros((len(state), len(risky)))
    cash_w = np.zeros(len(state))
    held = np.zeros(len(risky))
    held_cash = 0.0
    for t in range(len(state)):
        if decision[t]:
            new = target[t]
            same = (np.sign(new) == np.sign(held)) & (held != 0)

            def near(x, held=held, same=same):
                gap = np.abs(x - held)
                return same & ((gap <= band * np.abs(held)) | (gap < min_change))

            keep = near(new)
            new = np.where((held == 0) & (np.abs(new) < min_change), 0.0, new)
            while True:
                kept, free = np.abs(held[keep]).sum(), np.abs(new[~keep]).sum()
                scale = 1.0
                if kept + free > gross * (1 + 1e-12) and free > 0:
                    scale = max(gross - kept, 0.0) / free
                fit = new * scale
                more = ~keep & near(fit)
                tiny = ~keep & (held == 0) & (fit != 0) & (np.abs(fit) < min_change)
                if not (more.any() or tiny.any()):
                    break
                keep |= more
                new = np.where(tiny, 0.0, new)
            held = np.where(keep, held, fit)
            if cash is not None:
                c = max(gross - np.abs(held).sum(), 0.0)
                if c == 0.0 or abs(c - held_cash) >= min_change:
                    held_cash = c
        out[t] = held
        cash_w[t] = held_cash
    frame = pd.DataFrame(out, index=state.index, columns=risky)
    if cash is not None:
        frame[cash] = cash_w
    frame = frame.reindex(columns=columns, fill_value=0.0)
    return frame.shift(1, fill_value=0.0)
