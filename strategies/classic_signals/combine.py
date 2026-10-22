"""The switchboard of the classic_signals strategy: which rules are switched on, and what they
decide together, candle by candle.

1. Every switched-on TRIGGER proposes long (+1), short (-1) or nothing (0) at a candle's close.
   If two triggers propose opposite sides on the same candle, the signal is not clear: no trade.
2. Every switched-on FILTER says whether a long and whether a short is allowed. A side needs ALL
   of them, so switching a filter on can only remove trades, never add one.
3. The decision: the proposed side, if it is allowed.

The rules live in ``rules/``, one small file each, all with the same shape (see
``rules/band_fade.py``); :data:`RULE_FILES` lists them. Adding a rule: one new file in ``rules/``
and its name here. The notebook shows every rule as a switch (starting in the position its
``ON`` says, on when it says nothing), and every knob in its ``PARAMS`` as a slider under it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RULE_FILES = ("band_fade", "band_breakout", "moving_average", "stock_trend", "sector_trend",
              "market_trend", "major_levels", "previous_range", "open_wait", "earnings_guard",
              "options_calm", "iv_term", "put_call", "dealer_gamma", "option_walls")
OPTIONS = ("ratio", "term", "put/call", "gamma")      # what the option rules read (pipeline)
KINDS = ("trigger", "filter")


def load_rules() -> dict:
    """Every rule of :data:`RULE_FILES`, loaded by path: ``{rule name: module}``, in that order."""
    out = {}
    for name in RULE_FILES:
        spec = importlib.util.spec_from_file_location(f"classic_signals_rule_{name}",
                                                      HERE / "rules" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.KIND not in KINDS:
            raise ValueError(f"rules/{name}.py: KIND {module.KIND!r}, expected one of {KINDS}")
        out[module.NAME] = module
    return out


def make_context(names: dict, candles: pd.DataFrame, bands: pd.DataFrame, move: pd.Series,
                 trends: dict[str, pd.Series], average: pd.Series | None = None,
                 levels: dict[str, pd.DataFrame] | None = None,
                 report: pd.Series | None = None, daily_move: pd.Series | None = None,
                 options: pd.DataFrame | None = None) -> dict:
    """What the rules read for one stock, all on its regular-session candles.

    Args:
        names: ``{"stock", "sector", "market"}`` symbols.
        candles: The stock's regular-session candles (the size the strategy trades).
        bands: ``indicators.bollinger`` (any index: aligned here).
        move: The typical move in PRICE units known at each candle (the trailing stop's yardstick).
        trends: Role (``"stock"``, ``"sector"``, ``"market"``) -> daily trend state as known at each
            candle, i.e. judged at the previous session's close (``indicators.previous_session``).
        average: The minute chart's moving average (any index: aligned here; none = not known).
        levels: Kind (``"major"``, ``"previous"``) -> the nearest level ``above`` and ``below``
            each candle's close (``indicators.nearest_levels``); a missing kind blocks nothing.
        report: Sessions from each candle's session to the next report's reaction day (0 on that
            day; ``indicators.sessions_to_event``); none = no report known.
        daily_move: The stock's typical daily move in PRICE units, as known at each candle (the
            yardstick of the daily-move stops and target); none = not known.
        options: What the option rules read (:data:`OPTIONS`), as known at each candle
            (``pipeline.option_context``); none = no option data (NaN: the rules allow both).
    """
    idx = candles.index
    unknown = pd.Series(np.nan, index=idx)
    return {"names": dict(names), "candles": candles, "bands": bands.reindex(idx),
            "move": move.reindex(idx), "trend": {k: v.reindex(idx) for k, v in trends.items()},
            "average": unknown if average is None else average.reindex(idx),
            "levels": {k: v.reindex(idx) for k, v in (levels or {}).items()},
            "report": unknown if report is None else report.reindex(idx),
            "daily_move": unknown if daily_move is None else daily_move.reindex(idx),
            "options": (pd.DataFrame(np.nan, index=idx, columns=list(OPTIONS)) if options is None
                        else options.reindex(index=idx, columns=list(OPTIONS)))}


def decide(ctx: dict, rules: dict, switches: dict, params: dict | None = None, *,
           only_on: bool = False) -> pd.DataFrame:
    """Every rule's answer and the decision, candle by candle.

    Rules that are switched off still answer (so a chart can show what they would have said),
    but they take no part in the decision; ``only_on`` skips them (faster, for the tuner). A
    switched-on rule with a ``flat`` function can also ask for an open trade to be closed at a
    candle's close (the earnings guard, before a report).

    Args:
        ctx: :func:`make_context`.
        rules: :func:`load_rules`.
        switches: Rule name -> on (True) / off.
        params: The rules' knobs, by ``PARAMS`` name (a missing one is its default).

    Returns:
        One row per candle: per trigger its proposal (+1, -1, 0); per filter ``"<name>: long"``
        and ``"<name>: short"`` (allowed or not); ``proposal`` (from the switched-on triggers),
        ``conflict`` (two of them disagreed), ``allow long`` / ``allow short`` (every
        switched-on filter agrees), ``signal`` (+1 long, -1 short, 0 nothing: the proposal if
        its side is allowed), ``blocked`` (a proposal that a filter stopped) and ``flat``
        (close any open trade at this candle's close).
    """
    params = params or {}
    idx = ctx["candles"].index
    n = len(idx)
    out = pd.DataFrame(index=idx)
    up, down = np.zeros(n, bool), np.zeros(n, bool)
    allow_long, allow_short, flat = np.ones(n, bool), np.ones(n, bool), np.zeros(n, bool)
    for name, rule in rules.items():
        on = bool(switches.get(name, False))
        if only_on and not on:
            continue
        if on and hasattr(rule, "flat"):
            flat |= rule.flat(ctx, params).reindex(idx).fillna(False).to_numpy(bool)
        if rule.KIND == "trigger":
            s = rule.signal(ctx, params).reindex(idx).fillna(0).astype("int8").to_numpy()
            out[name] = s
            if on:
                up |= s == 1
                down |= s == -1
        else:
            a = rule.allow(ctx, params).reindex(idx)
            long_ok = a["long"].fillna(True).to_numpy(bool)
            short_ok = a["short"].fillna(True).to_numpy(bool)
            out[f"{name}: long"], out[f"{name}: short"] = long_ok, short_ok
            if on:
                allow_long &= long_ok
                allow_short &= short_ok
    conflict = up & down
    proposal = np.where(conflict, 0, up.astype(int) - down.astype(int))
    signal = np.where((proposal == 1) & allow_long, 1,
                      np.where((proposal == -1) & allow_short, -1, 0))
    out["proposal"], out["conflict"] = proposal.astype("int8"), conflict
    out["allow long"], out["allow short"] = allow_long, allow_short
    out["signal"] = signal.astype("int8")
    out["blocked"] = (proposal != 0) & (signal == 0)
    out["flat"] = flat
    return out
