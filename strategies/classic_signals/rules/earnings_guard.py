"""Earnings guard (a filter that also closes trades): no trade held through a quarterly report.

A report can move a stock by 10% or more over a night (NOW fell 14.6% through its April 2026
report). The guard looks at the report's reaction day (the first session that can move on it:
the same day for a report before the open, the next one for a report after the close):

* no new trade in the last ``guard_sessions`` sessions before the reaction day;
* an open trade is closed at the close of the session right before it (``flat``).

Trading on the reaction day itself is allowed again: by its open the report is known. The
context carries, for every candle, the sessions until the next reaction day
(``ctx["report"]``, from ``indicators.sessions_to_event``). Same file shape as every rule (see
``band_fade``), plus ``flat``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "earnings guard"
KIND = "filter"
DESCRIPTION = ("No new trade in the session(s) before a quarterly report moves the stock, and "
               "an open trade is closed at the close before it.")
PARAMS = {"guard_sessions": {"label": "earnings guard: sessions before a report without new "
                                      "trades", "value": 1, "start": 1, "stop": 5, "step": 1}}


def _sessions(params: dict) -> int:
    """The lever: sessions before the reaction day without new trades."""
    return int(params.get("guard_sessions", PARAMS["guard_sessions"]["value"]))


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    until = ctx["report"]
    ok = ~((until >= 1) & (until <= _sessions(params))).to_numpy()
    return pd.DataFrame({"long": ok, "short": ok}, index=until.index)


def flat(ctx: dict, params: dict) -> pd.Series:
    """True at the last candle of the session right before a reaction day: close there."""
    until = ctx["report"]
    idx = until.index
    day = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    last = np.r_[day[1:] != day[:-1], True] if len(idx) else np.zeros(0, bool)
    return pd.Series((until.to_numpy() == 1) & last, index=idx)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    until, stock = ctx["report"].iloc[i], ctx["names"]["stock"]
    if not np.isfinite(until):
        return f"earnings guard: no report of {stock} ahead in the data, both sides allowed"
    if 1 <= until <= _sessions(params):
        return (f"earnings guard: a report moves {stock} in {until:.0f} session(s): no new "
                "trade, and an open one closes at the close before it")
    return (f"earnings guard: the next report moves {stock} in {until:.0f} session(s): both "
            "sides allowed")
