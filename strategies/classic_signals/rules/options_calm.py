"""Options calm (a filter): no new trade while the options market expects a much bigger move than
the stock has been making: an event ahead (a report, news), when gaps run through stops.

The expected move is the at-the-money straddle of the nearest expiry at least 2 days out, as a
daily move (``indicators.implied_move``), divided by the stock's own recent daily moves (the last
20 sessions, ``indicators.options_ratio``): about 1.1 on an ordinary day, 2 to 5 before a report.
It is read from the previous session's end-of-day chain, so it is known at the open. A day
without option data (the chains start in October 2025) allows both sides. Options say how much a
stock may move, not which way, so the rule blocks both sides alike. Same file shape as every rule
(see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "options calm"
KIND = "filter"
DESCRIPTION = ("No new trade while the options expect a much bigger move than the stock has "
               "been making (an event ahead): their expected daily move more than a set "
               "multiple of the recent daily moves.")
ON = False
PARAMS = {"options_max_ratio": {"label": "options calm: the most the options may expect, × the "
                                         "recent daily moves",
                                "value": 1.8, "start": 1.0, "stop": 3.0, "step": 0.1}}


def _limit(params: dict) -> float:
    """The lever: the largest expected move allowed, as a multiple of the recent daily moves."""
    return float(params.get("options_max_ratio", PARAMS["options_max_ratio"]["value"]))


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    ratio = ctx["options"]["ratio"]
    ok = ~(ratio > _limit(params)).to_numpy()            # no option data (NaN): allowed
    return pd.DataFrame({"long": ok, "short": ok}, index=ratio.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    ratio, limit = ctx["options"]["ratio"].iloc[i], _limit(params)
    if not np.isfinite(ratio):
        return "options calm: no option data for this day, both sides allowed"
    verdict = "no new trade" if ratio > limit else "both sides allowed"
    return (f"options calm: the options expect {ratio:.2f}x the recent daily moves (at most "
            f"{limit:g}): {verdict}")
