"""Dealer gamma (a filter): new trades only while the option dealers' gamma is high enough.

If dealers hold the calls customers sold and the puts customers bought (the usual assumption),
their gamma is + call gamma − put gamma over all open interest (``indicators.option_features``,
from implied volatilities the notebook computes, the chains carry none). Positive, they hedge
against the move (buy dips, sell rallies) and damp it: fades should work; negative, they chase
it. Measured against its own usual size (the last 20 dates' mean absolute value,
``pipeline.option_context``), from the previous session's end-of-day chain: new trades only
while it is at least ``gamma_min`` (0: positive gamma only).

Measured on NOW, GEV and RGTI (2025-10 to 2026-07): no link with the next session's intraday
mean reversion (rank correlation −0.02 on average). No option data: both allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "dealer gamma"
KIND = "filter"
DESCRIPTION = ("New trades only while the option dealers' gamma (over its usual size) is at "
               "least a set level: positive gamma damps moves, which suits fades.")
ON = False
PARAMS = {"gamma_min": {"label": "dealer gamma: at least, × its usual size", "value": 0.0,
                        "start": -1.0, "stop": 1.0, "step": 0.1}}


def _limit(params: dict) -> float:
    """The lever: the lowest dealer gamma (over its usual size) a new trade needs."""
    return float(params.get("gamma_min", PARAMS["gamma_min"]["value"]))


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    gamma = ctx["options"]["gamma"]
    ok = ~(gamma < _limit(params)).to_numpy()            # no option data (NaN): allowed
    return pd.DataFrame({"long": ok, "short": ok}, index=gamma.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    gamma, limit = ctx["options"]["gamma"].iloc[i], _limit(params)
    if not np.isfinite(gamma):
        return "dealer gamma: no option data for this day, both sides allowed"
    verdict = "both sides allowed" if gamma >= limit else "no new trade"
    return (f"dealer gamma: {gamma:+.2f}x its usual size (at least {limit:+g}): {verdict}")
