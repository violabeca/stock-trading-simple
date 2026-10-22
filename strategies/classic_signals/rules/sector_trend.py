"""Sector trend (a filter): only trade in the direction the stock's sector is going.

The sector fund's daily trend (the rule of Figure 2b: its close against its trend average, and
the average's rise), judged at the PREVIOUS session's close, so it is known all day:

* up: longs only;
* down: shorts only;
* sideways, or not known yet: both sides.

The context carries that trend, already read at the previous close, for every candle
(``ctx["trend"]["sector"]``). Same file shape as every rule (see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "sector trend"
KIND = "filter"
DESCRIPTION = ("The sector fund's daily trend at the previous close: up allows longs only, down "
               "shorts only, sideways both.")
PARAMS: dict = {}
_WORDS = {1.0: "up", 0.0: "sideways", -1.0: "down"}


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    state = ctx["trend"]["sector"]
    return pd.DataFrame({"long": (state != -1).to_numpy(), "short": (state != 1).to_numpy()},
                        index=state.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    state = ctx["trend"]["sector"].iloc[i]
    fund = ctx["names"]["sector"]
    if not np.isfinite(state):
        return f"sector trend: {fund}'s trend is not known yet, both sides allowed"
    allowed = {1.0: "longs only", -1.0: "shorts only", 0.0: "both sides"}[float(state)]
    return f"sector trend: {fund} was {_WORDS[float(state)]} at the previous close, {allowed}"
