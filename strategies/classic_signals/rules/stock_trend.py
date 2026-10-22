"""Stock trend (a filter): only trade in the direction of the stock's own daily trend.

The stock's daily trend (the rule of Figure 2b: its close against its trend average, and the
average's rise), judged at the PREVIOUS session's close, so it is known all day:

* up: longs only;
* down: shorts only;
* sideways, or not known yet: both sides.

It is also the "market condition" every result is split by: switched on, it removes the trades
against the stock's own trend. Same file shape as every rule (see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "stock trend"
KIND = "filter"
DESCRIPTION = ("The stock's own daily trend at the previous close: up allows longs only, down "
               "shorts only, sideways both.")
ON = False
PARAMS: dict = {}
_WORDS = {1.0: "up", 0.0: "sideways", -1.0: "down"}


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    state = ctx["trend"]["stock"]
    return pd.DataFrame({"long": (state != -1).to_numpy(), "short": (state != 1).to_numpy()},
                        index=state.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    state = ctx["trend"]["stock"].iloc[i]
    stock = ctx["names"]["stock"]
    if not np.isfinite(state):
        return f"stock trend: {stock}'s trend is not known yet, both sides allowed"
    allowed = {1.0: "longs only", -1.0: "shorts only", 0.0: "both sides"}[float(state)]
    return f"stock trend: {stock} was {_WORDS[float(state)]} at the previous close, {allowed}"
