"""Market trend (a filter): only trade in the direction the whole market is going.

The daily trend of the market fund (SPY, the S&P 500; the rule of Figure 2b), judged at the
PREVIOUS session's close, so it is known all day:

* up: longs only;
* down: shorts only;
* sideways, or not known yet: both sides.

Same file shape as every rule (see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "market trend"
KIND = "filter"
DESCRIPTION = ("The market fund's (SPY) daily trend at the previous close: up allows longs only, "
               "down shorts only, sideways both.")
ON = False
PARAMS: dict = {}
_WORDS = {1.0: "up", 0.0: "sideways", -1.0: "down"}


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    state = ctx["trend"]["market"]
    return pd.DataFrame({"long": (state != -1).to_numpy(), "short": (state != 1).to_numpy()},
                        index=state.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    state = ctx["trend"]["market"].iloc[i]
    fund = ctx["names"]["market"]
    if not np.isfinite(state):
        return f"market trend: {fund}'s trend is not known yet, both sides allowed"
    allowed = {1.0: "longs only", -1.0: "shorts only", 0.0: "both sides"}[float(state)]
    return f"market trend: {fund} was {_WORDS[float(state)]} at the previous close, {allowed}"
