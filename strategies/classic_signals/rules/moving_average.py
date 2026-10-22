"""Moving average (a filter): only trade in the direction of the minute chart's moving average.

The up / sideways / down rule of the daily trends (Figure 2b), on the trading candles, judged at
each candle's close (where the decision is made):

* up: the close is above its moving average (the amber line of Figure 4; its length is set under
  "Moving average") AND the average rose over the last ``ma_rise_candles`` candles by more than
  ``ma_flat_moves`` typical moves: longs only;
* down: the close is below the average AND the average fell by more than that: shorts only;
* sideways (a flat average, or the close and the average disagreeing), or not known yet: both.

The typical move is the trailing stop's yardstick (Figure 4b), so "flat" means the same for a
calm and a jumpy stock. Same file shape as every rule (see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "moving average"
KIND = "filter"
DESCRIPTION = ("The minute chart's moving average: above a rising average allows longs only, "
               "below a falling one shorts only, a flat one both.")
ON = False
PARAMS = {
    "ma_rise_candles": {"label": "moving average: its rise is measured over, candles",
                        "value": 10, "start": 1, "stop": 60, "step": 1},
    "ma_flat_moves": {"label": "moving average: flat zone, ± typical moves",
                      "value": 0.5, "start": 0.0, "stop": 2.0, "step": 0.1},
}
_WORDS = {1.0: "up", 0.0: "sideways", -1.0: "down"}


def _knobs(params: dict) -> tuple[int, float]:
    """The levers: candles the rise is measured over, and the flat zone in typical moves."""
    return (int(params.get("ma_rise_candles", PARAMS["ma_rise_candles"]["value"])),
            float(params.get("ma_flat_moves", PARAMS["ma_flat_moves"]["value"])))


def trend(ctx: dict, params: dict) -> pd.DataFrame:
    """The minute chart's trend at every candle's close: ``state`` (1 up, 0 sideways, -1 down,
    NaN not known yet), ``average``, ``rise`` (its change over the last candles) and ``zone`` (the
    flat zone, ± in price)."""
    close, average = ctx["candles"]["close"], ctx["average"]
    candles, flat = _knobs(params)
    rise = average - average.shift(candles)
    zone = flat * ctx["move"]
    up = (close > average) & (rise > zone)
    down = (close < average) & (rise < -zone)
    state = pd.Series(np.select([up, down], [1.0, -1.0], 0.0), index=close.index)
    state[average.isna() | rise.isna() | zone.isna()] = np.nan
    return pd.DataFrame({"state": state, "average": average, "rise": rise, "zone": zone})


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    state = trend(ctx, params)["state"]
    return pd.DataFrame({"long": (state != -1).to_numpy(), "short": (state != 1).to_numpy()},
                        index=state.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    candles, _ = _knobs(params)
    row = trend(ctx, params).iloc[i]
    close = ctx["candles"]["close"].iloc[i]
    if not np.isfinite(row["state"]):
        return "moving average: not known yet, both sides allowed"
    allowed = {1.0: "longs only", -1.0: "shorts only", 0.0: "both sides"}[float(row["state"])]
    side = "above" if close > row["average"] else "below" if close < row["average"] else "on"
    return (f"moving average: the close {close:.2f} is {side} its average {row['average']:.2f}, "
            f"which moved {row['rise']:+.2f} over {candles} candles (flat zone "
            f"±{row['zone']:.2f}): {_WORDS[float(row['state'])]}, {allowed}")
