"""Previous high / low (a filter): no trade straight into the previous session's high or low.

The previous session's high and low (known at the open) act like levels: whichever is above the
price is a ceiling, whichever is below it a floor:

* a long is blocked while the nearest of them ABOVE the close is closer than
  ``prev_room_moves`` typical moves;
* a short is blocked while the nearest BELOW is closer than that;
* one right at the close blocks both sides.

The same rule as ``major_levels``, on two other prices. The context carries the nearest of them
for every candle (``ctx["levels"]["previous"]``, from ``indicators.nearest_levels``). Same file
shape as every rule (see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "previous high / low"
KIND = "filter"
DESCRIPTION = ("No long just under the previous session's high or low, no short just above one: "
               "it must leave some room (in typical moves).")
ON = False
PARAMS = {"prev_room_moves": {"label": "previous high / low: room needed, typical moves",
                              "value": 2.0, "start": 0.5, "stop": 10.0, "step": 0.5}}
_KEY, _KNOB, _WHAT = "previous", "prev_room_moves", "previous high or low"


def _room(ctx: dict, params: dict) -> pd.Series:
    """The room a trade needs to the next level, in price."""
    return float(params.get(_KNOB, PARAMS[_KNOB]["value"])) * ctx["move"]


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    close, room = ctx["candles"]["close"], _room(ctx, params)
    near = ctx["levels"].get(_KEY)
    if near is None:
        return pd.DataFrame({"long": True, "short": True}, index=close.index)
    return pd.DataFrame({"long": ~((near["above"] - close) <= room).to_numpy(),
                         "short": ~((close - near["below"]) <= room).to_numpy()},
                        index=close.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    close, move = ctx["candles"]["close"].iloc[i], ctx["move"].iloc[i]
    need = float(params.get(_KNOB, PARAMS[_KNOB]["value"]))
    near = ctx["levels"].get(_KEY)
    words = []
    for side, col, gap_sign, verb in (("long", "above", 1, "no long"),
                                      ("short", "below", -1, "no short")):
        level = np.nan if near is None else near[col].iloc[i]
        if not np.isfinite(level):
            words.append(f"no {_WHAT} {col}: {side}s allowed")
            continue
        away = gap_sign * (level - close) / move if move > 0 else np.nan
        ok = not away <= need
        words.append(f"the nearest {_WHAT} {col} is {level:.2f}, {away:.1f} typical moves away "
                     f"(needs {need:g}): {f'{side}s allowed' if ok else verb}")
    return f"{NAME}: " + "; ".join(words)
