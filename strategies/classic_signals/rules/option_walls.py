"""Option walls (a filter): no trade straight into a strike where the options are piled up.

Each session's walls come from the previous session's end-of-day chain: the strike above the
price with the most call open interest and the strike below it with the most put open interest
over the next 45 days (``indicators.option_features``), moved into the split-adjusted prices
(``pipeline.walls``). Prices are known to cluster at big strikes near expiry, so a wall can act
like a ceiling or a floor:

* a long is blocked while the nearest wall ABOVE the close is closer than ``wall_room_moves``
  typical moves;
* a short is blocked while the nearest BELOW is closer than that;
* one right at the close blocks both sides.

The same rule as ``major_levels`` on other prices (``ctx["levels"]["walls"]``); the walls can
also be the profit target (*option walls*). No option data: no walls, both sides allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "option walls"
KIND = "filter"
DESCRIPTION = ("No long just under an option wall (a strike with much open interest), no short "
               "just above one: it must leave some room (in typical moves).")
ON = False
PARAMS = {"wall_room_moves": {"label": "option walls: room needed, typical moves",
                              "value": 1.0, "start": 0.5, "stop": 10.0, "step": 0.5}}
_KEY, _KNOB, _WHAT = "walls", "wall_room_moves", "option wall"


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
