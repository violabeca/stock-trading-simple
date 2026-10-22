"""Put/call (a filter): when the options traded far more puts than usual, only one side.

The day's put volume ÷ call volume, against its usual (the last 20 dates' median), from the
previous session's end-of-day chain (``pipeline.option_context``, ``put/call``, a log ratio).
When it is more than ``pc_limit`` times its usual (many puts), or less than 1 / ``pc_limit``
(many calls), ``pc_side`` decides:

* ``pc_side`` +1, **with the option buyers**: many puts allow shorts only, many calls longs only
  (studies of US stocks found heavy put buying ahead of declines);
* ``pc_side`` -1, **against them**: many puts allow longs only (fear as a contrarian signal).

Measured on NOW, GEV and RGTI (2025-10 to 2026-07): no link with the next session's direction
(rank correlation +0.01 on average). No option data: both allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "put/call"
KIND = "filter"
DESCRIPTION = ("When far more puts (or calls) than usual traded the day before, only one side: "
               "with the option buyers (many puts: shorts only) or against them.")
ON = False
PARAMS = {
    "pc_limit": {"label": "put/call: unusual from, × its usual", "value": 1.5, "start": 1.2,
                 "stop": 4.0, "step": 0.1},
    "pc_side": {"label": "put/call: 1 with the option buyers, -1 against them", "value": 1,
                "start": -1, "stop": 1, "step": 2},
}


def _levers(params: dict) -> tuple[float, int]:
    """The levers: how unusual counts, and which side follows."""
    return (float(params.get("pc_limit", PARAMS["pc_limit"]["value"])),
            int(params.get("pc_side", PARAMS["pc_side"]["value"])))


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    limit, side = _levers(params)
    pc = ctx["options"]["put/call"].to_numpy(float)
    puts, calls = pc > np.log(limit), pc < -np.log(limit)     # NaN: neither
    bearish = puts if side == 1 else calls                   # the side the options lean to
    bullish = calls if side == 1 else puts
    return pd.DataFrame({"long": ~bearish, "short": ~bullish}, index=ctx["options"].index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    limit, side = _levers(params)
    pc = ctx["options"]["put/call"].iloc[i]
    if not np.isfinite(pc):
        return "put/call: no option data for this day, both sides allowed"
    a = allow(ctx, params).iloc[i]
    allowed = ("both sides" if a["long"] and a["short"] else
               "longs only" if a["long"] else "shorts only")
    return (f"put/call: {np.exp(pc):.2f}x the usual put/call volume (unusual from {limit:g}x, "
            f"{'with' if side == 1 else 'against'} the option buyers): {allowed}")
