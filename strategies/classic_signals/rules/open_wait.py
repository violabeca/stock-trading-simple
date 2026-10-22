"""Wait after the open (a filter): no new trade in the first minutes of the regular session.

The first minutes after 09:30 are the busiest of the day: the night's news gets priced in and
the bands still carry the jump over the night. This filter blocks both sides for every signal
whose trade would FILL in the first ``open_wait_minutes`` minutes after 09:30 (the fill is the
next candle's open, so the rule means the same for every candle size). Trades already open are
not touched: their stop works from the first minute. Same file shape as every rule (see
``band_fade``).
"""

from __future__ import annotations

import pandas as pd

NAME = "wait after the open"
KIND = "filter"
DESCRIPTION = ("No new trade in the first minutes after 09:30: a signal whose trade would fill "
               "in that time is dropped (open trades keep their stop).")
PARAMS = {"open_wait_minutes": {"label": "wait after the open: minutes after 09:30 without new "
                                         "trades", "value": 15, "start": 0, "stop": 120,
                                "step": 5}}
_OPEN = 9 * 60 + 30     # 09:30 New York, in minutes after midnight


def _minutes(params: dict) -> int:
    """The lever: minutes after 09:30 without new trades."""
    return int(params.get("open_wait_minutes", PARAMS["open_wait_minutes"]["value"]))


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    idx = ctx["candles"].index
    fill = pd.Series(idx, index=idx).shift(-1)      # a signal at a close fills at the next open
    after = fill.dt.hour * 60 + fill.dt.minute - _OPEN
    ok = ~(after < _minutes(params)).to_numpy()     # the last candle (no next one) is not blocked
    return pd.DataFrame({"long": ok, "short": ok}, index=idx)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    idx, wait = ctx["candles"].index, _minutes(params)
    if i + 1 >= len(idx):
        return "wait after the open: no next candle to fill on"
    fill = idx[i + 1]
    after = fill.hour * 60 + fill.minute - _OPEN
    since = f"{after} minute{'' if after == 1 else 's'} after the open"
    if after < wait:
        return (f"wait after the open: a trade would fill at {fill:%H:%M}, {since}, inside the "
                f"first {wait}: no new trade")
    return (f"wait after the open: the trade fills at {fill:%H:%M}, {since}, past the first "
            f"{wait}: both sides allowed")
