"""Band breakout (a trigger): follow the price when it breaks out of its Bollinger Bands.

The close stays OUTSIDE a band for a set number of candles in a row (the lever
``breakout_closes``), after a close inside it:

* that many closes in a row above the upper band: propose LONG (strength);
* that many closes in a row below the lower band: propose SHORT (weakness).

One close outside alone may still turn straight back inside, which is the band fade's case, so
the default asks for two. The trigger fires once per stretch, at the close of its n-th candle
(1 = the first close outside); if the filters allow that side, the trade fills at the next
candle's open. Same file shape as every rule (see ``band_fade``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "band breakout"
KIND = "trigger"
DESCRIPTION = ("The close stays outside a band for a set number of candles in a row: long above "
               "the upper band, short below the lower band.")
PARAMS = {"breakout_closes": {"label": "band breakout: closes in a row outside the band",
                              "value": 2, "start": 1, "stop": 10, "step": 1}}


def _closes(params: dict) -> int:
    """The lever: how many closes in a row outside the band the breakout needs."""
    return int(params.get("breakout_closes", PARAMS["breakout_closes"]["value"]))


def _stretch(outside: pd.Series, inside: pd.Series, n: int) -> pd.Series:
    """True at the n-th close in a row ``outside`` when the close before the stretch was
    ``inside``: once per stretch, never again while it lasts."""
    return (outside.astype(int).rolling(n).sum() == n) & inside.shift(n, fill_value=False)


def signal(ctx: dict, params: dict) -> pd.Series:
    """+1 (propose long), -1 (propose short) or 0 for every candle, decided at its close."""
    close, bands, n = ctx["candles"]["close"], ctx["bands"], _closes(params)
    long_ = _stretch(close > bands["upper"], close <= bands["upper"], n)
    short = _stretch(close < bands["lower"], close >= bands["lower"], n)
    return pd.Series(np.select([long_, short], [1, -1], 0), index=close.index).astype("int8")


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    close, bands, n = ctx["candles"]["close"], ctx["bands"], _closes(params)
    if i < n:
        return "band breakout: nothing"
    part = {"candles": ctx["candles"].iloc[i - n:i + 1], "bands": bands.iloc[i - n:i + 1]}
    said = signal(part, params).iloc[-1]     # the stretch and the close before it are enough
    if said == 0:
        return "band breakout: nothing"
    band, word = ("upper", "above") if said == 1 else ("lower", "below")
    t, first, before = close.index[i], close.index[i - n + 1], close.index[i - n]
    side = "long" if said == 1 else "short"
    if n == 1:
        return (f"band breakout, {side}: at {t:%H:%M} the close ({close.iloc[i]:.2f}) broke {word} "
                f"the {band} band ({bands[band].iloc[i]:.2f}); the close before was inside")
    return (f"band breakout, {side}: {n} closes in a row {word} the {band} band, from "
            f"{first:%H:%M} to {t:%H:%M} (the last {close.iloc[i]:.2f}, band "
            f"{bands[band].iloc[i]:.2f}); the close before them, at {before:%H:%M}, was inside")
