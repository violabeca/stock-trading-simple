"""Band fade (a trigger): trade the snap-back after the price stretched beyond a Bollinger Band.

A candle closes OUTSIDE a band, and the next candle closes back INSIDE it:

* back inside from below the lower band: propose LONG (the stretch down is over);
* back inside from above the upper band: propose SHORT (the stretch up is over).

The proposal is made at the close of the candle that came back inside; if the filters allow
that side, the trade fills at the next candle's open. The bands are the shared ones of the
context (length and width set once for every rule).

Every rule file has the same shape: ``NAME``, ``KIND`` ("trigger" proposes trades, "filter"
allows sides), ``DESCRIPTION``, ``PARAMS`` (the knobs the rule owns), a function that answers for
every candle (``signal`` for a trigger, ``allow`` for a filter) and ``explain`` (why, in words).
An optional ``ON = False`` makes the notebook's switch start off (it starts on otherwise).

``PARAMS`` maps each knob's name to its ``label`` (the slider's text), ``value`` (the default)
and ``start`` / ``stop`` / ``step`` (the slider's range, later the optimizer's). The notebook
builds a slider from each and passes every knob to every rule in one dict, so a knob's name
carries its rule's (``breakout_closes``); a rule reads a missing knob as its default. See
``band_breakout`` for one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "band fade"
KIND = "trigger"
DESCRIPTION = ("A close outside a band, then a close back inside: long when it comes back from "
               "below the lower band, short when it comes back from above the upper band.")
PARAMS: dict = {}


def signal(ctx: dict, params: dict) -> pd.Series:
    """+1 (propose long), -1 (propose short) or 0 for every candle, decided at its close."""
    close, bands = ctx["candles"]["close"], ctx["bands"]
    was_below = close.shift(1) < bands["lower"].shift(1)
    was_above = close.shift(1) > bands["upper"].shift(1)
    long_ = was_below & (close >= bands["lower"])
    short = was_above & (close <= bands["upper"])
    return pd.Series(np.select([long_, short], [1, -1], 0), index=close.index).astype("int8")


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    close, bands = ctx["candles"]["close"], ctx["bands"]
    t, p = close.index[i], close.index[i - 1]
    if close.iloc[i - 1] < bands["lower"].iloc[i - 1]:
        return (f"band fade, long: the close at {p:%H:%M} ({close.iloc[i - 1]:.2f}) was below the "
                f"lower band ({bands['lower'].iloc[i - 1]:.2f}), and at {t:%H:%M} it closed back "
                f"inside ({close.iloc[i]:.2f}, lower band {bands['lower'].iloc[i]:.2f})")
    if close.iloc[i - 1] > bands["upper"].iloc[i - 1]:
        return (f"band fade, short: the close at {p:%H:%M} ({close.iloc[i - 1]:.2f}) was above the "
                f"upper band ({bands['upper'].iloc[i - 1]:.2f}), and at {t:%H:%M} it closed back "
                f"inside ({close.iloc[i]:.2f}, upper band {bands['upper'].iloc[i]:.2f})")
    return "band fade: nothing"
