"""IV term (a filter): no new trade while the nearest options are priced much richer than the
later ones: the market prices an event before the later expiry (a report, news).

The at-the-money implied volatility of the nearest expiry at least 2 days out, divided by that of
the expiry nearest 30 days (``indicators.option_features``, ``term``): about 1.05 normally, 1.3
to 2 before a report. Read from the previous session's end-of-day chain. Measured on NOW, GEV
and RGTI (2025-10 to 2026-07): a higher term goes with a bigger next session (range: rank
correlation +0.20 on average), never with its direction. No option data: both allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "iv term"
KIND = "filter"
DESCRIPTION = ("No new trade while the nearest options' implied volatility is more than a set "
               "multiple of the 30-day options' (an event priced ahead).")
ON = False
PARAMS = {"iv_term_max": {"label": "iv term: the most the nearest options' volatility may be, "
                                   "× the 30-day options'",
                          "value": 1.3, "start": 1.0, "stop": 2.5, "step": 0.05}}


def _limit(params: dict) -> float:
    """The lever: the largest front ÷ 30-day volatility allowed."""
    return float(params.get("iv_term_max", PARAMS["iv_term_max"]["value"]))


def allow(ctx: dict, params: dict) -> pd.DataFrame:
    """For every candle, whether a long and whether a short is allowed (``long``, ``short``)."""
    term = ctx["options"]["term"]
    ok = ~(term > _limit(params)).to_numpy()             # no option data (NaN): allowed
    return pd.DataFrame({"long": ok, "short": ok}, index=term.index)


def explain(ctx: dict, params: dict, i: int) -> str:
    """What the rule saw at candle ``i`` (its row number), in words."""
    term, limit = ctx["options"]["term"].iloc[i], _limit(params)
    if not np.isfinite(term):
        return "iv term: no option data for this day, both sides allowed"
    verdict = "no new trade" if term > limit else "both sides allowed"
    return (f"iv term: the nearest options are priced at {term:.2f}x the 30-day options' "
            f"volatility (at most {limit:g}): {verdict}")
