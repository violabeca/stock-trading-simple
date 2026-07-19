"""Equal-weight buy-and-hold of the configured universe — the hurdle made visible.

This is the engine's :func:`trading_simple.backtest.benchmark` expressed through
the normal strategy contract: own every symbol in the panel in equal weight from
the first bar to the last, one shared cash account, paying the same costs. It
exists so the hurdle every strategy must clear can be run, inspected and compared
like any other strategy — per universe, per window — instead of living only
inside the objective. ``tests/test_baseline.py`` pins it to ``benchmark()``; its
own ``excess_cagr``/``info_ratio`` are ~0 on its own panel by construction.
"""

from __future__ import annotations

import pandas as pd

from trading_simple.data import Panel


def signals(panel: Panel) -> dict:
    """Constant equal weights: buy 1/N of everything at bar 0 and never trade again.

    Returns the exact frame :func:`trading_simple.backtest.benchmark` builds, so it
    runs through the identical portfolio-mode path (one shared ``init_cash`` book,
    next-bar-open fills, same costs).

    Args:
        panel: Price panel of the universe to hold.

    Returns:
        ``{"weights": <constant 1/N frame>}`` — portfolio mode.
    """
    weights = pd.DataFrame(1.0 / len(panel.close.columns),
                           index=panel.close.index, columns=panel.close.columns)
    return {"weights": weights}
