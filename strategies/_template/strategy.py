"""TEMPLATE — copy this folder to strategies/<your_name>/ and fill it in.

A strategy is ONE required function (``signals``) plus two optional hooks.
It must be pure pandas: no I/O, no database, no vectorbt import — the engine
handles execution, costs, and metrics identically for every strategy.

Checklist for a new strategy (also see .claude/skills/new-strategy/):
1. Write ``signals()`` below using the helpers in ``trading_simple.backtest``
   (``sma``, ``atr``, ``volume_profile``, ``next_bar``, ``exit_after``).
2. Mirror every parameter in ``config.yaml`` under ``params:`` and
   ``optuna.space:`` (and optionally ``sweep:``).
3. Add ``valid()`` if some parameter combos are impossible.
4. Add a test in ``tests/`` with an engineered signal on synthetic data.
5. Smoke it: ``uv run tsim optimize --strategy <your_name> --n-trials 5``.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(panel: Panel, *, example_window: int = 20, cash_frac: float = 0.25) -> dict:
    """Build execution-timed signal frames from the price panel.

    Args:
        panel: Wide adjusted OHLCV matrices (``close``, ``open_``, ``high``,
            ``low``, ``volume``), all (timestamp x symbol).
        example_window: Replace with your strategy's real parameters — every
            keyword argument here must appear in ``config.yaml``.
        cash_frac: Fraction of the shared wallet's free cash deployed per
            entry. Its presence runs the whole strategy as ONE ``init_cash``
            book (engine cash-shared signal mode) — the repo's default capital
            model, so every strategy trades the same $10k. Drop it only for a
            per-symbol diagnostic run; return a ``weights`` frame instead for
            portfolio mode (cross-sectional rankers).

    Returns:
        A dict with any subset of:
            ``entries`` / ``exits`` / ``short_entries`` / ``short_exits``
                — boolean frames, already shifted to the execution bar
                (use ``bt.next_bar``); fills happen at that bar's open.
            ``sl_stop`` / ``tp_stop`` — stop distance as a fraction of entry
                price (scalar or frame; NaN = no stop).
            ``sl_trail`` — True to make ``sl_stop`` trail the price.
            ``cash_frac`` — see above; an opposite-direction entry closes the
                open position (percent sizing cannot reverse in one order).
    """
    raw = panel.close > bt.sma(panel.close, example_window)  # your idea goes here
    entries = bt.next_bar(raw & ~raw.shift(1, fill_value=False))  # e.g. on crossover only
    return {"entries": entries, "exits": bt.exit_after(entries, 10), "cash_frac": cash_frac}


def valid(params: dict) -> bool:
    """Return False for impossible parameter combos (deleted if not needed)."""
    return params["example_window"] > 0


def indicators(close: pd.Series, volume: pd.Series, *, example_window: int = 20, **_) -> dict:
    """Optional: per-symbol overlay series for the strategy chart (or delete)."""
    return {"price": {f"SMA {example_window}": bt.sma(close, example_window)}, "volume": {}}
