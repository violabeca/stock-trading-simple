"""Donchian breakout: enter on a channel breakout, protect with a tight ATR trailing stop.

Thesis (the classic "Turtle" system): when price closes above its highest high
of the last ``entry_window`` bars, the odds of short-term continuation are a
little better than a coin flip (~52% in published breakout studies). That small
directional edge only pays if losses stay small — so every position carries a
**trailing stop** at ``atr_mult`` ATRs from the best price since entry. The
tight trail serves double duty: it caps the initial loss AND locks in profit
as the move extends. Expect many small stop-outs and a few larger winners.

Signals:

* ``upper`` / ``lower`` — rolling max(high) / min(low) over ``entry_window``
  bars, shifted one bar so a bar can't break out of a channel it defines.
* Long when close > upper; short when close < lower (close-confirmed, which
  filters intrabar noise). Executed at the next bar's open.
* Exits are stop-driven: the engine passes ``sl_stop`` (an ATR fraction of
  price, per entry) with ``sl_trail`` on to vectorbt, which evaluates it
  intrabar against high/low. An opposite breakout reverses the position.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(
    panel: Panel,
    *,
    entry_window: int = 20,
    atr_window: int = 14,
    atr_mult: float = 2.0,
    direction: str = "both",
) -> dict:
    """Build breakout entries and ATR trailing-stop parameters.

    Args:
        panel: Price panel (uses high/low/close).
        entry_window: Channel lookback in bars (breakout = new N-bar extreme).
        atr_window: ATR smoothing window in bars.
        atr_mult: Trailing-stop distance in ATRs (lower = tighter stop,
            smaller losses, more frequent stop-outs).
        direction: ``"long"`` | ``"short"`` | ``"both"``.

    Returns:
        Dict with ``entries``, ``short_entries``, ``sl_stop`` (ATR fraction of
        price at entry) and ``sl_trail=True``. No time-based exits: positions
        leave via the trailing stop or an opposite breakout (reversal).
    """
    upper = panel.high.rolling(entry_window, min_periods=entry_window).max().shift(1)
    lower = panel.low.rolling(entry_window, min_periods=entry_window).min().shift(1)
    none = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)

    return {
        "entries": bt.next_bar(panel.close > upper) if direction != "short" else none,
        "short_entries": bt.next_bar(panel.close < lower) if direction != "long" else none,
        "sl_stop": (atr_mult * bt.atr(panel, atr_window) / panel.close).clip(lower=1e-4),
        "sl_trail": True,
    }


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    entry_window: int = 20,
    **_,
) -> dict:
    """Per-symbol overlays: the Donchian channel bounds on the price panel.

    Args:
        close: One symbol's adjusted close (channel drawn from close as an
            approximation — signals use high/low, close tracks them closely).
        volume: Unused (kept for the protocol signature).
        entry_window: Channel lookback in bars.
        **_: Extra params ignored.

    Returns:
        ``{"price": {...}, "volume": {}}`` overlay series.
    """
    return {
        "price": {
            f"{entry_window}-bar high": close.rolling(entry_window).max().shift(1),
            f"{entry_window}-bar low": close.rolling(entry_window).min().shift(1),
        },
        "volume": {},
    }
