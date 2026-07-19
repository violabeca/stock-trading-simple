"""Donchian breakout, intraday: minute-channel breakouts with a tight ATR trailing stop.

Thesis (the classic "Turtle" entry, run inside the day): when price closes
above its highest high of the last ``entry_window`` minutes, the odds of
short-term continuation are a little better than a coin flip (~52% in
published breakout studies). That small directional edge only pays if losses
stay small — so every position carries a **trailing stop** at ``atr_mult``
ATRs from the best price since entry. The tight trail serves double duty: it
caps the initial loss AND locks in profit as the move extends. Expect many
small stop-outs and a few larger winners.

Signals:

* ``upper`` / ``lower`` — rolling max(high) / min(low) over ``entry_window``
  bars, shifted one bar so a bar can't break out of a channel it defines.
* Long when close > upper; short when close < lower (close-confirmed, which
  filters intrabar noise). Executed at the next bar's open.
* **Session guards** (intraday only): no entries within the first
  ``skip_open`` minutes (the opening range is still forming — every early bar
  is a trivial "breakout" of a channel built overnight) or the last
  ``skip_close`` minutes, and ``eod_exit`` forces every position flat on the
  day's final bar — nothing is held overnight.
* Exits are otherwise stop-driven: the engine passes ``sl_stop`` (an ATR
  fraction of price, per entry) with ``sl_trail`` on to vectorbt, which
  evaluates it intrabar against high/low. An opposite breakout closes the
  position (the shared wallet's percent sizing cannot reverse in one order).
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(
    panel: Panel,
    *,
    entry_window: int = 30,
    atr_window: int = 14,
    atr_mult: float = 2.0,
    direction: str = "both",
    skip_open: int = 15,
    skip_close: int = 15,
    eod_exit: bool = True,
    cash_frac: float = 0.25,
) -> dict:
    """Build breakout entries, ATR trailing-stop parameters and session exits.

    Args:
        panel: Price panel (uses high/low/close).
        entry_window: Channel lookback in bars (breakout = new N-bar extreme;
            bars = minutes on 1m data).
        atr_window: ATR smoothing window in bars.
        atr_mult: Trailing-stop distance in ATRs (lower = tighter stop,
            smaller losses, more frequent stop-outs).
        direction: ``"long"`` | ``"short"`` | ``"both"``.
        skip_open: No entries within the first N minutes of a session
            (intraday only).
        skip_close: No entries within the last N minutes of a session
            (intraday only).
        eod_exit: Force positions flat on each session's final bar
            (intraday only).
        cash_frac: Fraction of the shared wallet's free cash deployed per
            entry. Its presence runs the whole strategy as ONE ``init_cash``
            book (engine cash-shared signal mode) — total invested can never
            exceed the wallet.

    Returns:
        Dict with ``entries``, ``short_entries``, ``sl_stop`` (ATR fraction of
        price at entry), ``sl_trail=True`` and, intraday with ``eod_exit``,
        ``exits`` / ``short_exits`` on each session's final bar. There are no
        other time-based exits: positions leave via the trailing stop, the
        end of the session, or an opposite breakout (which closes, never
        reverses, under the shared wallet's percent sizing).
    """
    upper = panel.high.rolling(entry_window, min_periods=entry_window).max().shift(1)
    lower = panel.low.rolling(entry_window, min_periods=entry_window).min().shift(1)
    none = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)

    entries = bt.next_bar(panel.close > upper) if direction != "short" else none
    short_entries = bt.next_bar(panel.close < lower) if direction != "long" else none
    intraday = panel.bar != "1d"  # session guards are meaningless on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(panel.close.index, panel.tz,
                             skip_open=skip_open, skip_close=skip_close)
        entries = entries & ok.to_numpy()[:, None]
        short_entries = short_entries & ok.to_numpy()[:, None]

    out = {
        "entries": entries,
        "short_entries": short_entries,
        "sl_stop": (atr_mult * bt.atr(panel, atr_window) / panel.close).clip(lower=1e-4),
        "sl_trail": True,
    }
    if intraday and eod_exit:
        end = none | bt.session_end(panel.close.index, panel.tz).to_numpy()[:, None]
        out |= {"exits": end, "short_exits": end.copy()}
    out["cash_frac"] = cash_frac
    return out


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    entry_window: int = 30,
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
