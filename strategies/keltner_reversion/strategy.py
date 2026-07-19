"""Keltner reversion, intraday: buy ATR-deep dips in uptrends, sell the snap-back.

Thesis (short-term reversal, the mirror image of a breakout): high-volatility
stocks overshoot. When sellers push a name that is still in a long-term
uptrend far below its short-term mean — more than ``band_mult`` ATRs under an
SMA midline, the lower **Keltner band** — part of that move is liquidity
demand, not information, and it tends to snap back toward the mean within a
few bars. Academic short-term-reversal studies find the effect is strongest
exactly where this strategy hunts: volatile names in stressed moments, where
liquidity providers get paid the most. The long-term trend filter keeps the
dip-buying on the right side of the regime (falling knives in downtrends are
excluded); the optional short side fades rallies in downtrends symmetrically.

Signals:

* ``mid`` — SMA over ``band_window`` bars, the reversion target; the entry
  band sits ``band_mult`` ATRs below (above, for shorts).
* Long when close < mid − band_mult·ATR **and** close > SMA(``trend_window``);
  short when close > mid + band_mult·ATR and close < the trend SMA.
  Close-confirmed, executed at the next bar's open.
* Exits: close crossing back over ``mid`` (the profit-take), a ``horizon``-bar
  time stop for reversions that never come, and optionally a fixed stop-loss
  ``sl_mult`` ATRs from entry (``sl_mult: 0`` disables). No trailing stop — a
  trail contradicts a reversion-to-level bet.
* **Session guards** (intraday only): no entries within the first
  ``skip_open`` or last ``skip_close`` minutes, and ``eod_exit`` forces flat
  on the day's final bar so nothing is held overnight. On ``bar: 1d`` the
  guards go inert and the same ``signals()`` trades daily bars unchanged.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(
    panel: Panel,
    *,
    trend_window: int = 100,
    band_window: int = 20,
    atr_window: int = 14,
    band_mult: float = 2.0,
    horizon: int = 10,
    direction: str = "long",
    sl_mult: float = 0.0,
    skip_open: int = 15,
    skip_close: int = 15,
    eod_exit: bool = True,
    cash_frac: float = 0.25,
) -> dict:
    """Build stretched-dip entries, reversion/time exits and an optional ATR stop.

    Args:
        panel: Price panel (uses high/low/close).
        trend_window: Regime SMA lookback in bars; longs need close above it,
            shorts below.
        band_window: Midline SMA lookback in bars — the reversion target.
        atr_window: ATR smoothing window in bars.
        band_mult: Entry stretch in ATRs below/above the midline (higher =
            rarer, deeper dips).
        horizon: Time stop — exit ``horizon`` bars after entry if the midline
            was never touched.
        direction: ``"long"`` | ``"short"`` | ``"both"``.
        sl_mult: Stop-loss distance in ATRs at entry; ``0`` disables the stop.
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
        Dict with ``entries`` / ``exits`` / ``short_entries`` / ``short_exits``
        and, when ``sl_mult`` > 0, a fixed ``sl_stop`` frame (ATR fraction of
        price at entry). Exits fire on midline reversion, the ``horizon`` time
        stop and, intraday with ``eod_exit``, each session's final bar.
    """
    mid = bt.sma(panel.close, band_window)
    trend = bt.sma(panel.close, trend_window)
    rng = bt.atr(panel, atr_window)
    none = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)

    dip = (panel.close < mid - band_mult * rng) & (panel.close > trend)
    rip = (panel.close > mid + band_mult * rng) & (panel.close < trend)
    entries = bt.next_bar(dip) if direction != "short" else none
    short_entries = bt.next_bar(rip) if direction != "long" else none
    intraday = panel.bar != "1d"  # session guards are meaningless on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(panel.close.index, panel.tz,
                             skip_open=skip_open, skip_close=skip_close)
        entries = entries & ok.to_numpy()[:, None]
        short_entries = short_entries & ok.to_numpy()[:, None]

    exits = bt.next_bar(panel.close >= mid) | bt.exit_after(entries, horizon)
    short_exits = bt.next_bar(panel.close <= mid) | bt.exit_after(short_entries, horizon)
    if intraday and eod_exit:
        end = bt.session_end(panel.close.index, panel.tz).to_numpy()[:, None]
        exits = exits | end
        short_exits = short_exits | end

    out = {"entries": entries, "exits": exits,
           "short_entries": short_entries, "short_exits": short_exits}
    if sl_mult:
        out["sl_stop"] = (sl_mult * rng / panel.close).clip(lower=1e-4)
    out["cash_frac"] = cash_frac
    return out


def valid(params: dict) -> bool:
    """Reject combos whose midline is not faster than the trend filter.

    Args:
        params: Candidate parameter combo (needs ``band_window`` and
            ``trend_window``).

    Returns:
        True when ``band_window < trend_window`` — otherwise mean and regime
        swap roles and the setup is meaningless.
    """
    return params["band_window"] < params["trend_window"]


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    trend_window: int = 100,
    band_window: int = 20,
    atr_window: int = 14,
    band_mult: float = 2.0,
    **_,
) -> dict:
    """Per-symbol overlays: Keltner bands, trend SMA and the regime gap.

    Args:
        close: One symbol's adjusted close (bands drawn from a close-to-close
            range proxy — signals use the true high/low ATR, which close
            tracks closely).
        volume: Unused (kept for the protocol signature).
        trend_window: Regime SMA lookback in bars.
        band_window: Midline SMA lookback in bars.
        atr_window: ATR smoothing window in bars.
        band_mult: Band distance in ATRs.
        **_: Extra params ignored.

    Returns:
        ``{"price": {...}, "momentum": {...}, "volume": {}}`` overlay series;
        the momentum panel shows close vs the trend SMA, i.e. where the
        regime filter allowed longs (positive) or shorts (negative).
    """
    mid = bt.sma(close, band_window)
    trend = bt.sma(close, trend_window)
    rng = close.diff().abs().ewm(alpha=1.0 / atr_window, min_periods=atr_window).mean()
    return {
        "price": {
            f"SMA {band_window} midline": mid,
            f"lower band (-{band_mult:g} ATR)": mid - band_mult * rng,
            f"upper band (+{band_mult:g} ATR)": mid + band_mult * rng,
            f"SMA {trend_window} trend": trend,
        },
        "momentum": {f"close vs SMA {trend_window}": close / trend - 1.0},
        "volume": {},
    }
