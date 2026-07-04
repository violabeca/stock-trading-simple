"""Volume momentum: abnormal volume decides WHEN, the price trend decides WHICH WAY.

Thesis: a bar with far more volume than usual marks the arrival of new
information or a large participant. If the market is already trending, that
burst tends to continue the move for a short window — so we enter in the
trend's direction at the next bar's open and exit a fixed number of bars later.

Signals (all pure pandas, no look-ahead):

* **spike** — volume >= ``threshold`` x its expected level. The expectation is
  either a trailing SMA (``baseline: sma``, the classic choice for daily bars)
  or a time-of-day volume profile (``baseline: profile``, the right choice for
  intraday bars: it compares 09:31 volume only to other 09:31 volumes, so the
  U-shaped intraday curve doesn't blind the detector). ``baseline: auto`` picks
  by bar size.
* **trend** — sign of SMA(fast) − SMA(slow).
* Long on spike-in-uptrend, short on spike-in-downtrend (per ``direction``),
  executed next bar, closed ``horizon`` bars later.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(
    panel: Panel,
    *,
    fast: int,
    slow: int,
    threshold: float,
    horizon: int,
    baseline: str = "auto",
    vol_window: int = 20,
    profile_days: int = 10,
    direction: str = "both",
) -> dict:
    """Build execution-timed entry/exit frames for the volume-momentum rules.

    Args:
        panel: Price panel (close, volume; UTC index).
        fast: Fast price-SMA window in bars.
        slow: Slow price-SMA window in bars (must exceed ``fast``).
        threshold: Volume spike multiple vs the baseline (e.g. 3.0 = 3x normal).
        horizon: Holding period in bars after entry.
        baseline: ``"sma"`` | ``"profile"`` | ``"auto"`` (profile for intraday
            bars, sma for daily).
        vol_window: Trailing window in bars for the SMA baseline.
        profile_days: Trailing sessions for the time-of-day profile baseline.
        direction: ``"long"`` | ``"short"`` | ``"both"``.

    Returns:
        Dict with ``entries``, ``exits``, ``short_entries``, ``short_exits``.
    """
    base = _volume_baseline(panel, baseline, vol_window, profile_days)
    spike = panel.volume >= threshold * base  # NaN baseline (warm-up) compares False
    trend = bt.sma(panel.close, fast) - bt.sma(panel.close, slow)
    none = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)

    entries = bt.next_bar(spike & (trend > 0)) if direction != "short" else none
    short_entries = bt.next_bar(spike & (trend < 0)) if direction != "long" else none
    return {
        "entries": entries,
        "exits": bt.exit_after(entries, horizon),
        "short_entries": short_entries,
        "short_exits": bt.exit_after(short_entries, horizon),
    }


def valid(params: dict) -> bool:
    """Reject combos where the fast SMA is not faster than the slow one."""
    return params["fast"] < params["slow"]


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    fast: int,
    slow: int,
    threshold: float,
    baseline: str = "auto",
    vol_window: int = 20,
    profile_days: int = 10,
    bar: str = "1d",
    tz: str = "America/New_York",
    **_,
) -> dict:
    """Per-symbol plot overlays for :func:`trading_simple.plots.plot_symbol`.

    Args:
        close: One symbol's adjusted close.
        volume: One symbol's volume.
        fast: Fast price-SMA window.
        slow: Slow price-SMA window.
        threshold: Spike multiple (drawn as the trigger line on the volume panel).
        baseline: Baseline mode (see :func:`signals`).
        vol_window: SMA-baseline window.
        profile_days: Profile-baseline sessions.
        bar: The panel's bar size (resolves ``baseline="auto"``).
        tz: The panel's exchange timezone (profile grouping).
        **_: Extra params from the config are ignored.

    Returns:
        ``{"price": {...}, "volume": {...}}`` overlay series.
    """
    frame = volume.to_frame("v")
    mode = _resolve(baseline, bar)
    base = (bt.volume_profile(frame, tz, days=profile_days) if mode == "profile"
            else bt.sma(frame, vol_window).shift(1))["v"]
    return {
        "price": {f"SMA {fast}": bt.sma(close, fast), f"SMA {slow}": bt.sma(close, slow)},
        "volume": {"expected volume": base, f"spike trigger ({threshold}x)": threshold * base},
    }


def _resolve(baseline: str, bar: str) -> str:
    """Resolve ``baseline="auto"`` to sma (daily) or profile (intraday)."""
    return ("sma" if bar == "1d" else "profile") if baseline == "auto" else baseline


def _volume_baseline(panel: Panel, baseline: str, vol_window: int,
                     profile_days: int) -> pd.DataFrame:
    """Expected-volume frame for the chosen baseline mode."""
    if _resolve(baseline, panel.bar) == "profile":
        return bt.volume_profile(panel.volume, panel.tz, days=profile_days)
    return bt.sma(panel.volume, vol_window).shift(1)  # trailing; excludes the current bar
