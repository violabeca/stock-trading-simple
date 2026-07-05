"""Volume momentum: abnormal volume decides WHEN, the price trend decides WHICH WAY.

Intraday thesis: a minute with far more volume than usual marks the arrival of
new information or a large participant, and the burst tends to continue the
prevailing move for a short window. All pure pandas, no look-ahead:

* **spike** — volume >= ``threshold`` x its expected level. On intraday bars
  the expectation is the **time-of-day volume profile**: today's 09:35 volume
  is compared to the average of the last ``profile_days`` sessions at 09:35
  (``±smooth_minutes`` around it), so the routinely heavy open/close minutes
  never trigger by themselves (``baseline: profile``; a trailing
  ``vol_window``-bar SMA remains available as ``baseline: sma``, the classic
  daily-bar choice; ``auto`` picks by bar size).
* **momentum** — SMA(fast) − SMA(slow) decides the side from the current
  market condition alone: positive = long, negative = short. There is no
  long/short/both switch — both directions are always live.
* **session guards** (intraday only): no entries within the first
  ``skip_open`` / last ``skip_close`` minutes of each session, and
  ``eod_exit`` forces every position flat on the day's final bar.
* **stop loss** — ``sl_pct`` sets the stop distance as a fraction of entry
  price (e.g. 0.02 = 2%); ``sl_trail: true`` makes it follow the best price
  since entry instead of staying fixed. Evaluated intrabar by the engine.

Entries execute at the next bar's open; time exits ``horizon`` bars later.
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
    profile_days: int = 5,
    smooth_minutes: int = 2,
    skip_open: int = 15,
    skip_close: int = 15,
    sl_pct: float = 0.0,
    sl_trail: bool = False,
    eod_exit: bool = True,
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
        smooth_minutes: Profile smoothing half-width (compare to ±N minutes
            around the same time-of-day; intraday only).
        skip_open: No entries within the first N minutes of a session
            (intraday only).
        skip_close: No entries within the last N minutes of a session
            (intraday only).
        sl_pct: Stop-loss distance as a fraction of entry price; 0 disables.
        sl_trail: True = trailing stop, False = fixed at entry.
        eod_exit: Force positions flat on each session's final bar
            (intraday only).

    Returns:
        Dict with ``entries``, ``exits``, ``short_entries``, ``short_exits``
        and, when ``sl_pct`` is set, ``sl_stop`` / ``sl_trail``.
    """
    base = _volume_baseline(panel, baseline, vol_window, profile_days, smooth_minutes)
    spike = panel.volume >= threshold * base  # NaN baseline (warm-up) compares False
    trend = bt.sma(panel.close, fast) - bt.sma(panel.close, slow)

    entries = bt.next_bar(spike & (trend > 0))
    short_entries = bt.next_bar(spike & (trend < 0))
    intraday = panel.bar != "1d"  # session guards are meaningless on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(panel.close.index, panel.tz,
                             skip_open=skip_open, skip_close=skip_close)
        entries &= ok.to_numpy()[:, None]
        short_entries &= ok.to_numpy()[:, None]

    exits = bt.exit_after(entries, horizon)
    short_exits = bt.exit_after(short_entries, horizon)
    if intraday and eod_exit:
        end = bt.session_end(panel.close.index, panel.tz).to_numpy()[:, None]
        exits |= end
        short_exits |= end

    out = {"entries": entries, "exits": exits,
           "short_entries": short_entries, "short_exits": short_exits}
    if sl_pct:
        out |= {"sl_stop": float(sl_pct), "sl_trail": bool(sl_trail)}
    return out


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
    profile_days: int = 5,
    smooth_minutes: int = 2,
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
        smooth_minutes: Profile smoothing half-width (see :func:`signals`).
        bar: The panel's bar size (resolves ``baseline="auto"``).
        tz: The panel's exchange timezone (profile grouping).
        **_: Extra params from the config are ignored.

    Returns:
        ``{"price", "momentum", "volume", "volume_marks"}`` overlay series:
        the two SMAs, the trend momentum (fast−slow as a fraction of slow,
        the long/short direction filter), the expected-volume baseline with
        its spike trigger line, and the boolean spike bars.
    """
    frame = volume.to_frame("v")
    mode = _resolve(baseline, bar)
    base = (bt.volume_profile(frame, tz, days=profile_days, smooth=smooth_minutes)
            if mode == "profile" else bt.sma(frame, vol_window).shift(1))["v"]
    sma_fast, sma_slow = bt.sma(close, fast), bt.sma(close, slow)
    return {
        "price": {f"SMA {fast}": sma_fast, f"SMA {slow}": sma_slow},
        "momentum": {f"momentum {fast}−{slow} (%)": (sma_fast - sma_slow) / sma_slow},
        "volume": {"expected volume": base, f"spike trigger ({threshold}x)": threshold * base},
        "volume_marks": volume >= threshold * base,
    }


def _resolve(baseline: str, bar: str) -> str:
    """Resolve ``baseline="auto"`` to sma (daily) or profile (intraday)."""
    return ("sma" if bar == "1d" else "profile") if baseline == "auto" else baseline


def _volume_baseline(panel: Panel, baseline: str, vol_window: int,
                     profile_days: int, smooth_minutes: int) -> pd.DataFrame:
    """Expected-volume frame for the chosen baseline mode."""
    if _resolve(baseline, panel.bar) == "profile":
        return bt.volume_profile(panel.volume, panel.tz, days=profile_days,
                                 smooth=smooth_minutes)
    return bt.sma(panel.volume, vol_window).shift(1)  # trailing; excludes the current bar
