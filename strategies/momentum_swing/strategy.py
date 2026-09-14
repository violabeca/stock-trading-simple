"""Momentum swing: buy leaders breaking out of tight bases and neglected names gapping up on
huge volume, risk one fixed stop, trail the rest on a moving average — only in a rising tape.

Thesis. This is the systematic core of the swing-trading framework popularised by Kristjan
"Qullamaggie" Kullamägi (and, before him, Minervini's VCP and Bonde's episodic pivots): a
stock that has already moved a lot and then rests in a *tightening* range is being held by
strong hands; when it closes above the range on expanding volume, the move tends to resume,
and the resumption is worth several times the distance to the range's low. The academic
footing is real but partial: continuation after breakouts to highs (George & Hwang 2004),
the high-volume return premium (Gervais, Kaniel & Mingelgrin 2001), post-earnings drift
after large gaps (Bernard & Thomas 1989 and successors), volatility compression before
expansion, index-trend gates that cut momentum drawdowns (Faber 2007; Barroso &
Santa-Clara 2015), and stop-losses that tame momentum crashes (Han, Zhou & Zhu 2016).
Against that stand a low win rate with the profit concentrated in a few outliers
(Zarattini, Pagani & Wilcox 2025: under 7% of trend-following trades carry the profit),
the lottery-stock discount on high-range names (Bali, Cakici & Whitelaw 2011), and the
one faithful rule-based rebuild in the public domain losing to the index. So every
discretionary knob is a parameter here, the sweep grid is an ablation (setup × regime ×
trail), and the notebook's pre-registered event test runs before any optimisation.

Two long setups share one exit machinery (see the README for the third, the parabolic
short, and why it is deliberately not shipped):

* **Breakout** — the name rose at least ``lead_min`` over the ``lead_window`` bars *before*
  its base; the last ``base_window`` bars are a tight base (average daily range at most
  ``contraction`` × the leader-period average, high-to-low depth at most ``depth_max``); the
  close is above the base's highest high on volume at least ``vol_mult`` × its
  ``avg_window``-bar average.
* **Episodic pivot** — the name opened at least ``gap_min`` above the prior close on volume
  at least ``gap_vol_mult`` × average, held the gap (close ≥ open), and had *not* already
  rallied (prior ``lead_window`` return below ``lead_min``: neglect is the point).

Common gates: average daily range at least ``adr_min`` (the setup needs names that move),
the close above the ``trail_sma`` (there must be something to trail), the initial stop within
``max_stop`` of price (bounds the per-trade risk under fixed-fraction sizing), and the
market regime: ``none``, ``fast`` (the panel's equal-weight index has its ``regime_fast``
SMA above its ``regime_slow`` SMA — the community "10 over 20" traffic light) or ``slow``
(the index above its ``regime_long`` SMA — the 200-day gate every systematic rebuild of
this style ends up with).

Exits, engine-enforced: an initial stop ``stop_atr`` ATRs below entry (evaluated intrabar,
filled at the true open when a bar gaps through it — overnight gap risk is modelled, not
assumed away), an optional full take-profit at ``tp_r`` × the stop distance (0 = off; the
engine has no partial sells, so the "sell a third at 3–5R" rule becomes a searchable
whole-position target), and the trail: the next open after the first close below the
``trail_sma``-bar SMA.

Capital: **one shared wallet** (``cash_frac`` of the free cash per entry, the repo default),
so the book can never be more than fully invested and simultaneous breakouts compete for the
same cash, exactly as in one real account. Risk-per-trade sizing (position = risk ÷ stop
distance) is the framework's own rule and is *not* expressible through a scalar
``cash_frac``; the README carries the exact engine proposal. Every signal is next-bar-open
executed (:func:`trading_simple.backtest.next_bar`); the stop frame is shifted onto the fill
bar so it carries only signal-bar information. There is no look-ahead.

:func:`setups` exposes the raw masks and intermediates so the notebook can score the event
itself — forward returns after each setup, demeaned by the panel — without the engine.
"""

from __future__ import annotations

import inspect

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

_SETUPS = ("breakout", "pivot", "both")
_REGIMES = ("none", "fast", "slow")


def signals(
    panel: Panel,
    *,
    setup: str = "both",
    lead_window: int = 63,
    lead_min: float = 0.25,
    base_window: int = 15,
    contraction: float = 0.9,
    depth_max: float = 0.25,
    adr_min: float = 0.03,
    vol_mult: float = 1.5,
    gap_min: float = 0.08,
    gap_vol_mult: float = 3.0,
    regime: str = "fast",
    regime_fast: int = 10,
    regime_slow: int = 20,
    regime_long: int = 200,
    avg_window: int = 20,
    atr_window: int = 14,
    stop_atr: float = 1.5,
    max_stop: float = 0.12,
    trail_sma: int = 10,
    tp_r: float = 0.0,
    cash_frac: float = 0.25,
) -> dict:
    """Build execution-timed long entries, stop parameters and trailing-SMA exits.

    Args:
        panel: Daily price panel (uses open/high/low/close/volume).
        setup: ``"breakout"`` | ``"pivot"`` | ``"both"`` — which setups may enter.
        lead_window: Bars of the leader move (breakout: measured *before* the base; pivot:
            the neglect window ending the day before the gap).
        lead_min: Minimum leader return for a breakout; a pivot requires *less* than this.
        base_window: Bars in the consolidation; a breakout closes above their highest high.
        contraction: Base average daily range ÷ leader-period average must be at most this
            (the volatility-contraction test; ≥ ~1.2 effectively disables it).
        depth_max: Base high-to-low depth at most this (an orderly base, not a collapse).
        adr_min: Average daily range (high/low − 1, over ``lead_window`` bars) floor.
        vol_mult: Breakout-day volume vs its ``avg_window``-bar average.
        gap_min: Pivot: open at least this far above the prior close.
        gap_vol_mult: Pivot: gap-day volume vs its ``avg_window``-bar average.
        regime: ``"none"`` | ``"fast"`` | ``"slow"`` market gate (see module docstring).
        regime_fast: Fast SMA of the equal-weight index (``fast`` regime).
        regime_slow: Slow SMA of the equal-weight index (``fast`` regime).
        regime_long: Long SMA of the equal-weight index (``slow`` regime).
        avg_window: Bars in the volume average.
        atr_window: ATR smoothing window (bars).
        stop_atr: Initial stop distance in ATRs (fixed, not trailing).
        max_stop: Skip a setup whose stop would sit further than this fraction below price.
        trail_sma: Exit on the next open after a close below this SMA.
        tp_r: Full take-profit at this multiple of the stop distance; ``0`` = none.
        cash_frac: Fraction of the shared wallet's free cash deployed per entry (engine
            cash-shared mode: ONE ``init_cash`` book for all symbols).

    Returns:
        Dict with ``entries`` / ``exits`` (next-bar executed), ``sl_stop`` (a per-entry
        fraction of price, aligned to the fill bar), ``sl_trail=False``, ``tp_stop`` when
        ``tp_r > 0``, and ``cash_frac``. Long only.
    """
    parts = _compose(
        panel, setup=setup, lead_window=lead_window, lead_min=lead_min,
        base_window=base_window, contraction=contraction, depth_max=depth_max,
        adr_min=adr_min, vol_mult=vol_mult, gap_min=gap_min, gap_vol_mult=gap_vol_mult,
        regime=regime, regime_fast=regime_fast, regime_slow=regime_slow,
        regime_long=regime_long, avg_window=avg_window, atr_window=atr_window,
        stop_atr=stop_atr, max_stop=max_stop, trail_sma=trail_sma)
    # The stop is decided on the signal bar and applied at the fill bar (shift), so it never
    # reads the fill bar's own range; vectorbt takes the value at the entry bar.
    sl_stop = parts["stop_frac"].shift(1).clip(lower=1e-4)
    out = {
        "entries": bt.next_bar(parts["signal"]),
        "exits": bt.next_bar(panel.close < parts["trail"]),
        "sl_stop": sl_stop,
        "sl_trail": False,
        "cash_frac": cash_frac,
    }
    if tp_r > 0:
        out["tp_stop"] = sl_stop * tp_r
    return out


def _compose(
    panel: Panel,
    *,
    setup: str,
    lead_window: int,
    lead_min: float,
    base_window: int,
    contraction: float,
    depth_max: float,
    adr_min: float,
    vol_mult: float,
    gap_min: float,
    gap_vol_mult: float,
    regime: str,
    regime_fast: int,
    regime_slow: int,
    regime_long: int,
    avg_window: int,
    atr_window: int,
    stop_atr: float,
    max_stop: float,
    trail_sma: int,
    **_,
) -> dict:
    """Every intermediate the entries rest on (shared by :func:`signals` and :func:`setups`).

    All masks are on the *signal* clock (bar ``t`` = the bar whose close confirms the
    setup); :func:`signals` shifts them onto the fill bar.

    Args:
        panel: Daily price panel.
        (parameters): See :func:`signals`.
        **_: Ignored, so callers can splat a full parameter dict.

    Returns:
        ``{"breakout", "pivot", "gate", "signal", "regime_ok", "market", "adr", "stop_frac",
        "trail"}`` — boolean masks for the two raw setups, the common gate, the gated
        signal actually traded, the per-bar regime flag, the panel's equal-weight index,
        the average daily range, the stop distance as a fraction of price, and the trailing
        SMA.
    """
    close, high, low, open_, volume = panel.close, panel.high, panel.low, panel.open_, panel.volume
    prev = close.shift(1)
    day_range = high / low - 1.0
    adr = bt.sma(day_range, lead_window).shift(1)               # the name's typical range
    avg_vol = bt.sma(volume, avg_window).shift(1)
    trail = bt.sma(close, trail_sma)
    stop_frac = stop_atr * bt.atr(panel, atr_window) / close

    # Breakout: a leader (the move BEFORE the base), a tight orderly base, a close above it
    # on expanding volume.
    base_high = high.rolling(base_window, min_periods=base_window).max().shift(1)
    base_low = low.rolling(base_window, min_periods=base_window).min().shift(1)
    lead_before_base = close.shift(base_window) / close.shift(base_window + lead_window) - 1.0
    tight = bt.sma(day_range, base_window).shift(1) / adr
    breakout = ((close > base_high)
                & (lead_before_base >= lead_min)
                & (tight <= contraction)
                & (base_high / base_low - 1.0 <= depth_max)
                & (volume >= vol_mult * avg_vol))

    # Episodic pivot: a NEGLECTED name gaps up on huge volume and holds the gap.
    gap = open_ / prev - 1.0
    lead_to_yesterday = prev / close.shift(1 + lead_window) - 1.0
    pivot = ((gap >= gap_min)
             & (volume >= gap_vol_mult * avg_vol)
             & (close >= open_)
             & (lead_to_yesterday < lead_min))

    # The market: an equal-weight index of the panel itself — the same composition as the
    # benchmark, and free of any outside series.
    market = (1.0 + close.pct_change(fill_method=None).mean(axis=1)).cumprod()
    if regime == "fast":
        regime_ok = bt.sma(market, regime_fast) > bt.sma(market, regime_slow)
    elif regime == "slow":
        regime_ok = market > bt.sma(market, regime_long)
    else:
        regime_ok = pd.Series(True, index=close.index)
    gate = ((adr >= adr_min) & (close > trail) & (stop_frac <= max_stop)
            & regime_ok.to_numpy()[:, None])
    raw = {"breakout": breakout, "pivot": pivot, "both": breakout | pivot}[setup]
    return {"breakout": breakout, "pivot": pivot, "gate": gate, "signal": raw & gate,
            "regime_ok": regime_ok, "market": market, "adr": adr, "stop_frac": stop_frac,
            "trail": trail}


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the setup and regime names are known, the base is shorter than the leader
        window it is compared against, the regime SMAs are ordered, and every distance,
        multiple and window is positive where it must be.
    """
    return (params.get("setup", "both") in _SETUPS
            and params.get("regime", "fast") in _REGIMES
            and 2 <= params.get("base_window", 15) < params.get("lead_window", 63)
            and params.get("regime_fast", 10) < params.get("regime_slow", 20)
            and params.get("regime_long", 200) >= 2
            and params.get("lead_min", 0.3) >= 0.0
            and params.get("contraction", 0.75) > 0.0
            and params.get("depth_max", 0.25) > 0.0
            and params.get("gap_min", 0.08) > 0.0
            and params.get("stop_atr", 1.5) > 0.0
            and 0.0 < params.get("max_stop", 0.12) <= 1.0
            and params.get("trail_sma", 10) >= 2
            and params.get("tp_r", 0.0) >= 0.0)


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    base_window: int = 15,
    trail_sma: int = 10,
    avg_window: int = 20,
    vol_mult: float = 1.5,
    **_,
) -> dict:
    """Per-symbol overlays: the trailing SMA, the base high, the volume baseline and spikes.

    Args:
        close: One symbol's adjusted close (the base high is drawn from closes as an
            approximation — signals use highs, closes track them closely).
        volume: The symbol's volume.
        base_window: Base lookback in bars.
        trail_sma: Trailing SMA window.
        avg_window: Volume-average window.
        vol_mult: Volume multiple that marks a spike.
        **_: Extra params ignored.

    Returns:
        ``{"price": {...}, "volume": {...}, "volume_marks": <bool series>}`` overlays.
    """
    avg_vol = bt.sma(volume, avg_window).shift(1)
    return {
        "price": {f"SMA {trail_sma} (trail)": bt.sma(close, trail_sma),
                  f"{base_window}-bar base high": close.rolling(base_window).max().shift(1)},
        "volume": {f"avg {avg_window}": avg_vol},
        "volume_marks": volume >= vol_mult * avg_vol,
    }


_DEFAULTS = {name: p.default for name, p in inspect.signature(signals).parameters.items()
             if p.default is not inspect.Parameter.empty}


def setups(panel: Panel, **params) -> dict:
    """Expose the raw setup masks and intermediates for the notebook's event test.

    Shares its whole computation with :func:`signals` (via :func:`_compose`), so the event
    study scores exactly the setups the backtest trades. Missing parameters default to
    :func:`signals`' own signature.

    Args:
        panel: The panel the backtest runs on.
        **params: Any :func:`signals` parameters to override.

    Returns:
        The :func:`_compose` result — ``breakout``, ``pivot``, ``gate``, ``signal``,
        ``regime_ok``, ``market``, ``adr``, ``stop_frac``, ``trail`` — all on the signal
        clock (unshifted).
    """
    return _compose(panel, **{**_DEFAULTS, **params})
