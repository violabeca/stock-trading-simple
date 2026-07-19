"""Squeeze breakout, intraday: Donchian breaks taken only on multi-gate confluence.

Thesis (the selective complement of ``donchian_breakout``, which trades every
break): raw channel breakouts fail 50–70% of the time, so instead of
monetizing them with volume-of-trades and a tight trail, this strategy trades
FEW breaks and demands confluent evidence for each one. A breakout is taken
only when ALL enabled gates agree at the signal close:

1. **Squeeze** — the tape was compressed before the move started:
   ``ATR / SMA(ATR, squeeze_window)``, lagged ``squeeze_lag`` bars, below
   ``squeeze_thresh``. Volatility clusters; expansion follows compression
   (Crabel NR4/7 lineage). This is an ATR-compression ratio — a deliberate
   lightweight proxy for the TTM Bollinger-inside-Keltner squeeze.
2. **Breakout** — close above the prior ``break_window``-bar high (long) or
   below the prior low (short); close-confirmed, identical to donchian.
3. **Participation** — cumulative volume of the last ``vol_bars`` bars at
   least ``vol_mult``× its time-of-day expectation (``bt.volume_profile``).
   The k-bar sum rewards *building* participation instead of re-measuring the
   breakout bar's own mechanical volume print.
4. **Trend** — close above ``SMA(trend_window)`` for longs, below for shorts:
   break only with the session's drift.

Gate off-switches (``squeeze_thresh: 0`` / ``vol_mult: 0`` / ``trend_window:
0`` disable their gate) exist for ablation; the searched ranges keep every
gate on — if the optimizer only pays with a gate off, the thesis is disproved,
not shipped.

Exits are a layered, volatility-scaled bracket. The stop scale is
``max(ATR, SMA(ATR, squeeze_window))`` — NORMAL volatility, never the
squeeze-compressed reading the entry certifies — with a fixed stop at
``sl_mult`` scales, an optional target at ``tp_mult`` (0 = let winners run),
a ``horizon``-bar time stop, and (intraday) a forced flat on each session's
final bar. There is deliberately **no trailing stop** — the trail is
donchian's exit thesis; the bracket is this one's.

Engine semantics worth knowing (vectorbt, verified): stop fractions are read
at the FILL bar's row and anchor at that bar's close, activating the next
bar; when one bar spans both stop and target the STOP fills (pessimistic);
a stop hit swallows same-bar signal exits; stop legs pay fees but no
slippage, and a bar gapping through a stop fills at the stop level
(optimistic — engine-level, shared by every strategy here). As everywhere in
this repo, ``exit_after`` schedules a time exit for every *signal*, including
ones ignored while a position was already open — phantom exits can clip a
later trade early; rare at this selectivity.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(
    panel: Panel,
    *,
    break_window: int = 60,
    trend_window: int = 390,
    atr_window: int = 14,
    squeeze_window: int = 390,
    squeeze_thresh: float = 0.6,
    squeeze_lag: int = 5,
    vol_mult: float = 3.0,
    vol_bars: int = 3,
    profile_days: int = 10,
    sl_mult: float = 2.0,
    tp_mult: float = 0.0,
    horizon: int = 60,
    direction: str = "both",
    skip_open: int = 15,
    skip_close: int = 15,
    eod_exit: bool = True,
    cash_frac: float = 0.25,
) -> dict:
    """Build confluence-gated breakout entries and the layered bracket exits.

    Args:
        panel: Price panel (uses high/low/close/volume).
        break_window: Channel lookback in bars; breakout = new N-bar extreme.
        trend_window: Regime SMA lookback in bars; longs need close above it,
            shorts below. ``0`` disables the trend gate.
        atr_window: ATR smoothing window in bars (squeeze numerator and the
            spot leg of the stop scale).
        squeeze_window: ATR baseline window in bars — the definition of
            "normal" volatility for both the squeeze ratio and the stop scale.
        squeeze_thresh: Squeeze gate: lagged ATR/baseline ratio must be below
            this. ``0`` disables the gate.
        squeeze_lag: Bars the squeeze ratio is lagged, so a short multi-bar
            approach to the break doesn't void "quiet before the move".
        vol_mult: Participation gate: trailing ``vol_bars``-bar volume sum at
            least this multiple of its expected sum. ``0`` disables the gate.
        vol_bars: Bars in the participation sums.
        profile_days: Trailing sessions in the time-of-day volume baseline.
        sl_mult: Stop distance in stop-scale units (max of ATR and its
            baseline) at entry.
        tp_mult: Profit-target distance in the same units; ``0`` = no target
            (winners run to the time stop / session close).
        horizon: Time stop — exit ``horizon`` bars after entry.
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
        Dict with ``entries`` / ``exits`` / ``short_entries`` / ``short_exits``,
        ``sl_stop`` and, when ``tp_mult`` > 0, ``tp_stop`` (both fractions of
        price, scaled by normal volatility). Exits fire on the stop/target,
        the ``horizon`` time stop and, intraday with ``eod_exit``, each
        session's final bar.
    """
    rng = bt.atr(panel, atr_window)
    base_rng = bt.sma(rng, squeeze_window)

    long_ok = panel.close > panel.high.rolling(
        break_window, min_periods=break_window).max().shift(1)
    short_ok = panel.close < panel.low.rolling(
        break_window, min_periods=break_window).min().shift(1)
    if squeeze_thresh:
        squeeze = (rng / base_rng).shift(squeeze_lag) < squeeze_thresh
        long_ok &= squeeze
        short_ok &= squeeze
    if vol_mult:
        base_vol = bt.volume_profile(panel.volume, panel.tz, days=profile_days)
        vol_ok = (panel.volume.rolling(vol_bars, min_periods=vol_bars).sum()
                  >= vol_mult * base_vol.rolling(vol_bars, min_periods=vol_bars).sum())
        long_ok &= vol_ok
        short_ok &= vol_ok
    if trend_window:
        trend = bt.sma(panel.close, trend_window)
        long_ok &= panel.close > trend
        short_ok &= panel.close < trend

    none = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    entries = bt.next_bar(long_ok) if direction != "short" else none
    short_entries = bt.next_bar(short_ok) if direction != "long" else none
    intraday = panel.bar != "1d"  # session guards are meaningless on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(panel.close.index, panel.tz,
                             skip_open=skip_open, skip_close=skip_close)
        entries = entries & ok.to_numpy()[:, None]
        short_entries = short_entries & ok.to_numpy()[:, None]

    exits = bt.exit_after(entries, horizon)
    short_exits = bt.exit_after(short_entries, horizon)
    if intraday and eod_exit:
        end = bt.session_end(panel.close.index, panel.tz).to_numpy()[:, None]
        exits = exits | end
        short_exits = short_exits | end

    stop_rng = rng.clip(lower=base_rng)  # normal volatility, never the compressed reading
    out = {"entries": entries, "exits": exits,
           "short_entries": short_entries, "short_exits": short_exits,
           "sl_stop": (sl_mult * stop_rng / panel.close).clip(lower=1e-4)}
    if tp_mult:
        out["tp_stop"] = (tp_mult * stop_rng / panel.close).clip(lower=1e-4)
    out["cash_frac"] = cash_frac
    return out


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the trend SMA is meaningfully slower than the breakout
        channel (``trend_window >= 2 * break_window``, unless the gate is
        disabled) and the bracket is not a degenerate short-gamma shape
        (``tp_mult`` either 0 or at least half of ``sl_mult``).
    """
    trend, brk = params.get("trend_window", 390), params.get("break_window", 60)
    tp, sl = params.get("tp_mult", 0.0), params.get("sl_mult", 2.0)
    return (trend == 0 or trend >= 2 * brk) and (tp == 0 or tp >= 0.5 * sl)


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    break_window: int = 60,
    trend_window: int = 390,
    atr_window: int = 14,
    squeeze_window: int = 390,
    squeeze_thresh: float = 0.6,
    squeeze_lag: int = 5,
    vol_mult: float = 3.0,
    vol_bars: int = 3,
    profile_days: int = 10,
    tz: str = "America/New_York",
    **_,
) -> dict:
    """Per-symbol overlays: channel + trend, squeeze ratio, volume trigger.

    Args:
        close: One symbol's adjusted close (channel and ATR drawn from
            close-based proxies — signals use true high/low, which close
            tracks closely).
        volume: One symbol's volume.
        break_window: Channel lookback in bars.
        trend_window: Regime SMA lookback (0 = gate off, line omitted).
        atr_window: ATR smoothing window.
        squeeze_window: ATR baseline window.
        squeeze_thresh: Squeeze gate threshold (drawn as the trigger line).
        squeeze_lag: Squeeze ratio lag in bars.
        vol_mult: Participation multiple (drawn on the volume panel).
        vol_bars: Bars in the participation sums.
        profile_days: Volume-profile lookback sessions.
        tz: The panel's exchange timezone (volume-profile grouping).
        **_: Extra params ignored.

    Returns:
        ``{"price", "momentum", "volume", "volume_marks"}`` overlay series:
        channel bounds and trend SMA on price, the lagged squeeze ratio vs its
        threshold on momentum, and the expected-volume trigger with the bars
        that passed the participation gate.
    """
    rng = close.diff().abs().ewm(alpha=1.0 / atr_window, min_periods=atr_window).mean()
    ratio = (rng / rng.rolling(squeeze_window, min_periods=squeeze_window).mean()
             ).shift(squeeze_lag)
    frame = volume.to_frame("v")
    base = bt.volume_profile(frame, tz, days=profile_days)["v"]
    vol_sum = volume.rolling(vol_bars, min_periods=vol_bars).sum()
    base_sum = base.rolling(vol_bars, min_periods=vol_bars).sum()
    price = {
        f"{break_window}-bar high": close.rolling(break_window).max().shift(1),
        f"{break_window}-bar low": close.rolling(break_window).min().shift(1),
    }
    if trend_window:
        price[f"SMA {trend_window} trend"] = bt.sma(close, trend_window)
    return {
        "price": price,
        "momentum": {
            "squeeze ratio (ATR / baseline)": ratio,
            f"squeeze threshold ({squeeze_thresh:g})":
                pd.Series(squeeze_thresh, index=close.index),
        },
        "volume": {"expected volume": base, f"vol trigger ({vol_mult:g}x)": vol_mult * base},
        "volume_marks": vol_sum >= vol_mult * base_sum,
    }
