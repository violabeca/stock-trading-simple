"""Candlestick scalp, intraday: reversal candles at a stretched location, gated.

Thesis (an honest test of candlestick reading, not a claim it works): a "manual
scalper" watches a handful of highly-fluctuating names and buys a *rejection*
candle — a hammer with a long lower wick, or a bullish engulfing — betting the
sellers who pushed price down are exhausted. The academic verdict on candle
*shapes* as intraday signals is close to unanimously negative once transaction
costs and data-snooping are accounted for (Marshall-Young-Rose 2006;
Duvinage-Mazza-Petitjean 2013; Fock-Klein-Zwergel 2005): the raw edge of a shape
is smaller than the bid-ask spread. So this strategy does NOT trade shapes
naked. It treats the candle as a *confirmation* layered on the one intraday edge
with real support — short-term reversal at a volatility extreme (the same edge
``keltner_reversion`` fades) — plus optional trend and relative-volume gates.

Every gate has an off-switch, so the optimizer, the holdout and a gate-ablation
answer the real question: does reading the candle add anything over the location
edge alone? If ``pattern: any`` (no shape required) or a gate-off config wins the
front, the candlestick thesis is disproved for these names — that is a result,
not a failure, and such a config is not shipped.

Signals:

* **Pattern** (``pattern``) — the reversal candle the entry requires:
  ``hammer`` (long lower wick dominates the body + upper wick — rejection of
  lower prices; the bearish mirror is a shooting star), ``engulfing`` (the bar's
  body engulfs the prior, opposite-coloured body), or ``any`` (no shape
  requirement — the ablation control that reduces the strategy to a pure
  location reversal).
* **Location** (``stretch_mult``, 0 = off) — the candle only counts when price
  is stretched ``stretch_mult`` ATRs below (above) ``SMA(band_window)``: the
  overshoot that gives a reversal something to revert.
* **Trend** (``trend_window``, 0 = off) — longs only above the regime SMA,
  shorts only below it: fade dips with the drift, not into a collapse.
* **Volume** (``vol_mult``, 0 = off) — the rejection bar carries at least
  ``vol_mult``x its time-of-day expected volume (``bt.volume_profile``): real
  capitulation, not a thin-liquidity wick.
* **Exits** — a volatility-scaled bracket: a fixed ``sl_mult``-ATR stop, an
  optional ``tp_mult``-ATR target (0 = run to the time stop), a ``horizon``-bar
  time stop, and (intraday) a forced flat on each session's final bar.
* **Session guards** (intraday only) — no entries in the first ``skip_open`` or
  last ``skip_close`` minutes; ``eod_exit`` holds nothing overnight. On
  ``bar: 1d`` the guards go inert and the same ``signals()`` trades daily bars.

Costs are the whole game here. A scalp target must clear ~10-15 bps of
round-trip friction on a volatile name; the always-on cost model will bury
anything that does not, which is exactly the point of building it this way.
Engine stop semantics (shared by every strategy here, vectorbt-verified): stop
fractions anchor at the fill bar's close and activate the next bar; when one bar
spans both stop and target the STOP fills (pessimistic); a gap through a stop
fills at the bar's true open with slippage charged (engine-pessimistic since
2026-07-19). On a 1-minute bar the intrabar path is unknown, so a
tight bracket's win rate is not trustworthy — judge on the time-stop / EOD exits.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def signals(
    panel: Panel,
    *,
    pattern: str = "hammer",
    wick_mult: float = 2.0,
    band_window: int = 120,
    atr_window: int = 30,
    stretch_mult: float = 3.0,
    trend_window: int = 960,
    vol_mult: float = 0.0,
    profile_days: int = 10,
    sl_mult: float = 2.0,
    tp_mult: float = 3.0,
    horizon: int = 60,
    direction: str = "both",
    skip_open: int = 15,
    skip_close: int = 15,
    eod_exit: bool = True,
    cash_frac: float = 0.25,
) -> dict:
    """Build reversal-candle entries at a stretched location and bracket exits.

    Args:
        panel: Price panel (uses open/high/low/close and, when ``vol_mult``,
            volume).
        pattern: Reversal candle required to enter — ``"hammer"`` (rejection
            wick), ``"engulfing"`` (body engulfs the prior bar) or ``"any"``
            (no shape requirement; the location-only ablation control).
        wick_mult: Hammer/shooting-star strictness — the rejection wick must be
            at least this multiple of the rest of the candle (body + opposite
            wick). Higher = a purer rejection bar. Ignored unless
            ``pattern == "hammer"``.
        band_window: Midline SMA lookback in bars — the "mean" price overshoots.
        atr_window: ATR smoothing window in bars; sets the stretch and stop scale.
        stretch_mult: Entry stretch in ATRs below/above the midline; ``0``
            disables the location gate (fires on the candle alone).
        trend_window: Regime SMA lookback in bars; longs need close above it,
            shorts below. ``0`` disables the trend gate.
        vol_mult: Participation gate — the entry bar's volume must be at least
            this multiple of its time-of-day expectation. ``0`` disables it.
        profile_days: Trailing sessions in the time-of-day volume baseline.
        sl_mult: Stop-loss distance in ATRs at entry.
        tp_mult: Take-profit distance in ATRs; ``0`` = no target (run to the
            time stop / session close).
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
        ``sl_stop`` and, when ``tp_mult`` > 0, ``tp_stop`` (both ATR fractions of
        price). Exits fire on the stop/target, the ``horizon`` time stop and,
        intraday with ``eod_exit``, each session's final bar.
    """
    o, h, l, c = panel.open_, panel.high, panel.low, panel.close  # noqa: E741
    top = c.where(c >= o, o)          # elementwise max(open, close)
    bot = c.where(c <= o, o)          # elementwise min(open, close)
    body = (c - o).abs()
    upper_wick = h - top
    lower_wick = bot - l

    # --- reversal-candle triggers (bullish for longs, bearish for shorts) ---
    if pattern == "hammer":
        bull_pat = lower_wick >= wick_mult * (body + upper_wick)
        bear_pat = upper_wick >= wick_mult * (body + lower_wick)
    elif pattern == "engulfing":
        o1, c1 = o.shift(1), c.shift(1)
        bull_pat = (c > o) & (c1 < o1) & (o <= c1) & (c >= o1)
        bear_pat = (c < o) & (c1 > o1) & (o >= c1) & (c <= o1)
    else:  # "any" — no shape requirement (location-only control)
        bull_pat = pd.DataFrame(True, index=c.index, columns=c.columns)
        bear_pat = bull_pat

    long_ok, short_ok = bull_pat, bear_pat
    rng = bt.atr(panel, atr_window)
    if stretch_mult:
        mid = bt.sma(c, band_window)
        long_ok = long_ok & (c < mid - stretch_mult * rng)
        short_ok = short_ok & (c > mid + stretch_mult * rng)
    if trend_window:
        trend = bt.sma(c, trend_window)
        long_ok = long_ok & (c > trend)
        short_ok = short_ok & (c < trend)
    if vol_mult:
        base_vol = bt.volume_profile(panel.volume, panel.tz, days=profile_days)
        vol_ok = panel.volume >= vol_mult * base_vol
        long_ok = long_ok & vol_ok
        short_ok = short_ok & vol_ok

    none = pd.DataFrame(False, index=c.index, columns=c.columns)
    entries = bt.next_bar(long_ok) if direction != "short" else none
    short_entries = bt.next_bar(short_ok) if direction != "long" else none
    intraday = panel.bar != "1d"  # session guards are meaningless on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(c.index, panel.tz, skip_open=skip_open, skip_close=skip_close)
        entries = entries & ok.to_numpy()[:, None]
        short_entries = short_entries & ok.to_numpy()[:, None]

    exits = bt.exit_after(entries, horizon)
    short_exits = bt.exit_after(short_entries, horizon)
    if intraday and eod_exit:
        end = bt.session_end(c.index, panel.tz).to_numpy()[:, None]
        exits = exits | end
        short_exits = short_exits | end

    out = {"entries": entries, "exits": exits,
           "short_entries": short_entries, "short_exits": short_exits,
           "sl_stop": (sl_mult * rng / c).clip(lower=1e-4)}
    if tp_mult:
        out["tp_stop"] = (tp_mult * rng / c).clip(lower=1e-4)
    out["cash_frac"] = cash_frac
    return out


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the midline is faster than the regime filter
        (``band_window < trend_window``, unless the trend gate is off), a
        stop exists (``sl_mult > 0``), and the ablation control keeps some
        edge (``pattern != "any"`` or ``stretch_mult > 0`` — never fire on
        every bar).
    """
    band = params.get("band_window", 120)
    trend = params.get("trend_window", 960)
    stretch = params.get("stretch_mult", 3.0)
    return (
        (trend == 0 or band < trend)
        and params.get("sl_mult", 2.0) > 0
        and (params.get("pattern", "hammer") != "any" or stretch > 0)
    )


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    band_window: int = 120,
    atr_window: int = 30,
    stretch_mult: float = 3.0,
    trend_window: int = 960,
    vol_mult: float = 0.0,
    profile_days: int = 10,
    tz: str = "America/New_York",
    **_,
) -> dict:
    """Per-symbol overlays: the stretch band, the trend SMA and the volume gate.

    Args:
        close: One symbol's adjusted close (band drawn from a close-to-close
            range proxy — signals use the true high/low ATR, which close
            tracks closely).
        volume: One symbol's volume.
        band_window: Midline SMA lookback in bars.
        atr_window: ATR smoothing window in bars.
        stretch_mult: Band distance in ATRs (0 = gate off, bands omitted).
        trend_window: Regime SMA lookback (0 = gate off, line omitted).
        vol_mult: Participation multiple (drawn on the volume panel).
        profile_days: Volume-profile lookback sessions.
        tz: The panel's exchange timezone (volume-profile grouping).
        **_: Extra params ignored.

    Returns:
        ``{"price", "momentum", "volume"}`` overlay series: the stretch band and
        trend SMA on price, the close-vs-trend regime gap on momentum, and the
        expected-volume trigger on volume.
    """
    mid = bt.sma(close, band_window)
    rng = close.diff().abs().ewm(alpha=1.0 / atr_window, min_periods=atr_window).mean()
    price = {}
    if stretch_mult:
        price[f"SMA {band_window} midline"] = mid
        price[f"lower band (-{stretch_mult:g} ATR)"] = mid - stretch_mult * rng
        price[f"upper band (+{stretch_mult:g} ATR)"] = mid + stretch_mult * rng
    momentum = {}
    if trend_window:
        trend = bt.sma(close, trend_window)
        price[f"SMA {trend_window} trend"] = trend
        momentum[f"close vs SMA {trend_window}"] = close / trend - 1.0
    vol = {}
    if vol_mult:
        base = bt.volume_profile(volume.to_frame("v"), tz, days=profile_days)["v"]
        vol = {"expected volume": base, f"vol trigger ({vol_mult:g}x)": vol_mult * base}
    return {"price": price, "momentum": momentum, "volume": vol}
