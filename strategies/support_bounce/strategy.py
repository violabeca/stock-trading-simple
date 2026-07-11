"""Support bounce, intraday: anticipatory longs where a tested support level holds.

Thesis: a horizontal level that recently stopped the selling tends to stop it
again. The premise has real evidence behind it — published support/resistance
levels predicted intraday trend interruptions far better than chance (Osler,
FRBNY 2000), and such levels coincide with resting-depth peaks in the limit
order book (Kavajecz & Odders-White, RFS 2004) — but the naive execution is
exactly what the cautionary literature kills: tight fixed-percent stops under a
mean-reversion entry systematically sell the bounce (Kaminski & Lo), resting
limit bids fill disproportionately when price is crashing *through* the level
(adverse selection), and 1-minute churn dies to costs. So this strategy buys
*near* a tested support in a non-bear regime, sizes everything in ATRs, keeps
the stop below the level's noise band (never tight under entry), and lets the
optimizer arbitrate the contentious choices: anticipatory touch vs confirmed
bounce, fixed vs trailing stop, flat-at-close vs overnight carry.

Signals:

* **Support** — rolling min of ``low`` over ``level_window`` minutes, shifted
  one bar. The **touch band** sits ``touch_tol`` ATRs above it (a "touch" =
  the level got tested but held); the **entry zone** ends ``entry_dist`` ATRs
  above it (close must be near the level, not chasing).
* **Tested level** (``min_touches``, 0 = off) — the level needs that many
  *distinct prior touch episodes* (consecutive touch bars count once) inside
  the level window. A freshly broken-down level starts untested, which keeps
  the strategy from buying every retest of a failed floor.
* **Regime** (``ema_fast``/``ema_slow``/``regime_dz``) — EMAs with a dead
  zone classify bull / bear / sideways. Longs need bull (or sideways with
  ``allow_sideways``); bear never enters and optionally exits
  (``exit_on_bear``).
* **Trigger at ``tf_minutes``** — the candle-interval knob, derived from 1m
  bars: a dip = the lowest low of the trailing ``tf_minutes`` reached the
  touch band. ``entry_mode: touch`` enters on the dip itself — the closest
  no-look-ahead stand-in for a resting limit order (the engine fills at the
  *next bar's open*, ~1 minute after a real limit fill, never at the limit
  price). ``entry_mode: bounce`` additionally requires the current close back
  above the band — an N-minute reclaim/hammer evaluated every minute.
* **Exits** — ``sl_atr``-ATR stop (``sl_trail`` trails it behind the best
  price — the lock-in-profit variant; off = fixed), optional ``tp_mult``-ATR
  target, a ``max_hold``-minute time stop, the bear regime flip, and
  (intraday) ``eod_exit`` / session-edge entry blackouts.

Windows are **minutes**, converted to bars from ``panel.bar`` — the same
``signals()`` runs on 1m (shipped), 5m/1h/1d panels; an unmapped bar size
fails loudly. Engine stop semantics (vectorbt-verified, shared by every
strategy here): stop fractions anchor at the fill bar's close and activate
the next bar, evaluated intrabar against high/low; when one bar spans both
stop and target the STOP fills (pessimistic); a gap through a stop fills at
the stop level (optimistic for overnight holds — judge ``eod_exit: false``
configs with the notebook's gap scan). On a 1-minute bar the intrabar path is
unknown, so a tight bracket's win rate is not trustworthy — judge on the
time-stop / EOD exit population.
"""

from __future__ import annotations

import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

#: Minutes per bar for every panel this strategy can run on (390 = one RTH day).
_MIN_PER_BAR = {"1m": 1, "5m": 5, "1h": 60, "1d": 390}


def _bars(minutes: float, bar: str) -> int:
    """Convert a window expressed in minutes to a bar count for ``bar``.

    Args:
        minutes: Window length in minutes.
        bar: Panel bar size (``"1m"`` | ``"5m"`` | ``"1h"`` | ``"1d"``).

    Returns:
        The window in bars, at least 1.

    Raises:
        KeyError: If ``bar`` has no minutes-per-bar mapping (e.g. ``"15m"``).
    """
    return max(1, round(minutes / _MIN_PER_BAR[bar]))


def signals(
    panel: Panel,
    *,
    level_window: int = 780,
    min_touches: int = 2,
    touch_tol: float = 0.25,
    entry_dist: float = 0.75,
    entry_mode: str = "bounce",
    tf_minutes: int = 5,
    ema_fast: int = 240,
    ema_slow: int = 960,
    regime_dz: float = 0.002,
    allow_sideways: bool = True,
    atr_window: int = 30,
    sl_atr: float = 3.0,
    sl_trail: bool = False,
    tp_mult: float = 0.0,
    max_hold: int = 240,
    exit_on_bear: bool = True,
    eod_exit: bool = False,
    skip_open: int = 15,
    skip_close: int = 15,
) -> dict:
    """Build support-touch/bounce entries with ATR stops and regime/time exits.

    Args:
        panel: Price panel (uses open/high/low/close).
        level_window: Support lookback in minutes — the level is the rolling
            min of ``low`` over this window (shifted one bar), and the same
            window bounds the touch-episode count.
        min_touches: Distinct prior touch episodes the level needs before an
            entry may fire; ``0`` disables the tested-level filter.
        touch_tol: Touch-band width in ATRs above the level — defines both a
            historical "test" and the entry trigger's dip target.
        entry_dist: Entry-zone height in ATRs above the level; the close must
            sit inside it (near the level, not chasing the tape).
        entry_mode: ``"touch"`` (enter on the dip — anticipatory, closest to
            a resting limit order) or ``"bounce"`` (dip plus the close back
            above the touch band — confirmation against adverse selection).
        tf_minutes: Trigger granularity in minutes — the dip is the lowest low
            of this trailing window, so 5/15 emulate coarser candles on 1m
            data. Must not exceed ``skip_open`` (see ``valid``).
        ema_fast: Fast regime EMA span in minutes.
        ema_slow: Slow regime EMA span in minutes (must exceed ``ema_fast``).
        regime_dz: Regime dead zone as a fraction — bull needs
            ``fast > slow*(1+dz)``, bear ``fast < slow*(1-dz)``; between the
            two is sideways.
        allow_sideways: Enter in bull and sideways regimes; ``False`` = bull
            only. Bear never enters.
        atr_window: ATR smoothing window in minutes; sets every band/stop
            scale.
        sl_atr: Stop distance in ATRs at entry. Keep it beyond
            ``entry_dist + touch_tol`` so the stop sits below the level's
            noise band (enforced by ``valid``).
        sl_trail: ``True`` trails the stop behind the best price since entry
            (locks in profit, the user thesis); ``False`` keeps it fixed.
        tp_mult: Take-profit distance in ATRs; ``0`` = no target (run to the
            time stop / regime flip / session close).
        max_hold: Time stop in minutes held.
        exit_on_bear: Also exit when the regime flips to bear (next bar open).
        eod_exit: Force positions flat on each session's final bar (intraday
            only); ``False`` lets a bounce carry overnight.
        skip_open: No entries within the first N minutes of a session
            (intraday only).
        skip_close: No entries within the last N minutes of a session
            (intraday only).

    Returns:
        Dict with ``entries`` / ``exits``, ``sl_stop`` and, when enabled,
        ``sl_trail`` / ``tp_stop`` (stops as ATR fractions of price). Long
        only — no short keys. Exits fire on the stop/target, the ``max_hold``
        time stop, the bear flip and, intraday with ``eod_exit``, each
        session's final bar.
    """
    bar = panel.bar
    close = panel.close
    lw = _bars(level_window, bar)

    support = panel.low.rolling(lw, min_periods=lw).min().shift(1)
    rng = bt.atr(panel, _bars(atr_window, bar))
    band = support + touch_tol * rng
    zone_top = support + entry_dist * rng

    fast = close.ewm(span=_bars(ema_fast, bar), adjust=False,
                     min_periods=_bars(ema_fast, bar)).mean()
    slow = close.ewm(span=_bars(ema_slow, bar), adjust=False,
                     min_periods=_bars(ema_slow, bar)).mean()
    bull = fast > slow * (1.0 + regime_dz)
    bear = fast < slow * (1.0 - regime_dz)
    # NaN guard: during EMA warm-up neither bull nor bear is True, which would
    # otherwise read as "sideways" and open the gate on unwarmed data.
    regime_ok = (~bear & slow.notna()) if allow_sideways else bull

    dip = panel.low.rolling(_bars(tf_minutes, bar), min_periods=1).min() <= band
    raw = (close > support) & (close <= zone_top) & dip & regime_ok
    if min_touches:
        touched = (panel.low <= band) & (close > support)   # tested and held
        starts = touched & ~touched.shift(1, fill_value=False)
        prior = starts.astype(float).rolling(lw, min_periods=1).sum().shift(1)
        raw = raw & (prior >= min_touches)
    if entry_mode == "bounce":
        raw = raw & (close > band)                          # reclaimed the band

    entries = bt.next_bar(raw)
    intraday = bar != "1d"  # session guards are meaningless on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(close.index, panel.tz,
                             skip_open=skip_open, skip_close=skip_close)
        entries = entries & ok.to_numpy()[:, None]

    exits = bt.exit_after(entries, _bars(max_hold, bar))
    if exit_on_bear:
        exits = exits | bt.next_bar(bear)
    if intraday and eod_exit:
        exits = exits | bt.session_end(close.index, panel.tz).to_numpy()[:, None]

    out = {"entries": entries, "exits": exits,
           "sl_stop": (sl_atr * rng / close).clip(lower=1e-4)}
    if sl_trail:
        out["sl_trail"] = True
    if tp_mult:
        out["tp_stop"] = (tp_mult * rng / close).clip(lower=1e-4)
    return out


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the regime EMAs keep their roles (``ema_fast < ema_slow``),
        the touch band fits inside the entry zone (``touch_tol < entry_dist``
        — otherwise a bounce close can never sit above the band yet inside
        the zone), the stop clears the level's noise band even from a
        worst-case fill at the zone top (``sl_atr > entry_dist + touch_tol``,
        the Kaminski-Lo never-tight-under-a-reversion-entry rule), and the
        trigger window cannot straddle the session open
        (``tf_minutes <= skip_open``).
    """
    tol = params.get("touch_tol", 0.25)
    dist = params.get("entry_dist", 0.75)
    return (
        params.get("ema_fast", 240) < params.get("ema_slow", 960)
        and tol < dist
        and params.get("sl_atr", 3.0) > dist + tol
        and params.get("tf_minutes", 5) <= params.get("skip_open", 15)
    )


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    level_window: int = 780,
    touch_tol: float = 0.25,
    entry_dist: float = 0.75,
    atr_window: int = 30,
    ema_fast: int = 240,
    ema_slow: int = 960,
    bar: str = "1m",
    **_,
) -> dict:
    """Per-symbol overlays: the support level, its bands and the regime EMAs.

    Args:
        close: One symbol's adjusted close (level drawn from close as an
            approximation — signals use the true low, and a close-to-close
            range proxy stands in for the high/low ATR).
        volume: Unused (kept for the protocol signature).
        level_window: Support lookback in minutes.
        touch_tol: Touch-band width in ATRs.
        entry_dist: Entry-zone height in ATRs.
        atr_window: ATR smoothing window in minutes.
        ema_fast: Fast regime EMA span in minutes.
        ema_slow: Slow regime EMA span in minutes.
        bar: Bar size the minute windows are converted with (defaults to the
            shipped 1m config).
        **_: Extra params ignored.

    Returns:
        ``{"price", "momentum", "volume"}`` overlay series: the support line
        with its touch band and zone top plus the regime EMAs on price, and
        the fast-vs-slow regime gap on momentum.
    """
    lw = _bars(level_window, bar)
    support = close.rolling(lw, min_periods=lw).min().shift(1)
    aw = _bars(atr_window, bar)
    rng = close.diff().abs().ewm(alpha=1.0 / aw, min_periods=aw).mean()
    fast = close.ewm(span=_bars(ema_fast, bar), adjust=False).mean()
    slow = close.ewm(span=_bars(ema_slow, bar), adjust=False).mean()
    return {
        "price": {
            f"support (min low, {level_window}m)": support,
            f"touch band (+{touch_tol:g} ATR)": support + touch_tol * rng,
            f"zone top (+{entry_dist:g} ATR)": support + entry_dist * rng,
            f"EMA {ema_fast}m": fast,
            f"EMA {ema_slow}m": slow,
        },
        "momentum": {f"EMA {ema_fast}m vs {ema_slow}m": fast / slow - 1.0},
        "volume": {},
    }
