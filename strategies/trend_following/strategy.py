"""Trend following, long and short: enter at the confirmed start of a trend, then ride it
behind an ATR chandelier stop that follows the price and only ever tightens.

Thesis. Prices trend more often than a random walk would, in both directions (time-series
momentum: Moskowitz, Ooi & Pedersen 2012; on single US stocks, Wilcox & Crittenden 2005 and
Zarattini, Pagani & Wilcox 2025). A trend follower does not forecast: it waits until the price
has left its range decisively, joins the move, and lets a volatility-scaled stop decide when the
move is over. The payoff is asymmetric by design — many small losses on false starts, a few
large winners that run for months — so the entry filter's job is to cut false starts and the
stop's job is to cut losers early without choking winners.

The start of a trend, decided on the signal bar's close:

* **Breakout** — the close is above the highest high of the prior ``entry_window`` bars
  (long) or below the lowest low (short): the price has left its range (the Donchian /
  Turtle entry).
* **Directional strength** — Wilder's ADX over ``adx_window`` bars is at least ``adx_min``
  and the directional indicators agree (+DI above −DI for a long, below for a short): the
  move that produced the breakout is a directional one, not a noisy poke through the
  channel. ``adx_min: 0`` switches the confirmation off.

The exit — whichever comes first:

* **The ATR stop.** At entry it sits ``atr_mult`` ATRs from the signal bar's close. Every bar
  after, it is the chandelier: the best price since entry (highest high for a long, lowest
  low for a short) minus / plus ``atr_mult`` × the CURRENT ATR. It moves with the price,
  adjusts to the name's volatility as it changes, and is ratcheted — it never loosens, so an
  expanding ATR cannot hand back room already locked in. It is a **closing-basis** stop: a
  close through the level exits at the next open. The engine's intrabar ``sl_stop`` fixes
  its distance at entry and cannot follow a changing ATR, so the stop is walked here
  (:func:`_ride`); a bar that gaps past the level fills at that open, so gap risk is in the
  numbers.
* **An opposite start** — a confirmed start in the other direction closes the position; the
  new direction opens on a later start.

The book: ``max_positions`` equal slots in ONE shared account (engine portfolio mode). A
start opens a position at ``1 / max_positions`` of the book when the name is flat and a slot
is free. When more names start than slots are free, the **strongest relative trend** goes
first: the name whose 12-1 month return (``rs_window`` bars back to ``rs_skip`` bars back)
beats the average name's by the most (for a long) or trails it by the most (for a short) —
cross-sectional momentum (Jegadeesh & Titman 1993) deciding which of today's breakouts
deserve the book. The most recent month is skipped because one-month returns REVERSE
(Jegadeesh 1990): ranking on short horizons fitted 2022-23 and its best trials then lost
22-36%/yr against buy-and-hold in 2024 (README, v5). A start is ignored while the name
already holds a position in its direction. A position is left to drift until it exits —
never re-trimmed — and is sold whole. Fixed slots size every entry off the book's current
value, so the book never depends on the order the entries happened to arrive in (a
free-cash fraction does: its first entries take the book). Everything is next-bar-open
executed (:func:`trading_simple.backtest.next_bar`); there is no look-ahead.

Intraday bars (``trend_following_1m`` runs this module on 1-minute bars): no starts in the
first ``skip_open`` / last ``skip_close`` minutes of a session (:func:`bt.session_mask`), and
with ``eod_exit`` every position is closed before the session's final bar, so nothing is held
overnight. All three are ignored on daily bars. Every window is in bars, so on 1-minute data
it is in minutes.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

_DIRECTIONS = ("long", "short", "both")


def signals(
    panel: Panel,
    *,
    entry_window: int = 9,
    adx_window: int = 14,
    adx_min: float = 34.0,
    atr_window: int = 37,
    atr_mult: float = 14.0,
    direction: str = "both",
    max_positions: int = 20,
    rs_window: int = 252,
    rs_skip: int = 21,
    skip_open: int = 0,
    skip_close: int = 0,
    eod_exit: bool = False,
) -> dict:
    """Build the execution-timed book: one equal slot per open trend position.

    Args:
        panel: Price panel (uses high/low/close).
        entry_window: Donchian lookback in bars; a start is a close beyond the prior
            ``entry_window``-bar extreme.
        adx_window: Wilder smoothing window of the ADX and the directional indicators.
        adx_min: ADX floor a breakout needs to count as a start (0 = no confirmation).
        atr_window: ATR window of the stop (Wilder smoothing).
        atr_mult: Stop distance in ATRs — at entry and, trailing, behind the best price.
        direction: ``"long"`` | ``"short"`` | ``"both"``.
        max_positions: Equal slots in the book; each open position targets
            ``±1 / max_positions`` of it.
        rs_window: Relative-strength lookback in bars — when starts outnumber free slots,
            the strongest relative trend is admitted first.
        rs_skip: Most recent bars left out of the relative strength (short-term reversal).
        skip_open: Intraday only — no starts within the first N minutes of a session.
        skip_close: Intraday only — no starts within the last N minutes of a session.
        eod_exit: Intraday only — close every position before the session's final bar
            (the engine sells at that bar's open), so nothing is held overnight.

    Returns:
        ``{"weights"}`` — +1/−1/0 per name and bar (long / short / flat) divided by
        ``max_positions``, next-bar executed. Portfolio mode: a name trades only when its own
        target changes, so a held position drifts untouched until it exits.
    """
    parts = _compose(panel, entry_window=entry_window, adx_window=adx_window,
                     adx_min=adx_min, atr_window=atr_window, atr_mult=atr_mult,
                     direction=direction, max_positions=max_positions,
                     rs_window=rs_window, rs_skip=rs_skip, skip_open=skip_open,
                     skip_close=skip_close, eod_exit=eod_exit)
    return {"weights": bt.next_bar(parts["position"]).astype(float) / max_positions}


def adx(panel: Panel, window: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wilder's Average Directional Index with its two directional indicators.

    +DM is today's high minus yesterday's (when it beats the low's fall and is positive),
    −DM yesterday's low minus today's; each is smoothed (Wilder EWM, like :func:`bt.atr`) and
    divided by the ATR to give +DI / −DI. ADX is the smoothed
    ``|+DI − −DI| / (+DI + −DI)``: how directional the movement is, whichever the direction.

    Args:
        panel: Price panel (uses high/low/close).
        window: Smoothing window in bars.

    Returns:
        ``(adx, plus_di, minus_di)``, each in 0–100, NaN during the warm-up.
    """
    up, down = panel.high.diff(), -panel.low.diff()
    known = up.notna() & down.notna()
    plus = up.where((up > down) & (up > 0), 0.0).where(known)
    minus = down.where((down > up) & (down > 0), 0.0).where(known)

    def smooth(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.ewm(alpha=1.0 / window, min_periods=window).mean()

    atr = bt.atr(panel, window)
    plus_di, minus_di = 100.0 * smooth(plus) / atr, 100.0 * smooth(minus) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return smooth(dx), plus_di, minus_di


def _ride(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, atr: pd.DataFrame,
          long_start: pd.DataFrame, short_start: pd.DataFrame, rank: pd.DataFrame, *,
          atr_mult: float, max_positions: int, flat: np.ndarray | None = None) -> dict:
    """Walk the bars once: fill free slots with starts, trail each position's stop.

    Vectorized across symbols, sequential in time (a stop depends on the trade it protects,
    and a slot on every other trade). Everything happens on the signal clock: bar ``t``'s
    decisions use bar ``t``'s close, high, low, ATR and rank, and execute at ``t + 1``'s open
    once :func:`signals` shifts them. Exits come first, so a slot freed on a bar can take a
    start on the same bar (the engine sells before it buys within a bar).

    Args:
        close: Adjusted close.
        high: Adjusted high.
        low: Adjusted low.
        atr: The stop's ATR.
        long_start: Confirmed long starts (signal clock).
        short_start: Confirmed short starts (signal clock).
        rank: Admission priority when starts outnumber free slots (higher goes first).
        atr_mult: Stop distance in ATRs.
        max_positions: Slots in the book.
        flat: Optional per-bar mask: every position closes on a True bar and none opens
            (the intraday end-of-session exit). ``None`` = never.

    Returns:
        ``{"position", "stop", "long_exit", "short_exit"}`` — the position held after each
        bar's decisions (+1 / −1 / 0), the stop level in force (NaN when flat) and the two
        exit masks, all ``(bar x symbol)`` frames on ``close``'s axes.
    """
    c, h, lo, a, r = (f.to_numpy(dtype=float) for f in (close, high, low, atr, rank))
    ls, ss = long_start.to_numpy(dtype=bool), short_start.to_numpy(dtype=bool)
    n, m = c.shape
    flat = np.zeros(n, dtype=bool) if flat is None else flat
    pos = np.zeros(m, dtype=np.int8)
    best = np.full(m, np.nan)                    # highest high / lowest low since entry
    stop = np.full(m, np.nan)
    long_exit, short_exit = np.zeros((n, m), bool), np.zeros((n, m), bool)
    position, level = np.zeros((n, m), np.int8), np.full((n, m), np.nan)
    for t in range(n):
        is_l, is_s = pos == 1, pos == -1
        # Trail: the chandelier on today's bar, ratcheted (fmax/fmin also skip a NaN bar).
        best = np.where(is_l, np.fmax(best, h[t]), np.where(is_s, np.fmin(best, lo[t]), best))
        stop = np.where(is_l, np.fmax(stop, best - atr_mult * a[t]),
                        np.where(is_s, np.fmin(stop, best + atr_mult * a[t]), stop))
        out_l = is_l & ((c[t] < stop) | ss[t] | flat[t])   # a close through the stop, an
        out_s = is_s & ((c[t] > stop) | ls[t] | flat[t])   # opposite start, or the session end
        long_exit[t], short_exit[t] = out_l, out_s
        # Starts on names flat BEFORE this bar, into the slots left after today's exits.
        new = (pos == 0) & (ls[t] ^ ss[t]) & ~flat[t]
        pos = np.where(out_l | out_s, 0, pos)
        free = max_positions - int(np.count_nonzero(pos))
        if int(new.sum()) > free:
            cand = np.flatnonzero(new)
            keep = cand[np.argsort(-r[t, cand], kind="stable")[:max(free, 0)]]
            new = np.zeros(m, dtype=bool)
            new[keep] = True
        new_l, new_s = new & ls[t], new & ss[t]
        pos = np.where(new_l, 1, np.where(new_s, -1, pos)).astype(np.int8)
        best = np.where(new, c[t], np.where(pos == 0, np.nan, best))
        stop = np.where(new_l, c[t] - atr_mult * a[t],
                        np.where(new_s, c[t] + atr_mult * a[t],
                                 np.where(pos == 0, np.nan, stop)))
        position[t], level[t] = pos, stop

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=close.index, columns=close.columns)

    return {"position": frame(position), "stop": frame(level),
            "long_exit": frame(long_exit), "short_exit": frame(short_exit)}


def _compose(panel: Panel, *, entry_window: int, adx_window: int, adx_min: float,
             atr_window: int, atr_mult: float, direction: str, max_positions: int,
             rs_window: int, rs_skip: int, skip_open: int = 0, skip_close: int = 0,
             eod_exit: bool = False, **_) -> dict:
    """Every intermediate the book rests on (shared by :func:`signals` and :func:`compose`).

    Args:
        panel: Price panel.
        (parameters): See :func:`signals`.
        **_: Ignored, so callers can splat a full parameter dict.

    Returns:
        ``{"upper", "lower", "adx", "plus_di", "minus_di", "atr", "rel_strength",
        "long_start", "short_start", "position", "stop", "long_exit", "short_exit"}`` on the
        signal clock (``rel_strength``: the 12-1 month return minus the average name's —
        the admission priority).
    """
    close = panel.close
    upper = panel.high.rolling(entry_window, min_periods=entry_window).max().shift(1)
    lower = panel.low.rolling(entry_window, min_periods=entry_window).min().shift(1)
    strength, plus_di, minus_di = adx(panel, adx_window)
    atr = bt.atr(panel, atr_window)
    ready = atr.notna() & strength.notna()
    strong = strength >= adx_min
    long_start = ready & (close > upper) & strong & (plus_di > minus_di) & (direction != "short")
    short_start = ready & (close < lower) & strong & (minus_di > plus_di) & (direction != "long")
    intraday, flat = panel.bar != "1d", None     # session guards mean nothing on daily bars
    if intraday and (skip_open or skip_close):
        ok = bt.session_mask(close.index, panel.tz, skip_open=skip_open,
                             skip_close=skip_close).to_numpy()[:, None]
        long_start, short_start = long_start & ok, short_start & ok
    if intraday and eod_exit:
        # Decided on the bar BEFORE the session's last one, so the engine sells at the last
        # bar's open (where backtest.session_end exits fill); nothing opens on either bar.
        last = bt.session_end(close.index, panel.tz)
        flat = (last | last.shift(-1, fill_value=False)).to_numpy()
    # Admission priority: the 12-1 month return over the average name's (a long wants the
    # most outperformance, a short the most underperformance).
    ret = close.shift(rs_skip) / close.shift(rs_window) - 1.0
    rel = ret.sub(ret.mean(axis=1), axis=0)
    ride = _ride(close, panel.high, panel.low, atr, long_start, short_start,
                 rel.where(long_start, -rel), atr_mult=atr_mult, max_positions=max_positions,
                 flat=flat)
    return {"upper": upper, "lower": lower, "adx": strength, "plus_di": plus_di,
            "minus_di": minus_di, "atr": atr, "rel_strength": rel, "long_start": long_start,
            "short_start": short_start, **ride}


_DEFAULTS = {name: p.default for name, p in inspect.signature(signals).parameters.items()
             if p.default is not inspect.Parameter.empty}


def compose(panel: Panel, **params) -> dict:
    """Expose every intermediate — starts, stop line, position — for notebooks and tests.

    Shares its whole computation with :func:`signals`, so a chart of the stop line shows
    exactly the stop the backtest trades. Missing parameters default to :func:`signals`' own.

    Args:
        panel: The panel the backtest runs on.
        **params: Any :func:`signals` parameters to override.

    Returns:
        The :func:`_compose` result, all on the signal clock (unshifted).
    """
    return _compose(panel, **{**_DEFAULTS, **params})


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the direction is known, every window spans at least two bars, the ADX
        floor is a level ADX can take, the stop has a positive distance, the book has at
        least one slot, the relative-strength window reaches past the bars it skips and the
        session blackouts are not negative.
    """
    return (params.get("direction", "both") in _DIRECTIONS
            and params.get("entry_window", 9) >= 2
            and params.get("adx_window", 14) >= 2
            and params.get("atr_window", 37) >= 2
            and 0.0 <= params.get("adx_min", 34.0) < 100.0
            and params.get("atr_mult", 14.0) > 0.0
            and params.get("max_positions", 20) >= 1
            and 0 <= params.get("rs_skip", 21) < params.get("rs_window", 252)
            and params.get("skip_open", 0) >= 0 and params.get("skip_close", 0) >= 0)


def indicators(close: pd.Series, volume: pd.Series, *, entry_window: int = 9, **_) -> dict:
    """Per-symbol overlays: the Donchian channel a start must close beyond.

    Args:
        close: One symbol's adjusted close (the channel is drawn from closes as an
            approximation — signals use highs and lows; the stop line itself needs the
            whole panel, so notebooks read it from :func:`compose`).
        volume: Unused (kept for the protocol signature).
        entry_window: Donchian lookback in bars.
        **_: Extra params ignored.

    Returns:
        ``{"price": {...}, "volume": {}}`` overlay series.
    """
    return {
        "price": {f"{entry_window}-bar high": close.rolling(entry_window).max().shift(1),
                  f"{entry_window}-bar low": close.rolling(entry_window).min().shift(1)},
        "volume": {},
    }
